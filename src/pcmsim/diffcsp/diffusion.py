"""Port of DiffCSP CSPDiffusion (CSP mode) inference to modern PyTorch, no Lightning / Hydra.

Source: jiaor17/DiffCSP @ 7121d15  diffcsp/pl_modules/diffusion.py, diff_utils.py, common/data_utils.py.
The predictor-corrector sampling loop reproduces the original call order of the RNG
(randn lattice, rand coords, then per step: corrector randn_like(l), randn_like(x),
predictor randn_like(l), randn_like(x)), so with the same seed and the same weights the
trajectory matches the original implementation.

`decoder_fn` hook: `sample()` calls `self.decoder(...)` for every network evaluation, and
passes `t` and `substep` ('corrector'/'predictor') to an optional callback so that the
noise-budget sampler (plan appendix A) can switch analog/digital per sub-step later.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from .cspnet import CSPNet


# ---------------------------------------------------------------- schedules
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-((x + T * i) ** 2) / 2 / sigma**2)
    return p_


def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (x + T * i) / sigma**2 * torch.exp(-((x + T * i) ** 2) / 2 / sigma**2)
    return p_ / p_wrapped_normal(x, sigma, N, T)


def sigma_norm(sigma, T=1.0, sn=10000):
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T=T)
    return (normal_**2).mean(dim=0)


class BetaScheduler(nn.Module):
    def __init__(self, timesteps, scheduler_mode="cosine"):
        super().__init__()
        assert scheduler_mode == "cosine"
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        sigmas = torch.zeros_like(betas)
        sigmas[1:] = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        sigmas = torch.sqrt(sigmas)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sigmas", sigmas)


class SigmaScheduler(nn.Module):
    def __init__(self, timesteps, sigma_begin=0.01, sigma_end=1.0, compute_norm=False):
        super().__init__()
        self.timesteps = timesteps
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        sigmas = torch.FloatTensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), timesteps)))
        # sigmas_norm is a Monte-Carlo estimate in the original; it is overwritten by the checkpoint buffer.
        sn = sigma_norm(sigmas) if compute_norm else torch.ones_like(sigmas)
        self.register_buffer("sigmas", torch.cat([torch.zeros([1]), sigmas], dim=0))
        self.register_buffer("sigmas_norm", torch.cat([torch.ones([1]), sn], dim=0))


class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


# ---------------------------------------------------------------- lattice ops
def lattice_params_to_matrix_torch(lengths, angles):
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)
    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)
    z = torch.zeros(lengths.size(0), device=lengths.device, dtype=lengths.dtype)
    vector_a = torch.stack([lengths[:, 0] * sins[:, 1], z, lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack([z, z, lengths[:, 2]], dim=1)
    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def lattices_to_params_shape(lattices):
    lengths = torch.sqrt(torch.sum(lattices**2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[..., i] = torch.clamp(
            torch.sum(lattices[..., j, :] * lattices[..., k, :], dim=-1) / (lengths[..., j] * lengths[..., k]),
            -1.0,
            1.0,
        )
    angles = torch.arccos(angles) * 180.0 / np.pi
    return lengths, angles


# ---------------------------------------------------------------- model
class CSPDiffusion(nn.Module):
    """Inference-only CSPDiffusion. hparams: the `model:` section of DiffCSP hparams.yaml."""

    def __init__(self, decoder_kwargs, timesteps=1000, time_dim=256, latent_dim=0,
                 sigma_begin=0.005, sigma_end=0.5, cost_coord=1.0, cost_lattice=1.0):
        super().__init__()
        self.decoder = CSPNet(**{**decoder_kwargs, "latent_dim": latent_dim + time_dim})
        self.beta_scheduler = BetaScheduler(timesteps, "cosine")
        self.sigma_scheduler = SigmaScheduler(timesteps, sigma_begin, sigma_end)
        self.time_dim = time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(time_dim)
        self.keep_lattice = cost_lattice < 1e-5
        self.keep_coords = cost_coord < 1e-5

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(self, num_atoms, atom_types, node2graph, step_lr=1e-5, timesteps=None,
               callback=None, return_traj=False, show_progress=False):
        """Predictor-corrector CSP sampling (original DiffCSP algorithm).

        num_atoms: (B,) long; atom_types: (N,) long (Z); node2graph: (N,) long.
        timesteps: if given and < T, uses a *strided* schedule (smoke-test only; not the original
                   algorithm). None -> the original 1000-step loop.
        """
        dev = self.device
        batch_size = num_atoms.shape[0]
        N = atom_types.shape[0]
        l_T, x_T = torch.randn([batch_size, 3, 3]).to(dev), torch.rand([N, 3]).to(dev)
        T = self.beta_scheduler.timesteps
        if timesteps is None or timesteps >= T:
            ts = list(range(T, 0, -1))
        else:
            ts = [int(v) for v in np.unique(np.linspace(1, T, timesteps).round().astype(int))[::-1]]
        x_t, l_t = x_T % 1.0, l_T
        traj = [(x_t, l_t)] if return_traj else None
        it = ts
        if show_progress:
            from tqdm import tqdm
            it = tqdm(ts)
        for idx, t in enumerate(it):
            t_prev = ts[idx + 1] if idx + 1 < len(ts) else 0
            times = torch.full((batch_size,), t, device=dev)
            time_emb = self.time_embedding(times)
            if t_prev == t - 1:
                alphas = self.beta_scheduler.alphas[t]
            else:  # strided schedule: effective alpha between t and t_prev
                alphas = self.beta_scheduler.alphas_cumprod[t] / self.beta_scheduler.alphas_cumprod[t_prev]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            if t_prev == t - 1:
                sigmas = self.beta_scheduler.sigmas[t]
            else:
                b = 1 - alphas
                sigmas = torch.sqrt(b * (1 - self.beta_scheduler.alphas_cumprod[t_prev]) / (1 - alphas_cumprod))
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm_t = self.sigma_scheduler.sigmas_norm[t]
            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            # Corrector
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)
            if callback is not None:
                callback(t, "corrector")
            pred_l, pred_x = self.decoder(time_emb, atom_types, x_t, l_t, num_atoms, node2graph)
            pred_x = pred_x * torch.sqrt(sigma_norm_t)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_05 = l_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            adjacent_sigma_x = self.sigma_scheduler.sigmas[t_prev]
            step_size = sigma_x**2 - adjacent_sigma_x**2
            std_x = torch.sqrt((adjacent_sigma_x**2 * (sigma_x**2 - adjacent_sigma_x**2)) / (sigma_x**2))
            if callback is not None:
                callback(t, "predictor")
            pred_l, pred_x = self.decoder(time_emb, atom_types, x_t_minus_05, l_t_minus_05, num_atoms, node2graph)
            pred_x = pred_x * torch.sqrt(sigma_norm_t)
            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t

            x_t, l_t = x_t_minus_1 % 1.0, l_t_minus_1
            if return_traj:
                traj.append((x_t, l_t))
        out = {"num_atoms": num_atoms, "atom_types": atom_types, "frac_coords": x_t, "lattices": l_t}
        return (out, traj) if return_traj else out


def load_mp20_csp(ckpt_path, hparams_path=None, map_location="cpu"):
    """Build the port from DiffCSP hparams.yaml and load the Lightning checkpoint (strict)."""
    import yaml

    if hparams_path is None:
        import os
        hparams_path = os.path.join(os.path.dirname(ckpt_path), "hparams.yaml")
    with open(hparams_path) as f:
        hp = yaml.safe_load(f)["model"]
    dec = dict(hp["decoder"])
    dec.pop("_target_", None)
    dec["max_neighbors"] = hp["max_neighbors"]  # interpolations ${model.max_neighbors}
    dec["cutoff"] = hp["radius"]
    model = CSPDiffusion(
        decoder_kwargs={k: v for k, v in dec.items() if k != "latent_dim"},
        timesteps=hp["timesteps"],
        time_dim=hp["time_dim"],
        latent_dim=hp["latent_dim"],
        sigma_begin=hp["sigma_scheduler"]["sigma_begin"],
        sigma_end=hp["sigma_scheduler"]["sigma_end"],
        cost_coord=hp["cost_coord"],
        cost_lattice=hp["cost_lattice"],
    )
    ck = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    sd = ck["state_dict"]
    model.load_state_dict(sd, strict=True)  # raises on any missing/unexpected key or shape mismatch
    model.eval()
    return model, ck
