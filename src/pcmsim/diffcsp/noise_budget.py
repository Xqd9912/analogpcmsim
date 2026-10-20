"""Noise-budget predictor-corrector sampler for analog DiffCSP (plan appendix A, CLAUDE.md section 9).

Every PC sub-step injects Gaussian noise anyway. Read noise of the analog network output enters the state as
    corrector  (coords):  s_x = step_size_c * sigma_a_x(t)
    predictor  (coords):  s_x = step_size_p * sigma_a_x(t)
    predictor  (lattice): s_l = c0 * c1 * sigma_a_l(t)
so only the remainder  std_inject = sqrt(max(std^2 - s^2, 0))  is injected. A sub-step whose read-noise share
exceeds the noise it should inject (s^2 > std^2 for coords, or for the lattice in the predictor) is evaluated
digitally. Only read noise is budgeted; programming noise is a static error and is not.
sigma_a(t): RMS over output components of the per-component std across 16 forwards with independent read
noise, measured on a fixed chip and calibration batch at ~50 timesteps of a digital trajectory, interpolated.
"""
from __future__ import annotations

import numpy as np
import torch

from ..analog import digital_mode


@torch.no_grad()
def _digital_states(model, nat, types, n2g, t_keep: set[int], seed: int) -> dict:
    """Decoder inputs (time_emb, x_t, l_t) of a digital trajectory at the corrector sub-step of each t in t_keep."""
    dec = model.decoder
    states, cur = {}, {}

    def cb(t, sub):
        cur["t"], cur["sub"] = t, sub

    orig = dec.forward

    def rec(te, at, x, l, na, ng):
        if cur.get("sub") == "corrector" and cur.get("t") in t_keep:
            states[cur["t"]] = (te.clone(), x.clone(), l.clone())
        return orig(te, at, x, l, na, ng)

    dec.forward = rec
    try:
        with digital_mode(model):
            torch.manual_seed(seed)
            model.sample(nat, types, n2g, step_lr=1e-5, callback=cb)
    finally:
        dec.forward = orig
    return states


@torch.no_grad()
def calibrate_read_noise(model, nat, types, n2g, n_t: int = 50, n_rep: int = 16, seed: int = 777) -> dict:
    """sigma_a_x(t), sigma_a_l(t) on the current chip; returns {'t': array, 'sx': array, 'sl': array}."""
    T = model.beta_scheduler.timesteps
    t_grid = sorted({int(v) for v in np.linspace(1, T, n_t).round()})
    states = _digital_states(model, nat, types, n2g, set(t_grid), seed)
    sx, sl = [], []
    for t in t_grid:
        te, x, l = states[t]
        outs_l, outs_x = [], []
        for _ in range(n_rep):
            pl, px = model.decoder(te, types, x, l, nat, n2g)
            outs_l.append(pl)
            outs_x.append(px * torch.sqrt(model.sigma_scheduler.sigmas_norm[t]))
        sx.append(float(torch.stack(outs_x).var(0).mean().sqrt()))
        sl.append(float(torch.stack(outs_l).var(0).mean().sqrt()))
    return {"t": np.array(t_grid, dtype=float), "sx": np.array(sx), "sl": np.array(sl)}


def _interp(cal: dict, key: str, t: int) -> float:
    return float(np.interp(t, cal["t"], cal[key]))


@torch.no_grad()
def sample_noise_budget(model, num_atoms, atom_types, node2graph, cal: dict | None, step_lr: float = 1e-5,
                        log: list | None = None, digital_tail: int = 0, snapshots: dict | None = None) -> dict:
    """PC sampling with read-noise budgeting (cal=None: naive analog sampling, full injection).

    digital_tail: the last `digital_tail` timesteps (t <= digital_tail) run digitally with full noise
    injection, so the final refinement corrects static (quantization / programming) errors.

    RNG call order is identical to CSPDiffusion.sample, so with the same seed the sampler noise matches the
    digital baseline. `log` (optional) receives one dict per sub-step for figure 2b.
    snapshots (optional): {t: None, ...}; filled in place with the state (frac_coords, lattices) on the CPU at those
    t (T = initial noise, t - 1 after step t, 0 = final), for trajectory figures.
    """
    dev = model.device
    batch_size = num_atoms.shape[0]
    N = atom_types.shape[0]
    l_T, x_T = torch.randn([batch_size, 3, 3]).to(dev), torch.rand([N, 3]).to(dev)
    T = model.beta_scheduler.timesteps
    x_t, l_t = x_T % 1.0, l_T
    n_analog = n_total = 0

    def snap(t):
        if snapshots is not None and t in snapshots:
            snapshots[t] = (x_t.detach().cpu().clone(), l_t.detach().cpu().clone())

    snap(T)

    def run(analog: bool, te, x, l):
        nonlocal n_analog, n_total
        n_total += 1
        if analog:
            n_analog += 1
            return model.decoder(te, atom_types, x, l, num_atoms, node2graph)
        with digital_mode(model):
            return model.decoder(te, atom_types, x, l, num_atoms, node2graph)

    for t in range(T, 0, -1):
        times = torch.full((batch_size,), t, device=dev)
        te = model.time_embedding(times)
        alphas = model.beta_scheduler.alphas[t]
        alphas_cumprod = model.beta_scheduler.alphas_cumprod[t]
        sigmas = model.beta_scheduler.sigmas[t]
        sigma_x = model.sigma_scheduler.sigmas[t]
        sigma_norm_t = model.sigma_scheduler.sigmas_norm[t]
        c0 = 1.0 / torch.sqrt(alphas)
        c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
        sax = _interp(cal, "sx", t) if cal else 0.0
        sal = _interp(cal, "sl", t) if cal else 0.0
        tail = t <= digital_tail

        # ---- corrector (coords only; lattice unchanged)
        rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)  # kept for RNG order
        rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        step_size = step_lr * (sigma_x / model.sigma_scheduler.sigma_begin) ** 2
        std_x = float(torch.sqrt(2 * step_size)) if t > 1 else 0.0
        s_x = float(step_size) * sax
        analog = not tail and (cal is None or s_x ** 2 <= std_x ** 2)
        inj_x = std_x if cal is None or not analog else float(np.sqrt(max(std_x ** 2 - s_x ** 2, 0.0)))
        if log is not None:
            log.append({"t": t, "sub": "corrector", "std_x": std_x, "s_x": s_x, "analog": analog})
        pred_l, pred_x = run(analog, te, x_t, l_t)
        pred_x = pred_x * torch.sqrt(sigma_norm_t)
        x_05 = x_t - step_size * pred_x + inj_x * rand_x
        l_05 = l_t

        # ---- predictor
        rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
        rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        adj = model.sigma_scheduler.sigmas[t - 1]
        step_size = sigma_x ** 2 - adj ** 2
        std_x = float(torch.sqrt((adj ** 2 * (sigma_x ** 2 - adj ** 2)) / (sigma_x ** 2))) if t > 1 else 0.0
        std_l = float(sigmas) if t > 1 else 0.0
        s_x = float(step_size) * sax
        s_l = float(c0 * c1) * sal
        analog = not tail and (cal is None or (s_x ** 2 <= std_x ** 2 and s_l ** 2 <= std_l ** 2))
        if cal is None or not analog:
            inj_x, inj_l = std_x, std_l
        else:
            inj_x = float(np.sqrt(max(std_x ** 2 - s_x ** 2, 0.0)))
            inj_l = float(np.sqrt(max(std_l ** 2 - s_l ** 2, 0.0)))
        if log is not None:
            log.append({"t": t, "sub": "predictor", "std_x": std_x, "s_x": s_x, "std_l": std_l, "s_l": s_l,
                        "analog": analog})
        pred_l, pred_x = run(analog, te, x_05, l_05)
        pred_x = pred_x * torch.sqrt(sigma_norm_t)
        x_t = (x_05 - step_size * pred_x + inj_x * rand_x) % 1.0
        l_t = c0 * (l_05 - c1 * pred_l) + inj_l * rand_l
        snap(t - 1)

    return {"num_atoms": num_atoms, "atom_types": atom_types, "frac_coords": x_t, "lattices": l_t,
            "analog_fraction": n_analog / max(n_total, 1)}
