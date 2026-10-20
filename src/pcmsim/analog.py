"""Analog in-memory MVM simulation on differential PCM pairs (inference only).

Per mapped weight matrix: DAC (8 b) -> 256x256 tiles -> analog partial sums with per-read
relative read noise -> ADC (8 b, calibrated full scale) -> fp32 digital accumulation
-> alpha_j / dG -> global drift-compensation scale -> + digital bias.

Two independent random streams (CLAUDE.md section 4):
    programming noise: sampled once per chip instance in instantiate_chip(model, chip_seed); static
    read noise:        resampled at every forward from a dedicated generator (set_read_seed)
The floating-point master weights are never modified.
Conductances in uS.
"""
from __future__ import annotations

import contextlib
import math
from typing import Callable

import torch
from torch import nn

from . import mapping
from .device import PCMConfig


def _quant(x: torch.Tensor, rng: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric uniform quantizer with full scale rng (broadcastable to x)."""
    n = 2 ** (bits - 1) - 1
    scale = rng / n
    return torch.round(x / scale).clamp(-n, n) * scale


class AnalogMatrix(nn.Module):
    """One crossbar-mapped weight matrix W (out, in); forward computes x @ W^T (no bias).

    Rows are split into blocks of <= `tile` rows (one crossbar tile per block and 256-column group).
    `row_groups` (sizes of consecutive input-feature groups) makes blocks group-aligned, so that
    differently scaled input features never share a tile.
    cfg.block_alpha / cfg.block_dac (options; False = CLAUDE.md nominal: one alpha_j per output column,
    one DAC full scale per layer): alpha_j and/or DAC full scale per row block; each block's partial sum
    is rescaled digitally before the fp32 accumulation. cfg.block_scale=True switches both on.
    cfg.n_slices=2 (dual-slice ablation, decision point 3): a second differential pair per weight stores the
    residual w - w1, where w1 is the *programmed* (read-back) slice-1 weight including its programming error
    (write-verify residual programming). Slice 2 has its own alpha_j, independent programming / read noise
    and its own ADC; both slices share the DAC input and are summed digitally. Drift is not applied to slice 2.
    """

    def __init__(self, weight: torch.Tensor, cfg: PCMConfig, row_groups: list[int] | None = None):
        super().__init__()
        self.cfg = cfg
        self.out_features, self.in_features = weight.shape
        self.weight = nn.Parameter(weight.detach().clone().float(), requires_grad=False)  # digital master
        self.register_buffer("states", torch.tensor(sorted(cfg.states), dtype=torch.float32))
        self.dg = float(self.states[-1] - self.states[0])
        t = cfg.tile
        groups = row_groups or [self.in_features]
        assert sum(groups) == self.in_features, (groups, self.in_features)
        self.blocks, s0 = [], 0
        for g in groups:
            for b in range(s0, s0 + g, t):
                self.blocks.append((b, min(b + t, s0 + g)))
            s0 += g
        both = bool(getattr(cfg, "block_scale", False))
        self.block_alpha = both or bool(getattr(cfg, "block_alpha", False))
        self.block_dac = both or bool(getattr(cfg, "block_dac", False))
        nb = len(self.blocks)
        self.n_in_tiles = nb
        self.n_out_tiles = math.ceil(self.out_features / t)
        z = lambda *s: torch.zeros(*s)
        self.register_buffer("alpha", z(nb, self.out_features))  # rows identical unless block_scale
        self.register_buffer("g_pos", z(self.out_features, self.in_features))
        self.register_buffer("g_neg", z(self.out_features, self.in_features))
        self.register_buffer("nu_pos", z(self.out_features, self.in_features))
        self.register_buffer("nu_neg", z(self.out_features, self.in_features))
        self.register_buffer("dac_range", z(nb))  # entries identical unless block_scale
        self.register_buffer("adc_range", z(nb, self.n_out_tiles))
        self.register_buffer("gdc_scale", torch.tensor(1.0))
        self.register_buffer("gdc_ref", torch.tensor(0.0))
        self.n_slices = int(getattr(cfg, "n_slices", 1))
        if self.n_slices == 2:
            self.register_buffer("alpha2", z(nb, self.out_features))
            self.register_buffer("g_pos2", z(self.out_features, self.in_features))
            self.register_buffer("g_neg2", z(self.out_features, self.in_features))
            self.register_buffer("adc_range2", z(nb, self.n_out_tiles))
        elif self.n_slices != 1:
            raise ValueError("n_slices must be 1 or 2")
        self.programmed = False
        self.digital = False
        self.read_noise = True
        self.use_dac = True
        self.use_adc = True
        self.t_eval = cfg.t0
        self.read_gen: torch.Generator | None = None
        self.count_macs = False
        self.n_macs = 0
        self._mode = "run"  # run | calib_range | calib_ref | calib_drift
        self._stat, self._stat_n = 0.0, 0
        self._cache = None
        self._samples: dict = {}
        self._cal_gen: torch.Generator | None = None

    @property
    def use_dac_adc(self) -> bool:
        return self.use_dac and self.use_adc

    @use_dac_adc.setter
    def use_dac_adc(self, flag: bool) -> None:
        self.use_dac = self.use_adc = bool(flag)

    # ---------------------------------------------------------------- programming
    @torch.no_grad()
    def program(self, gen: torch.Generator) -> None:
        """alpha search + static programming noise for one chip instance (gen: CPU generator)."""
        w = self.weight.detach()
        gp, gn = torch.empty_like(w), torch.empty_like(w)
        if not self.block_alpha:
            self.alpha.copy_(mapping.search_alpha(w, self.states, self.cfg.alpha_grid).expand_as(self.alpha))
        for k, (b0, b1) in enumerate(self.blocks):
            if self.block_alpha:
                self.alpha[k] = mapping.search_alpha(w[:, b0:b1], self.states, self.cfg.alpha_grid)
            gp[:, b0:b1], gn[:, b0:b1] = mapping.target_conductance(w[:, b0:b1], self.alpha[k], self.states)
        noise = lambda: (torch.randn(gp.shape, generator=gen) * self.cfg.sigma_prog).to(gp.device)
        self.g_pos.copy_((gp + noise()).clamp_min(0.0))
        self.g_neg.copy_((gn + noise()).clamp_min(0.0))
        nud = lambda: (self.cfg.nu + self.cfg.nu_std * torch.randn(gp.shape, generator=gen)).to(gp.device)
        self.nu_pos.copy_(nud())
        self.nu_neg.copy_(nud())
        if self.n_slices == 2:  # residual of the programmed slice 1 (write-verify)
            r = w - self.alpha_full() / self.dg * (self.g_pos - self.g_neg)
            gp2, gn2 = torch.empty_like(w), torch.empty_like(w)
            if not self.block_alpha:
                self.alpha2.copy_(mapping.search_alpha(r, self.states, self.cfg.alpha_grid).expand_as(self.alpha2))
            for k, (b0, b1) in enumerate(self.blocks):
                if self.block_alpha:
                    self.alpha2[k] = mapping.search_alpha(r[:, b0:b1], self.states, self.cfg.alpha_grid)
                gp2[:, b0:b1], gn2[:, b0:b1] = mapping.target_conductance(r[:, b0:b1], self.alpha2[k], self.states)
            self.g_pos2.copy_((gp2 + noise()).clamp_min(0.0))
            self.g_neg2.copy_((gn2 + noise()).clamp_min(0.0))
        self.gdc_scale.fill_(1.0)
        self.t_eval = self.cfg.t0
        self.programmed = True
        self._cache = None

    def drifted(self) -> tuple[torch.Tensor, torch.Tensor]:
        ratio = max(self.t_eval / self.cfg.t0, 1.0)
        if ratio == 1.0:
            return self.g_pos, self.g_neg
        lr = math.log(ratio)
        return self.g_pos * torch.exp(-self.nu_pos * lr), self.g_neg * torch.exp(-self.nu_neg * lr)

    def alpha_full(self, alpha: torch.Tensor | None = None) -> torch.Tensor:
        """Per-element scale (out, in): alpha of the block each input row belongs to."""
        alpha = self.alpha if alpha is None else alpha
        return torch.cat([alpha[k].unsqueeze(1).expand(-1, b1 - b0)
                          for k, (b0, b1) in enumerate(self.blocks)], dim=1)

    def effective_weight(self) -> torch.Tensor:
        """Programmed (drifted) weight, without read noise / DAC / ADC / drift compensation."""
        gp, gn = self.drifted()
        w = self.alpha_full() / self.dg * (gp - gn)
        if self.n_slices == 2:
            w = w + self.alpha_full(self.alpha2) / self.dg * (self.g_pos2 - self.g_neg2)
        return w

    def _conductances(self):
        """[(G+ - G-, G+^2 + G-^2, alpha, adc_range, adc_key)] per slice."""
        if self._cache is None or self._cache[0] != self.t_eval or self._cache[1][0][0].device != self.g_pos.device:
            gp, gn = self.drifted()
            sl = [(gp - gn, gp * gp + gn * gn, self.alpha, self.adc_range, "adc")]
            if self.n_slices == 2:
                sl.append((self.g_pos2 - self.g_neg2, self.g_pos2 ** 2 + self.g_neg2 ** 2, self.alpha2,
                           self.adc_range2, "adc2"))
            self._cache = (self.t_eval, sl)
        return self._cache[1]

    # --------------------------------------------------------------- calibration
    _SAMPLES_PER_CALL = 20000
    _SAMPLES_MAX = 400000

    def _collect(self, key, v: torch.Tensor) -> None:
        """Keep a random subsample of calibration magnitudes for the full-scale search."""
        if self.cfg.range_method == "max":
            return
        v = v.reshape(-1)
        if v.numel() > self._SAMPLES_PER_CALL:
            if self._cal_gen is None or self._cal_gen.device != v.device:
                self._cal_gen = torch.Generator(device=v.device).manual_seed(0)
            idx = torch.randint(v.numel(), (self._SAMPLES_PER_CALL,), generator=self._cal_gen, device=v.device)
            v = v[idx]
        self._samples.setdefault(key, []).append(v.float())

    @torch.no_grad()
    def finalize_ranges(self) -> None:
        """Set DAC / ADC full scales from the calibration samples (range_method='mse': minimum
        quantization + clipping MSE over full-scale candidates max * logspace(-3, 0, 121))."""
        if self.cfg.range_method == "max" or not self._samples:
            self._samples = {}
            return

        def best(vals, bits):
            v = torch.cat(vals)
            if v.numel() > self._SAMPLES_MAX:
                v = v[torch.randperm(v.numel(), device=v.device)[: self._SAMPLES_MAX]]
            vmax = float(v.max())
            if vmax <= 0:
                return vmax
            n = 2 ** (bits - 1) - 1
            cand = vmax * torch.logspace(-3, 0, 121, device=v.device)
            err = torch.stack([((torch.round(v / (r / n)).clamp(max=n) * (r / n) - v) ** 2).mean() for r in cand])
            return float(cand[err.argmin()])

        for key, vals in self._samples.items():
            if key[0] == "dac":
                r = best(vals, self.cfg.dac_bits)
                if self.block_dac:
                    self.dac_range[key[1]] = r
                else:
                    self.dac_range.fill_(r)
            elif key[0] == "adc2":
                self.adc_range2[key[1], key[2]] = best(vals, self.cfg.adc_bits)
            else:
                self.adc_range[key[1], key[2]] = best(vals, self.cfg.adc_bits)
        self._samples = {}

    # ------------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x = x.reshape(-1, self.in_features)
        if self.count_macs:  # crossbar MACs (each slice is a separate array MVM)
            self.n_macs += x.shape[0] * self.in_features * self.out_features * (1 if self.digital else self.n_slices)
        if self.digital:
            return (x @ self.weight.t()).reshape(*lead, self.out_features)
        if not self.programmed:
            raise RuntimeError("AnalogMatrix not programmed: call instantiate_chip(model, chip_seed) first")
        mode, t = self._mode, self.cfg.tile
        if mode == "calib_range":
            if self.cfg.dac_scaling != "dynamic":
                for k, (b0, b1) in enumerate(self.blocks):
                    v = x[:, b0:b1].detach().abs()
                    self.dac_range[k] = torch.maximum(self.dac_range[k], v.max())
                    self._collect(("dac", k if self.block_dac else 0), v)
        slices = self._conductances()
        noisy = self.read_noise and mode == "run" and self.cfg.read_rel > 0
        y = None
        dynamic = self.cfg.dac_scaling == "dynamic"
        for k, (b0, b1) in enumerate(self.blocks):
            xs = x[:, b0:b1]
            if dynamic:  # per-input-vector abs-max normalisation (digital), undone after the ADC
                sc = xs.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
                xs = xs / sc
            if mode != "calib_range" and self.use_dac:
                xs = _quant(xs, torch.ones((), device=xs.device) if dynamic else
                            self.dac_range[k].clamp_min(1e-30), self.cfg.dac_bits)
            for gd, gsq, alpha, adc_range, akey in slices:
                part = xs @ gd[:, b0:b1].t()
                if noisy:
                    # independent relative read noise on every device and read:
                    # var(partial_j) = sum_i x_i^2 r^2 (G+_ji^2 + G-_ji^2)
                    std = self.cfg.read_rel * torch.sqrt((xs * xs) @ gsq[:, b0:b1].t() + 1e-30)
                    part = part + std * torch.randn(part.shape, generator=self.read_gen, device=part.device,
                                                    dtype=part.dtype)
                if mode == "calib_range":
                    pa = part.detach().abs()
                    for j in range(self.n_out_tiles):
                        v = pa[:, j * t:(j + 1) * t]
                        adc_range[k, j] = torch.maximum(adc_range[k, j], v.max())
                        self._collect((akey, k, j), v)
                elif self.use_adc:
                    rng = adc_range[k].repeat_interleave(t)[: self.out_features].clamp_min(1e-30)
                    part = _quant(part, rng, self.cfg.adc_bits)
                part = part * (alpha[k] / self.dg)
                if dynamic:
                    part = part * sc
                y = part if y is None else y + part
        if mode in ("calib_ref", "calib_drift"):
            self._stat += float(y.detach().abs().mean())
            self._stat_n += 1
        y = y * self.gdc_scale
        return y.reshape(*lead, self.out_features)


# ============================================================== layer wrappers
class AnalogLinear(nn.Module):
    """Replacement for nn.Linear; the bias stays digital."""

    def __init__(self, lin: nn.Linear, cfg: PCMConfig, row_groups: list[int] | None = None):
        super().__init__()
        self.mat = AnalogMatrix(lin.weight, cfg, row_groups)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        y = self.mat(x)
        return y if self.bias is None else y + self.bias


class AnalogFCLayer(nn.Module):
    """Replacement for one e3nn.nn.FullyConnectedNet layer (radial MLP).

    e3nn: x @ (W / sqrt(h_in var_in)) -> act -> * sqrt(var_out)   (with act)
          x @ (W / sqrt(h_in var_in / var_out))                   (without act)
    The constant normalization is folded into the mapped matrix.
    """

    def __init__(self, layer, cfg: PCMConfig):
        super().__init__()
        h_in = layer.weight.shape[0]
        if layer.act is not None:
            w = layer.weight / (h_in * layer.var_in) ** 0.5
        else:
            w = layer.weight / (h_in * layer.var_in / layer.var_out) ** 0.5
        self.mat = AnalogMatrix(w.t().contiguous(), cfg)
        self.act = layer.act
        self.var_out = layer.var_out

    def forward(self, x):
        y = self.mat(x)
        if self.act is not None:
            y = self.act(y) * self.var_out ** 0.5
        return y


class _IrrepPathLinear(nn.Module):
    """Equivariant linear map built from per-irrep-path crossbar matrices.

    paths: list of (i_in, i_out, coefficient, W (mul_in, mul_out)). Each path owns one crossbar
    of shape (mul_out, mul_in); the 2l+1 m-components are fed time-multiplexed, so they share
    the same programmed conductances but get independent read noise.
    """

    def __init__(self, irreps_in, irreps_out, paths, cfg: PCMConfig):
        super().__init__()
        self.irreps_in, self.irreps_out = irreps_in, irreps_out
        self.in_slices = [(s.start, s.stop) for s in irreps_in.slices()]
        self.mats = nn.ModuleList([AnalogMatrix((c * w).t().contiguous(), cfg) for _, _, c, w in paths])
        self.paths = [(i, o) for i, o, _, _ in paths]

    def forward(self, x):
        lead = x.shape[:-1]
        x = x.reshape(-1, self.irreps_in.dim)
        b = x.shape[0]
        outs = [x.new_zeros(b, mul, ir.dim) for mul, ir in self.irreps_out]
        for mat, (i_in, i_out) in zip(self.mats, self.paths):
            mul_in, ir = self.irreps_in[i_in]
            s0, s1 = self.in_slices[i_in]
            xi = x[:, s0:s1].reshape(b, mul_in, ir.dim).transpose(1, 2)  # (b, m, mul_in)
            outs[i_out] = outs[i_out] + mat(xi).transpose(1, 2)
        return torch.cat([o.reshape(b, -1) for o in outs], dim=-1).reshape(*lead, self.irreps_out.dim)


class AnalogO3Linear(_IrrepPathLinear):
    """Replacement for e3nn.o3.Linear with internal shared weights (f_in=None)."""

    def __init__(self, lin, cfg: PCMConfig):
        if not lin.internal_weights or getattr(lin, "f_in", None) is not None:
            raise ValueError("only o3.Linear with internal shared weights and f_in=None is supported")
        ins_list = [ins for ins in lin.instructions if 0 not in ins.path_shape]
        paths, biases = [], []
        w_off = b_off = 0
        flat_w = lin.weight.detach() if lin.weight_numel > 0 else None
        for ins in ins_list:
            n = math.prod(ins.path_shape)
            if ins.i_in == -1:
                biases.append((ins.i_out, float(ins.path_weight), b_off, n))
                b_off += n
                continue
            w = flat_w[w_off:w_off + n].reshape(ins.path_shape)
            w_off += n
            if lin.irreps_in[ins.i_in].dim == 0 or lin.irreps_out[ins.i_out].dim == 0:
                continue
            paths.append((ins.i_in, ins.i_out, float(ins.path_weight), w))
        super().__init__(lin.irreps_in, lin.irreps_out, paths, cfg)
        self.bias_ins = biases
        self.bias = (nn.Parameter(lin.bias.detach().clone(), requires_grad=False)
                     if lin.bias_numel > 0 else None)
        self.out_slices = [(s.start, s.stop) for s in lin.irreps_out.slices()]

    def forward(self, x):
        y = super().forward(x)
        for i_out, pw, off, n in self.bias_ins:
            s0, s1 = self.out_slices[i_out]
            y[..., s0:s1] = y[..., s0:s1] + pw * self.bias[off:off + n]
        return y


@torch.no_grad()
def _fctp_path_coeffs(tp) -> list[float]:
    """Scalar coefficient of each weighted path of a FullyConnectedTensorProduct whose second
    input is 1x0e (=1): out[w, m] = c * sum_u W[u, 0, w] x[u, m]. Measured by probing."""
    coeffs = []
    dev = next(tp.parameters()).device if any(True for _ in tp.parameters()) else "cpu"
    in_sl = list(tp.irreps_in1.slices())
    out_sl = list(tp.irreps_out.slices())
    for k, (ins_i, ins, _) in enumerate(tp.weight_views(yield_instruction=True)):
        wvec = torch.zeros(tp.weight_numel, device=dev)
        for kk, (_, _, v) in enumerate(tp.weight_views(weight=wvec, yield_instruction=True)):
            if kk == k:
                v[0, 0, 0] = 1.0
        x = torch.zeros(1, tp.irreps_in1.dim, device=dev)
        x[0, in_sl[ins.i_in1].start] = 1.0
        out = tp(x, torch.ones(1, 1, device=dev), wvec)
        coeffs.append(float(out[0, out_sl[ins.i_out].start]))
    return coeffs


class AnalogFCTPLinear(_IrrepPathLinear):
    """Replacement for FullyConnectedTensorProduct(x, 1x0e, out) with constant attribute 1.

    With the attribute fixed to ones this product is an equivariant linear map (one weight matrix
    per irrep path), e.g. ChargE3Net's sc / lin1 / lin2 / lin3.
    """

    def __init__(self, tp, cfg: PCMConfig):
        from e3nn import o3
        if tp.irreps_in2 != o3.Irreps("1x0e") or not tp.internal_weights:
            raise ValueError("only FullyConnectedTensorProduct(x, 1x0e, out) with internal weights is supported")
        coeffs = _fctp_path_coeffs(tp)
        paths = []
        for c, (_, ins, w) in zip(coeffs, tp.weight_views(yield_instruction=True)):
            assert ins.connection_mode == "uvw" and w.shape[1] == 1
            paths.append((ins.i_in1, ins.i_out, c, w.detach()[:, 0, :]))
        super().__init__(tp.irreps_in1, tp.irreps_out, paths, cfg)
        self.register_buffer("output_mask", tp.output_mask.clone())

    def forward(self, x, attr=None):  # attr is the constant 1x0e input; ignored by construction
        return super().forward(x)


# ===================================================================== model API
def _kind(m) -> str | None:
    if isinstance(m, nn.Linear):
        return "linear"
    try:
        from e3nn import o3
        from e3nn.nn._fc import _Layer
    except ImportError:
        return None
    if isinstance(m, o3.Linear) and m.internal_weights and m.weight_numel > 0:
        return "o3_linear"
    if isinstance(m, o3.FullyConnectedTensorProduct) and m.irreps_in2 == o3.Irreps("1x0e") and m.internal_weights:
        return "fctp_linear"
    if isinstance(m, _Layer):
        return "fc_layer"
    return None


_WRAP = {"linear": AnalogLinear, "o3_linear": AnalogO3Linear, "fctp_linear": AnalogFCTPLinear,
         "fc_layer": AnalogFCLayer}


def convert_to_analog(model: nn.Module, cfg: PCMConfig,
                      include: Callable[[str, nn.Module], bool] | None = None,
                      row_groups: Callable[[str, nn.Module], list[int] | None] | None = None) -> list[tuple[str, str]]:
    """Replace mappable submodules selected by include(name, module); returns [(name, kind)].

    row_groups(name, module) may return input-feature group sizes for nn.Linear layers (group-aligned tiles).
    """
    include = include or (lambda n, m: True)
    targets = [(n, _kind(m)) for n, m in model.named_modules() if _kind(m) and include(n, m)]
    for name, kind in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, attr)
        p = next(old.parameters(), None)
        groups = row_groups(name, old) if (row_groups and kind == "linear") else None
        new = AnalogLinear(old, cfg, groups) if kind == "linear" else _WRAP[kind](old, cfg)
        if p is not None:
            new.to(p.device)
        setattr(parent, attr, new)
    return targets


def analog_matrices(model: nn.Module) -> list[AnalogMatrix]:
    return [m for m in model.modules() if isinstance(m, AnalogMatrix)]


def instantiate_chip(model: nn.Module, chip_seed: int) -> None:
    """Program every analog matrix (alpha search + static programming noise) for one chip instance."""
    gen = torch.Generator().manual_seed(int(chip_seed))
    for m in analog_matrices(model):
        m.program(gen)


def set_read_seed(model: nn.Module, read_seed: int, device: torch.device | str | None = None) -> None:
    """One dedicated read-noise stream shared by all matrices (independent of the sampler's RNG)."""
    mats = analog_matrices(model)
    if not mats:
        return
    dev = device or mats[0].g_pos.device
    gen = torch.Generator(device=dev).manual_seed(int(read_seed))
    for m in mats:
        m.read_gen = gen


def set_digital(model: nn.Module, flag: bool = True) -> None:
    for m in analog_matrices(model):
        m.digital = flag


@contextlib.contextmanager
def digital_mode(model: nn.Module, flag: bool = True):
    mats = analog_matrices(model)
    old = [m.digital for m in mats]
    for m in mats:
        m.digital = flag
    try:
        yield
    finally:
        for m, o in zip(mats, old):
            m.digital = o


def _set_mode(mats, mode):
    for m in mats:
        m._mode = mode
        m._stat, m._stat_n = 0.0, 0


@torch.no_grad()
def calibrate(model: nn.Module, run_fn: Callable[[], None]) -> None:
    """DAC/ADC full scales from calibration data at t0, then the drift-compensation reference.

    run_fn() must run the model forward on the calibration batch(es).
    """
    mats = analog_matrices(model)
    for m in mats:
        m.dac_range.zero_()
        m.adc_range.zero_()
        if m.n_slices == 2:
            m.adc_range2.zero_()
        m.gdc_scale.fill_(1.0)
        m.t_eval = m.cfg.t0
    _set_mode(mats, "calib_range")
    run_fn()
    for m in mats:
        m.finalize_ranges()
    _set_mode(mats, "calib_ref")
    run_fn()
    for m in mats:
        m.gdc_ref.fill_(m._stat / max(m._stat_n, 1))
    _set_mode(mats, "run")


@torch.no_grad()
def set_drift_time(model: nn.Module, t: float, run_fn: Callable[[], None] | None = None,
                   iters: int = 3, compensate: bool = True) -> None:
    """Move all matrices to time t (s) and recompute the per-layer global drift compensation."""
    mats = analog_matrices(model)
    for m in mats:
        m.t_eval = t
        m.gdc_scale.fill_(1.0)
    if not compensate or run_fn is None:
        return
    for _ in range(iters):  # fixed point: upstream layers already compensated
        _set_mode(mats, "calib_drift")
        run_fn()
        for m in mats:
            if m._stat_n:
                m.gdc_scale.fill_(float(m.gdc_ref) / max(m._stat / m._stat_n, 1e-30))
    _set_mode(mats, "run")


@contextlib.contextmanager
def count_macs(model: nn.Module):
    """Count crossbar MACs (rows x in x out, per m-component for irrep paths) inside the block."""
    mats = analog_matrices(model)
    for m in mats:
        m.count_macs, m.n_macs = True, 0
    res = {}
    try:
        yield res
    finally:
        res["total"] = sum(m.n_macs for m in mats)
        for m in mats:
            m.count_macs = False
