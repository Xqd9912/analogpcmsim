"""Differential-pair weight mapping (CLAUDE.md section 5). Mapping logic lives only here.

For a weight matrix W (out, in) and per-output-column scale alpha_j:
    dG = Gmax - Gmin
    w >= 0: G+ = Q(Gmin + |w|/alpha_j * dG), G- = Gmin;  w < 0: swapped;  w = 0: both Gmin
    Q snaps to the nearest state of S;  w_q = alpha_j / dG * (G+ - G-)
Conductances in uS.
"""
from __future__ import annotations

import torch


def snap_to_states(g: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Nearest state of the sorted state set `states` (uS) for every entry of g (uS)."""
    mids = 0.5 * (states[1:] + states[:-1])
    return states[torch.bucketize(g, mids)]


def target_conductance(w: torch.Tensor, alpha: torch.Tensor, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Noise-free (G+, G-) in uS for weights w (out, in) with per-row scale alpha (out,)."""
    g_min, g_max = states[0], states[-1]
    a = alpha.clamp_min(1e-30).unsqueeze(-1)
    g_q = snap_to_states(g_min + (w.abs() / a).clamp(max=1.0) * (g_max - g_min), states)
    g0 = torch.full_like(g_q, float(g_min))
    pos = w >= 0
    return torch.where(pos, g_q, g0), torch.where(pos, g0, g_q)


def conductance_to_weight(g_pos: torch.Tensor, g_neg: torch.Tensor, alpha: torch.Tensor,
                          states: torch.Tensor) -> torch.Tensor:
    """Effective weight alpha_j / dG * (G+ - G-)."""
    return alpha.unsqueeze(-1) / (states[-1] - states[0]) * (g_pos - g_neg)


def quantized_weight(w: torch.Tensor, alpha: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    return conductance_to_weight(*target_conductance(w, alpha, states), alpha, states)


@torch.no_grad()
def search_alpha(w: torch.Tensor, states: torch.Tensor, n_grid: int = 91, return_curve: bool = False):
    """Per-row alpha_j = r * max|w_j|, r in linspace(0.1, 1, n_grid), minimizing the quantization MSE."""
    ratios = torch.linspace(0.1, 1.0, n_grid, device=w.device)
    wmax = w.abs().amax(dim=1).clamp_min(1e-30)
    mse = torch.empty(n_grid, w.shape[0], device=w.device)
    for i, r in enumerate(ratios):
        mse[i] = ((quantized_weight(w, r * wmax, states) - w) ** 2).mean(dim=1)
    alpha = ratios[mse.argmin(dim=0)] * wmax
    alpha[w.abs().amax(dim=1) == 0] = 1.0
    return (alpha, ratios, mse) if return_curve else alpha
