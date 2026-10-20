"""Model-specific on-array layer selection (CLAUDE.md section 6) and crossbar layer tables.

Only fixed-weight x input MVMs go on the array. Everything else (sinusoidal / spherical-harmonic
encodings, CG tensor products with edge attributes, LayerNorm, activations, gates, scatter
aggregation, embedding lookups, lattice ops) stays digital.
"""
from __future__ import annotations

import dataclasses
import re

from torch import nn

from .analog import (AnalogFCLayer, AnalogFCTPLinear, AnalogLinear, AnalogMatrix, AnalogO3Linear,
                     convert_to_analog)
from .device import PCMConfig

# DiffCSP CSPNet: CSPLayer edge_mlp / node_mlp linears, atom_latent_emb, coord / lattice output heads.
_DIFFCSP = re.compile(r"(^|\.)(csp_layer_\d+\.(edge_mlp|node_mlp)\.\d+|atom_latent_emb|coord_out|lattice_out)$")


def diffcsp_rule(name: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) and bool(_DIFFCSP.search(name))


def charge3net_rule(name: str, module: nn.Module) -> bool:
    """Probe-model radial MLP layers (fc) and sc / lin2 / lin3 (FCTP with constant 1x0e).

    Digital:
    - the readout o3.Linear (adopted 2026-10-17): it alone adds ~3.9 % NMAE when mapped, while carrying
      ~0.002 % of the crossbar MACs;
    - everything acting on atom features (adopted 2026-10-18): the atom model (once per structure) and the
      probe convolutions' lin1 (sender atom features, once per probe batch). On the array their read noise is
      sampled once and frozen into every grid point, so symmetry-equivalent atoms get different features
      (Ge2Sb2Te5: charge errors -0.037 vs -0.104 e on the two Ge) and NSCF band gaps of layered GST close.
      Together ~0.004 % of the crossbar MACs.
    """
    if name.startswith("atom_model") or name.endswith("readout"):
        return False
    return not (name.startswith("probe_model") and name.split(".")[-1] == "lin1")


def diffcsp_row_groups(name: str, module: nn.Module) -> list[int] | None:
    """Input-feature groups: edge_mlp.0 = [h_i, h_j, lattice inner products (9), distance encoding],
    node_mlp.0 = [h, aggregated edges], atom_latent_emb = [node embedding, time embedding]."""
    h = module.out_features
    if name.endswith("edge_mlp.0"):
        return [h, h, 9, module.in_features - 2 * h - 9]
    if name.endswith("node_mlp.0") or name.endswith("atom_latent_emb"):
        return [h, module.in_features - h]
    return None


def to_analog_diffcsp(model: nn.Module, cfg: PCMConfig) -> list[tuple[str, str]]:
    return convert_to_analog(model, cfg, diffcsp_rule, diffcsp_row_groups)


_C3_FC = re.compile(r"\.fc\.layer\d+(\.|$)")


def to_analog_charge3net(model: nn.Module, cfg: PCMConfig, scope: str = "all") -> list[tuple[str, str]]:
    """scope='all': atom and probe networks on the array; scope='probe': only the probe network (the atom network
    runs once per structure, ~1e-5 of the MACs, and stays digital).

    The radial-MLP (fc) crossbars use the calibration maximum as ADC full scale (range_method='max', adopted
    2026-10-18): the MSE full scale clips the tails of their partial sums, a biased error that removed ~0.025 e from
    the Ge sites of Ge2Sb2Te5 and closed the NSCF band gap (-255 meV -> +87 meV with 'max', digital +63 meV).
    """
    rule = charge3net_rule if scope != "probe" else (lambda n, m: n.startswith("probe_model") and charge3net_rule(n, m))
    conv = convert_to_analog(model, cfg, rule)
    cfg_max = dataclasses.replace(cfg, range_method="max")
    for name, m in model.named_modules():
        if isinstance(m, AnalogMatrix) and _C3_FC.search(name):
            m.cfg = cfg_max
    return conv


def layer_table(model: nn.Module) -> list[dict]:
    """One row per crossbar matrix: module name, wrapper kind, (in, out), number of 256x256 tiles."""
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, (AnalogLinear, AnalogFCLayer)):
            mats = [("", m.mat)]
        elif isinstance(m, (AnalogO3Linear, AnalogFCTPLinear)):
            mats = []
            for k, (mat, (i_in, i_out)) in enumerate(zip(m.mats, m.paths)):
                ir = m.irreps_in[i_in].ir
                mats.append((f"path{k}:{m.irreps_in[i_in]}->{m.irreps_out[i_out]}(m={ir.dim})", mat))
        else:
            continue
        for tag, mat in mats:
            rows.append({"module": name, "kind": type(m).__name__, "path": tag,
                         "in": mat.in_features, "out": mat.out_features,
                         "tiles": mat.n_in_tiles * mat.n_out_tiles})
    return rows


def n_analog_params(model: nn.Module) -> int:
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, AnalogMatrix))
