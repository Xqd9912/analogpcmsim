"""DiffCSP CSP evaluation on MP-20 subsets: match rate and normalized RMSE (DiffCSP convention).

StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3), one sample per structure.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from .diffusion import lattices_to_params_shape

ROOT = Path(__file__).resolve().parents[3]


def gt_structure(cif: str) -> Structure:
    c = Structure.from_str(cif, fmt="cif").get_reduced_structure()
    return Structure(Lattice.from_parameters(*c.lattice.parameters), c.species, c.frac_coords)


def load_subset(subset_csv: str | Path, split_csv: str | Path, n: int | None = None) -> list[tuple[str, Structure]]:
    """[(material_id, ground-truth structure)] for the ids listed in subset_csv (in file order)."""
    ids = pd.read_csv(ROOT / subset_csv).material_id.tolist()
    df = pd.read_csv(ROOT / split_csv).set_index("material_id")
    ids = ids[:n] if n else ids
    return [(m, gt_structure(df.loc[m, "cif"])) for m in ids]


def batch_inputs(structs: list[Structure], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nat = torch.tensor([len(s) for s in structs], device=device)
    types = torch.tensor(np.concatenate([s.atomic_numbers for s in structs]), device=device)
    n2g = torch.repeat_interleave(torch.arange(len(structs), device=device), nat)
    return nat, types, n2g


def to_structures(out: dict, species: list[list]) -> list[Structure | None]:
    lengths, angles = lattices_to_params_shape(out["lattices"].cpu())
    fx = out["frac_coords"].cpu().split(out["num_atoms"].cpu().tolist())
    res = []
    for i, sp in enumerate(species):
        try:
            res.append(Structure(Lattice.from_parameters(*lengths[i].tolist(), *angles[i].tolist()), sp,
                                 fx[i].numpy()))
        except Exception:  # degenerate lattice
            res.append(None)
    return res


def match(preds: list[Structure | None], gts: list[Structure]) -> list[dict]:
    sm = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    rows = []
    for p, g in zip(preds, gts):
        rms = None
        if p is not None and not 0.5 < p.volume / len(p) < 200:
            p = None  # degenerate lattice: StructureMatcher can hang on these; counted as no match
        if p is not None:
            try:
                r = sm.get_rms_dist(p, g)
                rms = None if r is None else float(r[0])
            except Exception:
                rms = None
        rows.append({"match": rms is not None, "rmse": rms})
    return rows


def sample_subset(model, items, batch_size: int, sample_seed: int, device, sampler=None) -> tuple[list, float, list]:
    """Sample one structure per item in batches; the sampler RNG is reseeded per batch for reproducibility.

    sampler(model, nat, types, n2g) -> out dict; default: the original DiffCSP PC sampler.
    Returns (structures, wall seconds, per-batch analog fractions).
    """
    preds, fracs, t0 = [], [], time.time()
    for b0 in range(0, len(items), batch_size):
        chunk = items[b0:b0 + batch_size]
        structs = [s for _, s in chunk]
        nat, types, n2g = batch_inputs(structs, device)
        torch.manual_seed(sample_seed * 100003 + b0)
        out = sampler(model, nat, types, n2g) if sampler else model.sample(nat, types, n2g, step_lr=1e-5)
        fracs.append(out.get("analog_fraction"))
        preds += to_structures(out, [s.species for s in structs])
    return preds, time.time() - t0, fracs


def summarize(rows: list[dict]) -> dict:
    m = [r for r in rows if r["match"]]
    return {"n": len(rows), "match_rate": len(m) / max(len(rows), 1),
            "rmse": float(np.mean([r["rmse"] for r in m])) if m else None}
