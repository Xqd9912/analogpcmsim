"""Relax structures (cell + positions) with MACE-MP-0 medium, in parallel.

env: mace.  usage: python relax.py --cifs <dir_or_glob>... --out results.jsonl [--workers 5]
Each line of the output jsonl: file, natoms, E0/E (eV/atom), converged, steps, min_dist, vol_ratio.
Relaxed structures are written next to the jsonl in <out_stem>_relaxed/.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

FMAX = 0.05
STEPS = 500
MIN_DIST_INVALID = 0.5  # Angstrom; generated structures with closer atoms are not relaxed

_calc = None


def _init(threads):
    global _calc
    import torch
    torch.set_num_threads(threads)
    from mace.calculators import mace_mp
    _calc = mace_mp(model="medium", default_dtype="float64", device="cpu")


def _relax(args):
    path, outdir = args
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    rec = {"file": os.path.basename(path)}
    try:
        s = Structure.from_file(path)
        n = len(s)
        dm = s.distance_matrix + np.eye(n) * 1e3
        rec.update(natoms=n, min_dist0=float(dm.min()), vol0=s.volume / n)
        if dm.min() < MIN_DIST_INVALID:
            rec.update(status="invalid_min_dist")
            return rec
        atoms = AseAtomsAdaptor.get_atoms(s)
        atoms.calc = _calc
        rec["E0"] = atoms.get_potential_energy() / n
        opt = FIRE(FrechetCellFilter(atoms), logfile=None)
        conv = opt.run(fmax=FMAX, steps=STEPS)
        sr = AseAtomsAdaptor.get_structure(atoms)
        dm = sr.distance_matrix + np.eye(n) * 1e3
        rec.update(E=atoms.get_potential_energy() / n, converged=bool(conv), steps=opt.get_number_of_steps(),
                   min_dist=float(dm.min()), vol_ratio=sr.volume / s.volume, status="ok")
        sr.to(filename=os.path.join(outdir, os.path.basename(path)))
    except Exception as e:  # noqa: BLE001
        rec.update(status="error", error=repr(e)[:300])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    files = []
    for c in a.cifs:
        files += sorted(glob.glob(os.path.join(c, "*.cif"))) if os.path.isdir(c) else sorted(glob.glob(c))
    out = Path(a.out)
    outdir = out.parent / f"{out.stem}_relaxed"
    outdir.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["file"] for l in out.read_text().splitlines() if l.strip()}
    todo = [(f, str(outdir)) for f in files if os.path.basename(f) not in done]
    print(f"{len(files)} files, {len(todo)} to relax", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.threads,)) as pool, open(out, "a") as fh:
        for i, rec in enumerate(pool.imap_unordered(_relax, todo)):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
