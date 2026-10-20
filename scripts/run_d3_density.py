"""D3 step 1: ChargE3Net digital / analog density on the VASP fine grid of GST-series structures.

For each structure: read the converged SCF CHGCAR (DFT reference rho*V on the fine grid), predict the density on
every grid point with the digital model and with one programmed analog model per chip, report NMAE vs DFT and vs
digital, save a 2D slice for figures and (optionally) write VASP CHGCARs (predicted grid + atomic augmentation
occupancies) for the non-self-consistent ICHARG=11 runs.
usage: python scripts/run_d3_density.py --ids mp-1224375 mp-938 [--chips 0 1 2] [--device cuda:0] [--write_chgcar]
Outputs: results/D3/density/<tag>/{metrics.json, per_structure.csv, slices/<id>.npz, chgcar/<id>_<model>.CHGCAR}
"""
import argparse
import glob
import json
import subprocess
import time
from pathlib import Path

import ase.io
import numpy as np
import pandas as pd
import torch
from pymatgen.io.ase import AseAtomsAdaptor

from pcmsim.analog import calibrate, instantiate_chip, set_read_seed
from pcmsim.convert import to_analog_charge3net
from pcmsim.density import charge3net as c3
from pcmsim.density.chgcar import read_augmentation_text, read_chgcar, write_chgcar
from pcmsim.device import PCMConfig

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src", "scripts", "configs"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return h + ("-dirty" if dirty else "")


def to_atoms(sd: dict):
    import ase
    return ase.Atoms(symbols=sd["species"], scaled_positions=sd["frac_coords"], cell=sd["lattice"], pbc=True)


def slice_index(atoms, grid, axis: int = 1) -> int:
    """Grid index along `axis` of the plane through the first atom."""
    return int(round(atoms.get_scaled_positions()[0, axis] * grid[axis])) % grid[axis]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--chgcar_dir", default="data/chgcar/gst")
    ap.add_argument("--chips", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--device_overrides", default="{}", help="JSON dict of PCMConfig overrides")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_probes", type=int, default=5000)
    ap.add_argument("--tag", default="nominal")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--write_chgcar", action="store_true")
    ap.add_argument("--read_seed", type=int, default=3000)
    ap.add_argument("--analog_scope", default="all", choices=["all", "probe"],
                    help="probe: atom network digital (runs once per structure), probe network on the array")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    out = ROOT / "results" / "D3" / "density" / a.tag
    (out / "slices").mkdir(parents=True, exist_ok=True)
    over = json.loads(a.device_overrides)

    # calibration set = ChargE3Net validation structures (same as D1)
    cal_atoms = [ase.io.read(f) for f in sorted(glob.glob(str(ROOT / "data/structures/chg3_val_calib/*.cif")))]
    rng = np.random.default_rng(0)
    cal_pts = [rng.random((2000, 3)) @ np.array(at.get_cell()) for at in cal_atoms]
    models = {"digital": c3.build_model(dev)}
    pcm = PCMConfig.from_yaml(ROOT / "configs/device_nominal.yaml", **over)
    for chip in a.chips:
        m = c3.build_model(dev)
        to_analog_charge3net(m, pcm, a.analog_scope)
        instantiate_chip(m, chip)
        m.to(dev)
        calibrate(m, lambda: [c3.predict_points(m, at, p, a.batch_probes, dev) for at, p in zip(cal_atoms, cal_pts)])
        set_read_seed(m, a.read_seed + chip, dev)
        models[f"analog_chip{chip}"] = m

    rows = []
    for mid in a.ids:
        f = ROOT / a.chgcar_dir / f"{mid}_default.CHGCAR"
        sd, total, _ = read_chgcar(f)
        atoms = to_atoms(sd)
        vol = atoms.get_volume()
        rho = total / vol
        grid = rho.shape
        pts = c3.grid_points(atoms, grid)
        t0 = time.time()
        batches = c3.prepare_batches(atoms, pts, a.batch_probes)
        row = {"material_id": mid, "formula": atoms.get_chemical_formula(), "natoms": len(atoms),
               "nelect_dft_grid": float(total.mean()),
               "grid": "x".join(map(str, grid)), "n_points": int(np.prod(grid)), "wall_graph_s": time.time() - t0}
        j = slice_index(atoms, grid)
        sl = {"dft": rho[:, j, :].astype(np.float32), "cell": np.array(atoms.get_cell()), "j": j,
              "frac_positions": atoms.get_scaled_positions(), "numbers": atoms.get_atomic_numbers()}
        preds = {}
        for name, m in models.items():
            t0 = time.time()
            p = c3.predict_batches(m, batches, dev).reshape(grid)
            row[f"wall_{name}_s"] = time.time() - t0
            row[f"nmae_{name}"] = c3.nmae(p, rho)
            preds[name] = p
            sl[name] = p[:, j, :].astype(np.float32)
            if name != "digital":
                row[f"nmae_vs_digital_{name}"] = c3.nmae(p, preds["digital"])
            if a.write_chgcar:
                aug = read_augmentation_text(ROOT / a.chgcar_dir / f"{mid}_atomaug.CHGCAR")
                (out / "chgcar").mkdir(exist_ok=True)
                info = write_chgcar(out / "chgcar" / f"{mid}_{name}.CHGCAR", sd, p * vol, units="rhoV",
                                    n_electrons=float(total.mean()), comment=f"ChargE3Net {name} {mid}",
                                    aug_text=aug)  # renormalised to the DFT electron count
                row[f"electrons_before_norm_{name}"] = info["electrons_before"]
        del batches
        np.savez_compressed(out / "slices" / f"{mid}.npz", **sl)
        rows.append(row)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()
                          if k.startswith(("material_id", "n_points", "nmae"))}), flush=True)
        pd.DataFrame(rows).to_csv(out / "per_structure.csv", index=False)
    meta = {"git_commit": git_commit(), "device": str(dev), "chips": a.chips, "pcm_config": pcm.__dict__,
            "analog_scope": a.analog_scope,
            "ids": a.ids, "chgcar_dir": a.chgcar_dir}
    (out / "metrics.json").write_text(json.dumps(meta, indent=1, default=str))
    print("->", out)


if __name__ == "__main__":
    main()
