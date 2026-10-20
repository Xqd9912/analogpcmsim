"""D3 diagnostic: project predicted CHGCARs onto the plane-wave sphere |G| <= 2 sqrt(ENCUT / (hbar^2/2m)).

usage: python scripts/run_d3_lowpass.py --ids mp-32 ... [--src_tag gpu_dual] [--encut 520]
Reads results/D3/density/<src_tag>_<id>/chgcar/*.CHGCAR, writes results/D3/density/<src_tag>_lp_<id>/chgcar/ (same
file names, augmentation unchanged) and <src_tag>_lp_<id>/lowpass.csv. ENCUT = 520 eV (MP / our VASP setting).
Motivation: the analog prediction error has a white (grid-point independent) read-noise floor; the DFT density has no
components beyond 2 G_wf, so the projection removes most of that noise without touching the physical density.
"""
import argparse
from pathlib import Path

import pandas as pd

from pcmsim.density.chgcar import lowpass_chgcar

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--src_tag", default="gpu_dual")
    ap.add_argument("--encut", type=float, default=520.0)
    ap.add_argument("--factor", type=float, default=2.0)
    a = ap.parse_args()
    for mid in a.ids:
        src = ROOT / "results/D3/density" / f"{a.src_tag}_{mid}" / "chgcar"
        dst = ROOT / "results/D3/density" / f"{a.src_tag}_lp_{mid}" / "chgcar"
        dst.mkdir(parents=True, exist_ok=True)
        rows = []
        for f in sorted(src.glob(f"{mid}_*.CHGCAR")):
            info = lowpass_chgcar(f, dst / f.name, a.encut, a.factor)
            rows.append({"file": f.name, **{k: info[k] for k in ("gcut_inv_A", "removed_rel_l1", "electrons_after")}})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(dst.parent / "lowpass.csv", index=False)


if __name__ == "__main__":
    main()
