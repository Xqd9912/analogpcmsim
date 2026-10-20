"""D3: density averaged over independently programmed chips (multi-array averaging cancels programming noise).

usage: python scripts/run_d3_chipmean.py --ids mp-32 ... [--src_tag gpu_dual_am] [--chips 0 1 2] [--encut 520]
Reads results/D3/density/<src_tag>_<id>/chgcar/<id>_analog_chip<k>.CHGCAR (each already renormalised to the DFT electron
count, same atomic augmentation occupancies), averages the grids, projects onto |G| <= 2 sqrt(ENCUT/(hbar^2/2m)) and
writes results/D3/density/<src_tag>_mean<n>_lp_<id>/chgcar/<id>_analog_mean<n>.CHGCAR plus mean.csv (NMAE vs DFT
before projection for each chip and for the mean). Decision 2026-10-19: the chip-averaged density is the D3 nominal.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pcmsim.density.chgcar import HBAR2_2M, lowpass_sphere, read_augmentation_text, read_chgcar, write_chgcar

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--src_tag", default="gpu_dual_am")
    ap.add_argument("--chips", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--encut", type=float, default=520.0)
    a = ap.parse_args()
    nc = len(a.chips)
    gcut = 2 * np.sqrt(a.encut / HBAR2_2M)
    for mid in a.ids:
        src = ROOT / "results/D3/density" / f"{a.src_tag}_{mid}" / "chgcar"
        dst = ROOT / "results/D3/density" / f"{a.src_tag}_mean{nc}_lp_{mid}"
        (dst / "chgcar").mkdir(parents=True, exist_ok=True)
        _, ref, _ = read_chgcar(ROOT / "data/chgcar/gst" / f"{mid}_default.CHGCAR")
        nmae = lambda x: float(np.abs(x - ref).sum() / np.abs(ref).sum())
        grids, row = [], {"material_id": mid}
        for k in a.chips:
            sd, g, _ = read_chgcar(src / f"{mid}_analog_chip{k}.CHGCAR")
            grids.append(g)
            row[f"nmae_chip{k}"] = nmae(g)
        mean = np.mean(grids, axis=0)
        row[f"nmae_mean{nc}"] = nmae(mean)
        aug = read_augmentation_text(src / f"{mid}_analog_chip{a.chips[0]}.CHGCAR")
        info = write_chgcar(dst / "chgcar" / f"{mid}_analog_mean{nc}.CHGCAR", sd, lowpass_sphere(mean, sd["lattice"], gcut),
                            units="rhoV", comment=f"ChargE3Net mean of chips {a.chips} {mid} lowpass |G|<={gcut:.2f}",
                            aug_text=aug)
        row["electrons"] = info["electrons_after"]
        pd.DataFrame([row]).to_csv(dst / "mean.csv", index=False)
        print(row, flush=True)


if __name__ == "__main__":
    main()
