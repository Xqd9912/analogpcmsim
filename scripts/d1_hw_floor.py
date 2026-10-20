"""D1: structure-independent hardware term of the analog ChargE3Net density error.

env: pcm_models.  usage: python scripts/d1_hw_floor.py [--run probes300k_dual_am_c012]
Model: NMAE_analog^2 = NMAE_digital^2 + sigma_hw^2 (errors add in quadrature), sigma_hw = sqrt(median(analog^2 - digital^2)),
analog = prediction averaged over the 3 arrays (chip seeds 0-2). Also reports the analog-vs-digital NMAE
(nmae_vs_digital_chipmean), i.e. the error the arrays add on top of the digital model. NMAE in %.
Output: results/D1/<run>/hw_floor.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="probes300k_dual_am_c012")
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "results" / "D1").glob(f"{a.run}_shard*of2/per_structure.csv"))])
    x, y = d.nmae_digital.to_numpy() * 100, d.nmae_analog_chipmean.to_numpy() * 100
    v = d.nmae_vs_digital_chipmean.to_numpy() * 100
    s = float(np.sqrt(np.median(np.clip(y ** 2 - x ** 2, 0, None))))
    pred = np.sqrt(x ** 2 + s ** 2)
    ly, lp = np.log(y), np.log(pred)
    out = {"run": a.run, "n": int(len(d)), "sigma_hw_pct": s,
           "model_r2_log": float(1 - np.sum((ly - lp) ** 2) / np.sum((ly - ly.mean()) ** 2)),
           "model_median_rel_dev": float(np.median(np.abs(y - pred) / y)),
           "analog_vs_digital_pct": {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())},
           "corr_analog_vs_digital_with_digital": float(np.corrcoef(v, x)[0, 1]),
           "median_digital_pct": float(np.median(x)), "median_analog_pct": float(np.median(y))}
    od = ROOT / "results" / "D1" / a.run
    od.mkdir(parents=True, exist_ok=True)
    (od / "hw_floor.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
