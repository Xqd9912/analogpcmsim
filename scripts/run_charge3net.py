"""ChargE3Net density NMAE, digital vs analog (T3.2 digital baseline, D1 analog).

usage: python scripts/run_charge3net.py --config configs/D1_dual_probes300k.yaml [--device cuda:0] [--tag name]
Per structure: MP CHGCAR (S3 json.gz) -> structure + total rho*V on the MP fine grid; predictions on the
full grid (probes: full) or on a random probe subset (probes: N, for CPU smoke tests).
NMAE = sum|pred - rho| / sum|rho| (uniform grid). Analog: one programmed chip per chip_seed, DAC/ADC full
scales calibrated once per chip on validation-split structures (disjoint from the evaluation set). With >= 2 chips the
prediction averaged over the chips is also scored (nmae_analog_chipmean; nominal for D1/D3 since 2026-10-19).
Writes results/<exp_id>/<run_tag>/{config.yaml, metrics.json, per_structure.csv}.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import ase
import ase.io
import numpy as np
import pandas as pd
import torch
import yaml

from pcmsim.analog import calibrate, instantiate_chip, set_read_seed
from pcmsim.convert import to_analog_charge3net
from pcmsim.density import charge3net as c3
from pcmsim.density.chgcar import load_mp_chgcar_json
from pcmsim.device import PCMConfig

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    import subprocess
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "src", "scripts", "configs"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def load_target(path):
    st, total, meta = load_mp_chgcar_json(path)
    atoms = ase.Atoms(symbols=st["species"], scaled_positions=st["frac_coords"], cell=st["lattice"], pbc=True)
    return atoms, total / atoms.get_volume(), meta  # density e/A^3 on the grid


def probe_set(atoms, rho, probes, seed):
    grid = rho.shape
    if probes == "full":
        return c3.grid_points(atoms, grid), rho.reshape(-1)
    flat = np.random.default_rng(seed).choice(rho.size, size=min(int(probes), rho.size), replace=False)
    ijk = np.stack(np.unravel_index(flat, grid), axis=1)
    return (ijk / np.array(grid)) @ np.array(atoms.get_cell()), rho.reshape(-1)[flat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="skip structures already in per_structure.csv")
    ap.add_argument("--batch_probes", type=int, default=None, help="override config batch_probes")
    ap.add_argument("--chips", nargs="*", type=int, default=None, help="override config chip_seeds")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = yaml.safe_load(Path(a.config).read_text())
    dev = torch.device(a.device)
    tag = a.tag or cfg.get("run_tag") or time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / cfg["exp_id"] / tag
    out.mkdir(parents=True, exist_ok=True)
    sub = pd.read_csv(ROOT / cfg["subset"])
    if cfg.get("n_structures"):
        sub = sub.head(cfg["n_structures"])
    if a.nshards > 1:  # structures k, k+n, ... -> results/<exp>/<tag>_shard<k>of<n>
        sub = sub.iloc[a.shard::a.nshards]
        tag = f"{tag}_shard{a.shard}of{a.nshards}"
        out = ROOT / "results" / cfg["exp_id"] / tag
        out.mkdir(parents=True, exist_ok=True)
    bp = a.batch_probes or cfg.get("batch_probes", 2500)

    # calibration inputs (structures only)
    cal_atoms = [ase.io.read(f) for f in sorted(glob.glob(str(ROOT / cfg["calib"]["structures_glob"])))]
    rng = np.random.default_rng(0)
    cal_pts = [rng.random((cfg["calib"]["n_probes"], 3)) @ np.array(at.get_cell()) for at in cal_atoms]

    digital = c3.build_model(dev)
    if a.chips is not None:
        cfg["chip_seeds"] = a.chips
    chips = cfg.get("chip_seeds", []) if cfg["mode"] == "analog" else []
    meta = {"config": cfg, "config_file": a.config, "git_commit": git_commit(), "device": str(dev),
            "torch": torch.__version__, "calib_structures": [at.get_chemical_formula() for at in cal_atoms]}
    analog = {}
    if chips:
        pcm = PCMConfig.from_yaml(ROOT / cfg["device_config"], **cfg.get("device_overrides", {}))
        meta["pcm_config"] = pcm.__dict__
        meta["calib_wall_s"] = {}
        for chip in chips:  # one programmed + calibrated model per chip instance
            t0 = time.time()
            m = c3.build_model(dev)
            conv = to_analog_charge3net(m, pcm)
            instantiate_chip(m, chip)
            m.to(dev)
            calibrate(m, lambda: [c3.predict_points(m, at, p, bp, dev) for at, p in zip(cal_atoms, cal_pts)])
            set_read_seed(m, cfg["read_seed"] + chip, dev)
            analog[chip] = m
            meta["calib_wall_s"][chip] = time.time() - t0
        meta["n_analog_layers"] = len(conv)

    rows = []
    if a.resume and (out / "per_structure.csv").exists():
        rows = pd.read_csv(out / "per_structure.csv").to_dict("records")
        done = {r["material_id"] for r in rows}
        sub = sub[~sub.material_id.isin(done)]
        print(f"resume: {len(done)} done, {len(sub)} to go", flush=True)
    for _, r in sub.iterrows():
        atoms, rho, m_ = load_target(ROOT / cfg["chgcar_dir"] / f"{r.task_id}.json.gz")
        pts, tgt = probe_set(atoms, rho, cfg.get("probes", "full"), cfg.get("probe_seed", 0))
        t0 = time.time()
        batches = c3.prepare_batches(atoms, pts, bp)  # probe graph built once per structure
        row = {"material_id": r.material_id, "task_id": r.task_id, "formula": atoms.get_chemical_formula(),
               "natoms": len(atoms), "n_probes": len(tgt), "grid": "x".join(map(str, rho.shape)),
               "spin_polarized": m_["spin_polarized"], "wall_graph_s": time.time() - t0}
        t0 = time.time()
        pd_ = c3.predict_batches(digital, batches, dev)
        row.update(nmae_digital=c3.nmae(pd_, tgt), wall_digital_s=time.time() - t0)
        preds = []
        for chip, m in analog.items():
            t0 = time.time()
            pa = c3.predict_batches(m, batches, dev)
            row[f"nmae_analog_chip{chip}"] = c3.nmae(pa, tgt)
            row[f"nmae_vs_digital_chip{chip}"] = c3.nmae(pa, pd_)
            row[f"wall_analog_chip{chip}_s"] = time.time() - t0
            preds.append(pa)
        if len(preds) > 1:  # D1/D3 nominal (2026-10-19): average of independently programmed arrays
            pm = np.mean(preds, axis=0)
            row["nmae_analog_chipmean"] = c3.nmae(pm, tgt)
            row["nmae_vs_digital_chipmean"] = c3.nmae(pm, pd_)
        rows.append(row)
        del batches
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()
                          if k.startswith(("material_id", "natoms", "n_probes", "nmae"))}), flush=True)
        pd.DataFrame(rows).to_csv(out / "per_structure.csv", index=False)  # incremental

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_structure.csv", index=False)
    summ = {"n_structures": len(df), "nmae_digital_mean": float(df.nmae_digital.mean())}
    if chips:
        per_chip = [float(df[f"nmae_analog_chip{c}"].mean()) for c in chips]
        summ.update(nmae_analog_mean=float(np.mean(per_chip)), nmae_analog_std_over_chips=float(np.std(per_chip)),
                    nmae_analog_per_chip=per_chip,
                    nmae_vs_digital_mean=float(np.mean([df[f"nmae_vs_digital_chip{c}"].mean() for c in chips])))
        if "nmae_analog_chipmean" in df:
            summ.update(nmae_chipmean=float(df.nmae_analog_chipmean.mean()),
                        nmae_chipmean_vs_digital=float(df.nmae_vs_digital_chipmean.mean()), chipmean_chips=chips)
    meta["summary"] = summ
    (out / "metrics.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps(summ), "->", out)


if __name__ == "__main__":
    main()
