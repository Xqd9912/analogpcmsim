"""DiffCSP digital / analog evaluation (T3 baseline, G1 ...).

usage: python scripts/run_diffcsp.py --config configs/G1_dual_full.yaml [--device cuda:0] [--tag name]
Writes results/<exp_id>/<run_tag>/{config.yaml, metrics.json, per_structure.csv}.
Analog mode: one run per chip_seed (programming noise), read noise from a dedicated read_seed stream,
DAC/ADC ranges calibrated on a strided sampling pass over a calibration batch from the val split.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from pcmsim.analog import analog_matrices, calibrate, instantiate_chip, set_read_seed
from pcmsim.convert import to_analog_diffcsp
from pcmsim.device import PCMConfig
from pcmsim.diffcsp import load_mp20_csp
from pcmsim.diffcsp.evaluate import batch_inputs, load_subset, match, sample_subset, summarize
from pcmsim.diffcsp.noise_budget import calibrate_read_noise, sample_noise_budget

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "src", "scripts", "configs"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chips", nargs="*", type=int, default=None,
                    help="override chip_seeds (one process per chip); results go to <run_tag>_chip<k>")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = yaml.safe_load(Path(a.config).read_text())
    dev = torch.device(a.device)
    tag = a.tag or cfg.get("run_tag") or time.strftime("%Y%m%d-%H%M%S")
    if a.chips is not None and cfg["mode"] == "analog":
        cfg["chip_seeds"] = a.chips
        tag = f"{tag}_chip{'-'.join(map(str, a.chips))}"
    out = ROOT / "results" / cfg["exp_id"] / tag
    out.mkdir(parents=True, exist_ok=True)

    items = load_subset(cfg["subset"], cfg["split_csv"], cfg.get("n_structures"))
    model, _ = load_mp20_csp(str(ROOT / cfg["checkpoint"]))
    model.to(dev)
    meta = {"config": cfg, "config_file": a.config, "git_commit": git_commit(), "device": str(dev),
            "torch": torch.__version__, "n_structures": len(items), "runs": []}

    if cfg["mode"] == "digital":
        runs = [None]
    else:
        pcm = PCMConfig.from_yaml(ROOT / cfg["device_config"], **cfg.get("device_overrides", {}))
        conv = to_analog_diffcsp(model, pcm)
        meta["analog_layers"] = [n for n, _ in conv]
        meta["pcm_config"] = pcm.__dict__
        cal = load_subset(cfg["calib"]["subset"], cfg["calib"]["split_csv"], cfg["calib"]["n_structures"])
        nat_c, types_c, n2g_c = batch_inputs([s for _, s in cal], dev)

        def calib_pass():
            torch.manual_seed(12345)
            model.sample(nat_c, types_c, n2g_c, step_lr=1e-5, timesteps=cfg["calib"]["timesteps"])

        runs = cfg["chip_seeds"]

    rows_all = []
    for chip in runs:
        t0 = time.time()
        if chip is not None:
            instantiate_chip(model, chip)
            model.to(dev)
            opts = cfg.get("analog_options", {})
            for m in analog_matrices(model):  # ablation switches (default: everything on)
                m.read_noise = opts.get("read_noise", True)
                m.use_dac_adc = opts.get("dac_adc", True)
                m.use_dac = opts.get("dac", m.use_dac)
                m.use_adc = opts.get("adc", m.use_adc)
            calibrate(model, calib_pass)
            set_read_seed(model, cfg["read_seed"] + chip, dev)
        sampler, extra = None, {}
        if chip is not None and cfg.get("sampler", "naive") == "budget":
            nb = cfg.get("noise_budget", {})
            cal_nb = calibrate_read_noise(model, nat_c, types_c, n2g_c, n_t=nb.get("n_t", 50),
                                          n_rep=nb.get("n_rep", 16))
            extra["sigma_a"] = {k: v.tolist() for k, v in cal_nb.items()}
            tail = nb.get("digital_tail", 0)
            sampler = lambda m, na, ty, ng: sample_noise_budget(m, na, ty, ng, cal_nb, digital_tail=tail)
        t_cal = time.time() - t0
        preds, t_s, fracs = sample_subset(model, items, cfg["batch_size"], cfg["sample_seed"], dev, sampler)
        if fracs and fracs[0] is not None:
            extra["analog_fraction"] = float(sum(fracs) / len(fracs))
        rows = match(preds, [s for _, s in items])
        for (mid, g), r in zip(items, rows):
            rows_all.append({"chip_seed": chip, "material_id": mid, "natoms": len(g), **r})
        s = summarize(rows)
        s.update(chip_seed=chip, wall_sample_s=t_s, wall_calib_program_s=t_cal,
                 s_per_structure=t_s / len(items), **extra)
        meta["runs"].append(s)
        print(json.dumps(s), flush=True)
        (out / "metrics.json").write_text(json.dumps(meta, indent=1, default=str))
        pd.DataFrame(rows_all).to_csv(out / "per_structure.csv", index=False)
    mr = [r["match_rate"] for r in meta["runs"]]
    meta["match_rate_mean"] = float(pd.Series(mr).mean())
    meta["match_rate_std"] = float(pd.Series(mr).std(ddof=0)) if len(mr) > 1 else 0.0
    (out / "metrics.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("->", out)


if __name__ == "__main__":
    main()
