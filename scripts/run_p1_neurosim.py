"""P1: NeuroSim (V1.4, 22 nm) projected energy / latency of the on-array part of DiffCSP and ChargE3Net.

usage: python scripts/run_p1_neurosim.py --config configs/P1_neurosim.yaml [--models diffcsp charge3net]
                                         [--variants nominal cols2 mlsa] [--tag name]
Traces: the experiment's own analog conversion (pcmsim.convert, nominal device config) run in digital mode with
hooks on every AnalogMatrix, i.e. exactly the crossbar matrices of G1/G3/D1. DiffCSP: one decoder evaluation of the
T3 typical structure (mp-2095, 20 atoms), input vectors pooled over 3 diffusion time steps. ChargE3Net: one batch
of random probes in mp-2095, probe model only (the atom part is ~1e-5 of the crossbar MACs).
All energies are NeuroSim projections (estimated, not measured). E_total = 2 * MAC / (NeuroSim TOPS/W), which
includes NeuroSim's post-layout factor zeta and leakage.
Writes results/P1_neurosim/<run_tag>/{config.yaml, metrics.json, layers_<model>_<variant>.csv, logs/}.
Build directories and trace files go to build/neurosim/<run_tag>/ (not tracked).
"""
import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import ase.io
import numpy as np
import pandas as pd
import torch
import yaml

from pcmsim import neurosim as ns
from pcmsim.analog import set_digital
from pcmsim.convert import to_analog_charge3net, to_analog_diffcsp
from pcmsim.device import PCMConfig

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "src", "scripts", "configs"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def trace_diffcsp(cfg, pcm) -> tuple[list[dict], dict]:
    from pcmsim.diffcsp import load_mp20_csp
    from pcmsim.diffcsp.evaluate import batch_inputs, gt_structure
    dc = cfg["diffcsp"]
    model, _ = load_mp20_csp(str(ROOT / dc["checkpoint"]))
    to_analog_diffcsp(model, pcm)
    set_digital(model, True)
    s = gt_structure(pd.read_csv(ROOT / dc["split_csv"]).set_index("material_id").loc[dc["structure_id"], "cif"])
    nat, types, n2g = batch_inputs([s], "cpu")
    rec = ns.TraceRecorder(model)
    n_eval = [0]
    h = model.decoder.register_forward_hook(lambda *_: n_eval.__setitem__(0, n_eval[0] + 1))
    torch.manual_seed(dc["sample_seed"])
    model.sample(nat, types, n2g, step_lr=1e-5, timesteps=dc["trace_timesteps"])
    h.remove()
    rec.remove()
    return rec.layers(), {"structure": dc["structure_id"], "natoms": len(s), "trace_evals_pooled": n_eval[0]}


def trace_charge3net(cfg, pcm) -> tuple[list[dict], dict]:
    from pcmsim.density import charge3net as c3
    cc = cfg["charge3net"]
    model = c3.build_model("cpu")
    to_analog_charge3net(model, pcm)
    set_digital(model, True)
    atoms = ase.io.read(ROOT / cc["structure"], format="vasp")
    cart = np.random.default_rng(cc["probe_seed"]).random((cc["n_probes"], 3)) @ np.array(atoms.get_cell())
    rec = ns.TraceRecorder(model, include=lambda n: n.startswith("probe_model"))
    c3.predict_points(model, atoms, cart, cc["n_probes"], "cpu")
    rec.remove()
    return rec.layers(), {"structure": cc["structure"], "natoms": len(atoms), "n_probes": cc["n_probes"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", default=["diffcsp", "charge3net"])
    ap.add_argument("--variants", nargs="*", default=["nominal", "cols2", "mlsa"])
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    tag = a.tag or cfg["run_tag"]
    out = ROOT / "results" / cfg["exp_id"] / tag
    (out / "logs").mkdir(parents=True, exist_ok=True)
    bdir = ROOT / "build" / "neurosim" / tag
    pcm = PCMConfig.from_yaml(ROOT / cfg["device_config"])
    meta = {"config": cfg, "config_file": a.config, "git_commit": git_commit(),
            "neurosim": "third_party/NeuroSim branch 2DInferenceV1.4 commit 8a88abf85844c0e1ba17cc771ea535fff6040456",
            "start": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": {}}
    if (out / "metrics.json").exists():  # keep runs of other models / variants
        meta["runs"] = json.loads((out / "metrics.json").read_text()).get("runs", {})
    zeta_note = "E_total_pJ = 2*MAC / TOPS/W (NeuroSim; includes post-layout factor zeta = 1.22 and leakage)"

    for model_name in a.models:
        t0 = time.time()
        layers, tinfo = (trace_diffcsp if model_name == "diffcsp" else trace_charge3net)(cfg, pcm)
        t_trace = time.time() - t0
        mac = float(sum(L["n_vec"] * L["weight"].shape[0] * L["weight"].shape[1] for L in layers))
        for var in a.variants:
            vcfg = cfg.get("variants", {}).get(var, {}) if var != "nominal" else {}
            args = {k: v["value"] for k, v in cfg["args"].items()}
            args.update(vcfg.get("args", {}))
            tdir = bdir / f"traces_{model_name}_sb{args['synapseBit']}"
            netcsv, files, toggle = ns.write_traces(layers, tdir, args["synapseBit"], args["numBitInput"])
            param = {k: v["value"] for k, v in cfg["param"].items()}
            param.update(vcfg.get("param", {}))
            param["inputtoggle"] = round(toggle, 4)
            t1 = time.time()
            binary = ns.build(param, bdir / f"bin_{model_name}_{var}")
            log = out / "logs" / f"neurosim_{model_name}_{var}.log"
            res = ns.parse(ns.run(binary, netcsv, files, args["synapseBit"], args["numBitInput"],
                                  args["numRowSubArray"], args["numRowParallel"], log))
            t_ns = time.time() - t1
            c = res["chip"]
            e_tot = 2 * mac / c["tops_per_w"]  # pJ per network evaluation
            run = {"model": model_name, "variant": var, "args": args, "param": param, "trace": tinfo,
                   "n_matrices": len(layers), "crossbar_mac_per_eval": mac, "energy_per_eval_pJ": e_tot,
                   "energy_per_mac_fJ": 1e3 * e_tot / mac, "latency_per_eval_ns": c["latency_ns"],
                   "chip": c, "energy_note": zeta_note, "wall_trace_s": t_trace, "wall_neurosim_s": t_ns}
            if model_name == "diffcsp":
                n = cfg["diffcsp"]["evals_per_sample"]
                ew = cfg["digital_elementwise_pJ_per_op"]["value"]
                elem = json.loads((ROOT / "results/T3/mac_profile_cpu/mac_profile.json").read_text())
                run["digital_elementwise_ops_per_eval"] = elem_ops = _diffcsp_elementwise(elem)
                run["energy_per_structure_uJ"] = {"crossbar": 1e-6 * e_tot * n,
                                                  "digital_elementwise_upper": 1e-6 * elem_ops * ew * n}
            else:
                run["energy_per_probe_pJ"] = e_tot / tinfo["n_probes"]
            meta["runs"][f"{model_name}/{var}"] = run
            df = pd.DataFrame(res["layers"])
            df.insert(0, "matrix", [L["name"] for L in layers][:len(df)])
            df.insert(1, "in", [L["weight"].shape[1] for L in layers][:len(df)])
            df.insert(2, "out", [L["weight"].shape[0] for L in layers][:len(df)])
            df.insert(3, "n_vec", [L["n_vec"] for L in layers][:len(df)])
            df.to_csv(out / f"layers_{model_name}_{var}.csv", index=False)
            print(json.dumps({k: run[k] for k in ("model", "variant", "crossbar_mac_per_eval", "energy_per_eval_pJ",
                                                  "energy_per_mac_fJ", "latency_per_eval_ns")}
                             | {"TOPS/W": c["tops_per_w"], "area_mm2": c["chip_area_um2"] / 1e6,
                                "inputtoggle": param["inputtoggle"], "wall_neurosim_s": round(t_ns, 1)}), flush=True)
            meta["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
            (out / "metrics.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("->", out)


def _diffcsp_elementwise(prof: dict) -> float:
    """Digital elementwise ops per DiffCSP decoder evaluation (T3 profile, 20-atom typical structure)."""
    return float(prof["models"]["diffcsp"]["macs_per_eval"]["digital_elementwise_ops"])


if __name__ == "__main__":
    main()
