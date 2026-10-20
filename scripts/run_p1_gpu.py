"""P1 GPU reference: measured wall time and board energy of *digital* DiffCSP sampling and ChargE3Net inference.

usage: python scripts/run_p1_gpu.py --config configs/P1_gpu.yaml --device cuda:0 [--tag name] [--allow_busy]
Run on an otherwise idle GPU (we used an A800-SXM4-80GB). Energy = difference of the NVML total-energy counter (mJ) over the
timed region (torch.cuda.synchronize at both ends); board power is also sampled every 50 ms for a trace.
Idle power is measured first and reported separately; headline numbers are total board energy (no idle
subtraction). Host CPU/DRAM energy is not measured. ChargE3Net probe-graph construction (CPU) is timed
separately and excluded from the GPU energy.
Writes results/P1_gpu/<run_tag>/{config.yaml, metrics.json, runs.csv, power_trace.csv}.
"""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import ase
import numpy as np
import pandas as pd
import pynvml
import torch
import yaml

from pcmsim.density import charge3net as c3
from pcmsim.density.chgcar import load_mp_chgcar_json
from pcmsim.diffcsp import load_mp20_csp
from pcmsim.diffcsp.evaluate import batch_inputs, load_subset

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "src", "scripts", "configs"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


class GpuMeter:
    """Board energy (J, NVML counter) and power trace (W, 50 ms) of one GPU over a `with` block."""

    def __init__(self, handle, period_s: float = 0.05):
        self.h, self.period = handle, period_s
        self.trace: list[tuple[float, float]] = []

    def _poll(self):
        while not self._stop.is_set():
            self.trace.append((time.perf_counter() - self.t0, pynvml.nvmlDeviceGetPowerUsage(self.h) / 1e3))
            time.sleep(self.period)

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.trace, self._stop = [], threading.Event()
        self.e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
        self.t0 = time.perf_counter()
        self._th = threading.Thread(target=self._poll, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.wall_s = time.perf_counter() - self.t0
        self.energy_J = (pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h) - self.e0) / 1e3
        self._stop.set()
        self._th.join()
        p = np.array([w for _, w in self.trace]) if self.trace else np.array([np.nan])
        self.stats = {"wall_s": self.wall_s, "energy_J": self.energy_J, "avg_power_W": self.energy_J / self.wall_s,
                      "trace_mean_W": float(p.mean()), "trace_max_W": float(p.max()), "n_samples": len(self.trace)}


def gpu_info(h) -> dict:
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
    return {"name": pynvml.nvmlDeviceGetName(h), "driver": pynvml.nvmlSystemGetDriverVersion(),
            "power_limit_W": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1e3,
            "sm_clock_MHz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "sm_clock_max_MHz": pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM),
            "temperature_C": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "n_other_compute_procs": len(procs), "other_procs_mem_MiB": [p.usedGpuMemory / 2**20 for p in procs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--allow_busy", action="store_true", help="run even if other processes use the GPU (smoke)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = yaml.safe_load(Path(a.config).read_text())
    dev = torch.device(a.device)
    tag = a.tag or cfg.get("run_tag") or time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / cfg["exp_id"] / tag
    out.mkdir(parents=True, exist_ok=True)

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(dev.index or 0)  # NVML index == CUDA index (no CUDA_VISIBLE_DEVICES)
    info0 = gpu_info(h)
    if info0["n_other_compute_procs"] and not a.allow_busy:
        raise SystemExit(f"GPU busy: {info0['n_other_compute_procs']} other compute processes; use --allow_busy")
    meta = {"config": cfg, "config_file": a.config, "git_commit": git_commit(), "device": str(dev),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu_before": info0,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "start": time.strftime("%Y-%m-%d %H:%M:%S")}
    rows, traces = [], []

    def record(meter, **kw):
        rid = len(rows)
        rows.append({"run_id": rid, **kw, **meter.stats})
        traces.extend({"run_id": rid, "t_s": t, "power_W": w} for t, w in meter.trace)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)

    # idle baseline
    with GpuMeter(h) as m:
        time.sleep(cfg["idle_s"])
    record(m, workload="idle")
    idle_W = m.stats["avg_power_W"]

    # DiffCSP: one batch of b structures per run, full 1000-step PC sampler (2 network evals per step)
    dc = cfg["diffcsp"]
    model, _ = load_mp20_csp(str(ROOT / dc["checkpoint"]))
    model.to(dev)
    n_eval = [0]
    model.decoder.register_forward_hook(lambda *_: n_eval.__setitem__(0, n_eval[0] + 1))
    items = load_subset(dc["subset"], dc["split_csv"], max(dc["batch_sizes"]))
    for b in dc["batch_sizes"]:
        structs = [s for _, s in items[:b]]
        nat, types, n2g = batch_inputs(structs, dev)
        torch.manual_seed(0)
        model.sample(nat, types, n2g, step_lr=1e-5, timesteps=dc.get("warmup_timesteps", 20))  # warm-up
        for r in range(cfg["repeats"]):
            torch.manual_seed(dc.get("sample_seed", 0) + r)
            n_eval[0] = 0
            with GpuMeter(h) as m:
                model.sample(nat, types, n2g, step_lr=1e-5, timesteps=dc.get("timesteps"))  # None = full 1000
            record(m, workload="diffcsp_sample", batch_size=b, repeat=r, n_atoms_total=int(nat.sum()),
                   evals_per_sample=n_eval[0], energy_per_sample_J=m.stats["energy_J"] / b,
                   energy_per_sample_eval_mJ=1e3 * m.stats["energy_J"] / (b * n_eval[0]),
                   wall_per_sample_s=m.stats["wall_s"] / b)
    del model
    torch.cuda.empty_cache()

    # ChargE3Net: all probes of one structure per run (atom representation once + probe batches)
    cc = cfg["charge3net"]
    c3m = c3.build_model(dev)
    sub = pd.read_csv(ROOT / cc["subset"]).head(cc["n_structures"])
    for _, s in sub.iterrows():
        st, total, _ = load_mp_chgcar_json(ROOT / cc["chgcar_dir"] / f"{s.task_id}.json.gz")
        atoms = ase.Atoms(symbols=st["species"], scaled_positions=st["frac_coords"], cell=st["lattice"], pbc=True)
        flat = np.random.default_rng(cc.get("probe_seed", 0)).choice(total.size, size=min(cc["probes"], total.size),
                                                                    replace=False)
        ijk = np.stack(np.unravel_index(flat, total.shape), axis=1)
        cart = (ijk / np.array(total.shape)) @ np.array(atoms.get_cell())
        for bp in cc["batch_probes"]:
            t0 = time.time()
            batches = c3.prepare_batches(atoms, cart, bp)
            t_graph = time.time() - t0
            c3.predict_batches(c3m, batches[:2], dev)  # warm-up
            for r in range(cfg["repeats"]):
                with GpuMeter(h) as m:
                    c3.predict_batches(c3m, batches, dev)
                record(m, workload="charge3net_predict", material_id=s.material_id, natoms=len(atoms),
                       n_probes=len(cart), batch_probes=bp, repeat=r, wall_graph_cpu_s=t_graph,
                       energy_per_Mprobe_J=1e6 * m.stats["energy_J"] / len(cart),
                       wall_per_Mprobe_s=1e6 * m.stats["wall_s"] / len(cart))
            del batches

    meta["gpu_after"] = gpu_info(h)
    meta["idle_power_W"] = idle_W
    df = pd.DataFrame(rows)
    summ = {}
    for b, g in df[df.workload == "diffcsp_sample"].groupby("batch_size"):
        summ[f"diffcsp_b{int(b)}"] = {"energy_per_sample_J": float(g.energy_per_sample_J.mean()),
                                 "energy_per_sample_J_std": float(g.energy_per_sample_J.std(ddof=0)),
                                 "energy_per_sample_eval_mJ": float(g.energy_per_sample_eval_mJ.mean()),
                                 "wall_per_sample_s": float(g.wall_per_sample_s.mean()),
                                 "avg_power_W": float(g.avg_power_W.mean()),
                                 "evals_per_sample": int(g.evals_per_sample.iloc[0])}
    for bp, g in df[df.workload == "charge3net_predict"].groupby("batch_probes"):
        summ[f"charge3net_bp{int(bp)}"] = {"energy_per_Mprobe_J": float(g.energy_per_Mprobe_J.mean()),
                                      "energy_per_Mprobe_J_std": float(g.energy_per_Mprobe_J.std(ddof=0)),
                                      "wall_per_Mprobe_s": float(g.wall_per_Mprobe_s.mean()),
                                      "avg_power_W": float(g.avg_power_W.mean())}
    meta["summary"] = summ
    meta["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    pd.DataFrame(traces).to_csv(out / "power_trace.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps(summ, indent=1), "->", out)


if __name__ == "__main__":
    main()
