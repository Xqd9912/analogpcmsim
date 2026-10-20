"""P1 summary: projected PCM-array energy (NeuroSim) vs measured GPU energy (A800) per DiffCSP sample and per
ChargE3Net grid point.

usage: python scripts/run_p1_compare.py [--neurosim results/P1_neurosim/nominal_22nm] [--gpu results/P1_gpu/a800_fp32]
DiffCSP: GPU energy of one batch of the first b structures of the T3 subset (measured, board energy) vs the projected
crossbar energy of the same structures, E = sum_layers n_vec(N) * in * out * e_MAC, with e_MAC = NeuroSim energy per
crossbar MAC (typical structure mp-2095) and n_vec scaling as N^2 (edge layers), N (node layers) or 1 (lattice
head); plus an upper bound for the digital elementwise ops (pJ/op from the P1 config, ops per evaluation from T3
scaled by the same N^2 / N mix). Evaluations per sample = 2000 for both.
ChargE3Net: Amdahl framing. Only the crossbar MACs move to the array; the rest stays on the GPU. With f = on-array
share of the GPU energy (approximated by the on-array MAC share, T3 profile), E_hybrid = (1 - f) * E_GPU + E_array,
and E_GPU / E_hybrid <= 1 / (1 - f).
All analog energies are projections (NeuroSim), GPU energies are measurements. Writes <neurosim dir>/compare.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pcmsim.diffcsp.evaluate import load_subset

ROOT = Path(__file__).resolve().parents[1]


def diffcsp_macs(layers: pd.DataFrame, natoms: list[int], n_ref: int) -> np.ndarray:
    """Crossbar MACs per evaluation for each structure from the per-matrix NeuroSim table (traced at n_ref atoms)."""
    w = (layers["in"] * layers["out"]).to_numpy(float)
    nv = layers["n_vec"].to_numpy()
    cls = np.where(nv == n_ref ** 2, 2, np.where(nv == n_ref, 1, 0))  # exponent of N
    return np.array([(w * np.power(float(n), cls)).sum() for n in natoms])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurosim", default="results/P1_neurosim/nominal_22nm")
    ap.add_argument("--gpu", default="results/P1_gpu/a800_fp32")
    a = ap.parse_args()
    nsd, gpud = ROOT / a.neurosim, ROOT / a.gpu
    ns = json.loads((nsd / "metrics.json").read_text())
    gpu = json.loads((gpud / "metrics.json").read_text())
    gruns = pd.read_csv(gpud / "runs.csv")
    prof = json.loads((ROOT / "results/T3/mac_profile_cpu/mac_profile.json").read_text())["models"]
    cfg = ns["config"]
    out = {"neurosim_dir": a.neurosim, "gpu_dir": a.gpu, "gpu": gpu["gpu_before"]["name"],
           "gpu_idle_power_W": gpu["idle_power_W"], "note": "analog = NeuroSim projection; GPU = measured board energy",
           "diffcsp": {}, "charge3net": {}}

    # ---------------------------------------------------------------- DiffCSP
    dc = gpu["config"]["diffcsp"]
    n_eval = cfg["diffcsp"]["evals_per_sample"]
    ew = cfg["digital_elementwise_pJ_per_op"]["value"]
    elem_ref = prof["diffcsp"]["macs_per_eval"]["digital_elementwise_ops"]
    for var in ("nominal", "cols2", "mlsa"):
        key = f"diffcsp/{var}"
        if key not in ns["runs"]:
            continue
        r = ns["runs"][key]
        layers = pd.read_csv(nsd / f"layers_diffcsp_{var}.csv")
        n_ref = r["trace"]["natoms"]
        e_mac = r["energy_per_eval_pJ"] / r["crossbar_mac_per_eval"]  # pJ / MAC
        mac_ref = diffcsp_macs(layers, [n_ref], n_ref)[0]
        res = {"energy_per_mac_pJ": e_mac, "tops_per_w": r["chip"]["tops_per_w"], "batches": {}}
        items = load_subset(dc["subset"], dc["split_csv"], max(dc["batch_sizes"]))
        for b, g in gruns[gruns.workload == "diffcsp_sample"].groupby("batch_size"):
            b = int(b)
            nat = [len(s) for _, s in items[:b]]
            mac = diffcsp_macs(layers, nat, n_ref) * n_eval  # per structure (whole sampling)
            elem = elem_ref * mac / (mac_ref * n_eval) * n_eval  # elementwise ops scaled with the crossbar work
            e_arr = mac * e_mac * 1e-12  # J
            e_elem = elem * ew * 1e-12
            e_gpu = g.energy_J.mean() / b  # J per structure
            res["batches"][b] = {
                "mean_natoms": float(np.mean(nat)), "crossbar_mac_per_structure": float(mac.mean()),
                "gpu_J_per_structure": float(e_gpu), "gpu_J_per_structure_std": float(g.energy_J.std(ddof=0) / b),
                "gpu_s_per_structure": float(g.wall_s.mean() / b), "gpu_pJ_per_mac": float(e_gpu / mac.mean() * 1e12),
                "analog_crossbar_J_per_structure": float(e_arr.mean()),
                "analog_elementwise_upper_J_per_structure": float(e_elem.mean()),
                "analog_total_J_per_structure": float(e_arr.mean() + e_elem.mean()),
                "analog_latency_s_per_structure": float(r["latency_per_eval_ns"] * 1e-9 * n_eval
                                                        * mac.mean() / (mac_ref * n_eval)),
                "energy_ratio_gpu_over_analog": float(e_gpu / (e_arr.mean() + e_elem.mean()))}
        out["diffcsp"][var] = res

    # ------------------------------------------------------------- ChargE3Net
    f = prof["charge3net"]["on_array_fraction"]["mac_only_e3nn_measured"]
    for var in ("nominal", "cols2", "mlsa"):
        key = f"charge3net/{var}"
        if key not in ns["runs"]:
            continue
        r = ns["runs"][key]
        e_arr = r["energy_per_probe_pJ"] * 1e6 * 1e-12  # J per 1e6 probes
        res = {"array_J_per_Mprobe": e_arr, "tops_per_w": r["chip"]["tops_per_w"], "on_array_mac_share": f,
               "amdahl_max_gain": 1 / (1 - f), "batches": {}}
        steady = gruns[(gruns.workload == "charge3net_predict") & (gruns.repeat > 0)]  # repeat 0: residual compile
        for bp, g in steady.groupby("batch_probes"):
            e_gpu = float(g.energy_per_Mprobe_J.mean())
            e_hyb = (1 - f) * e_gpu + e_arr
            res["batches"][int(bp)] = {"gpu_J_per_Mprobe": e_gpu, "gpu_s_per_Mprobe": float(g.wall_per_Mprobe_s.mean()),
                                       "hybrid_J_per_Mprobe": e_hyb, "energy_ratio_gpu_over_hybrid": e_gpu / e_hyb}
        out["charge3net"][var] = res
    (nsd / "compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
