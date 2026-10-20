"""T1: device model characterisation + quantitative unit-test data for Figure 1.

Usage: python scripts/run_t1.py   (env: pcm_models)
Writes results/T1/nominal/ (json/csv + preview png).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pcmsim.device import PCMConfig, load_trace, select_states, trace_stats, TEN_YEARS_S, N_LTP
from pcmsim import mapping
from pcmsim.analog import (
    AnalogLinear, analog_matrices, calibrate, instantiate_chip, set_drift_time,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "T1" / "nominal"
CHG3_CKPT = ROOT / "third_party" / "charge3net" / "models" / "charge3net_mp.pt"
CHIP_SEEDS = (0, 1, 2)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def real_weights() -> dict[str, torch.Tensor]:
    """Radial-MLP output layers of ChargE3Net-MP as realistic (out, in) weight matrices."""
    if not CHG3_CKPT.exists():
        return {}
    sd = torch.load(CHG3_CKPT, map_location="cpu", weights_only=False)["model"]
    keys = [k for k in sd if k.endswith("fc.layer1.weight") and k.startswith("probe_model")]
    # e3nn FullyConnectedNet stores (in, out) and scales by 1/sqrt(in); scale is irrelevant for relative errors
    return {k: sd[k].t().contiguous().float() for k in keys}


def weight_sets() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    ws = {
        "gaussian_256x256": torch.randn(256, 256, generator=g) * 0.05,
        "laplace_256x256": torch.distributions.Laplace(0.0, 0.05).sample((256, 256)),
    }
    ws.update({f"charge3net:{k}": v for k, v in real_weights().items()})
    return ws


def make_layer(w: torch.Tensor, cfg: PCMConfig) -> AnalogLinear:
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
    with torch.no_grad():
        lin.weight.copy_(w)
    return AnalogLinear(lin, cfg)


def rel_err(y, ref):
    return float((y - ref).norm() / ref.norm())


# ----------------------------------------------------------------------------------
def fig1_device(cfgs):
    g = load_trace()
    stats = trace_stats(g)
    (OUT / "device_stats.json").write_text(json.dumps(stats, indent=2))
    write_csv(OUT / "fig1a_ltp_ltd.csv", ["pulse", "phase", "G_uS"],
              [[i + 1, "LTP" if i < N_LTP else "LTD", round(float(v), 4)] for i, v in enumerate(g)])
    rows = []
    for name, cfg in cfgs.items():
        S = np.array(cfg.states)
        lv = np.concatenate([-(S[::-1] - S[0]), (S[1:] - S[0])]) / (S[-1] - S[0])
        for s in S:
            rows.append([name, "state_uS", round(float(s), 4)])
        for l in lv:
            rows.append([name, "weight_level_norm", round(float(l), 6)])
    write_csv(OUT / "fig1b_states_levels.csv", ["state_set", "kind", "value"], rows)
    nonideal = [
        ["conductance states S", "15 (2sigma) / 12 (3sigma) measured levels", "measured (R.csv)", "fixed"],
        ["programming noise", "sigma_prog = 2.5 uS per device", "estimated from pulse-to-pulse fluctuation (LTP 1.87, LTD 2.80 uS)", "once per chip instance (static)"],
        ["read noise", "1% of G (relative), per device per read", "literature nominal - SOURCE TO BE CONFIRMED", "every forward"],
        ["drift", "G(t)=G(t0)(t/t0)^-nu, nu=0.03, t0=20 s; eval at t0 and 10 y", "assumed - TO BE CONFIRMED for this device batch", "deterministic"],
        ["drift compensation", "global per-layer output scale from calibration batch", "standard (GDC)", "once per time point"],
        ["DAC / ADC", "8 bit / 8 bit, ADC range from calibration batch (max)", "assumed", "fixed"],
        ["tiling", "256x256, partial sums accumulated in fp32", "assumed", "fixed"],
    ]
    write_csv(OUT / "fig1c_nonidealities.csv", ["item", "value", "source", "sampling"], nonideal)
    return stats


def test1_alpha_curves(cfgs, ws):
    """T1-1: quantization MSE vs alpha (no noise), per weight set and state set."""
    rows, summary = [], {}
    for sname, cfg in cfgs.items():
        for wname, w in ws.items():
            a = make_layer(w, cfg)
            alpha, ratios, mse = mapping.search_alpha(w, a.mat.states, return_curve=True)
            var = (w ** 2).mean(dim=1)
            nmse = (mse / var).mean(dim=1)  # mean over columns of per-column normalized MSE
            for r, e in zip(ratios.tolist(), nmse.tolist()):
                rows.append([sname, wname, round(r, 4), e])
            wq = mapping.quantized_weight(w, alpha, a.mat.states)
            summary[f"{sname}|{wname}"] = {
                "shape": list(w.shape),
                "best_global_ratio": float(ratios[int(nmse.argmin())]),
                "nmse_at_best_global_ratio": float(nmse.min()),
                "nmse_per_column_alpha": float((((wq - w) ** 2).mean(1) / var).mean()),
                "nmse_at_ratio_1": float(nmse[-1]),
                "per_column_ratio_median": float((alpha / w.abs().amax(1)).median()),
                "frac_weights_zero_after_quant": float((wq == 0).float().mean()),
            }
    write_csv(OUT / "t1_1_quant_nmse_vs_alpha.csv", ["state_set", "weights", "alpha_over_maxabs", "nmse"], rows)
    return summary


def test2_prog_noise(cfg):
    """T1-2: measured conductance / weight error std vs set sigma_prog."""
    w = torch.randn(256, 512, generator=torch.Generator().manual_seed(1)) * 0.05
    rows = []
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        c = PCMConfig(**{**cfg.__dict__, "sigma_prog": cfg.sigma_prog * mult, "read_rel": 0.0})
        a = make_layer(w, c)
        m = nn.Sequential(a)
        for seed in CHIP_SEEDS:
            instantiate_chip(m, seed)
            gp, gn = mapping.target_conductance(w, a.mat.alpha, a.mat.states)
            dev_std = float(torch.cat([(a.mat.g_pos - gp).flatten(), (a.mat.g_neg - gn).flatten()]).std())
            w_q = mapping.quantized_weight(w, a.mat.alpha, a.mat.states)
            werr = (a.mat.effective_weight() - w_q) * a.mat.dg / a.mat.alpha.unsqueeze(1)
            rows.append([c.sigma_prog, seed, dev_std, float(werr.std()), math.sqrt(2) * c.sigma_prog])
    write_csv(OUT / "t1_2_prog_noise.csv",
              ["sigma_set_uS", "chip_seed", "device_err_std_uS", "pair_err_std_uS", "pair_expected_uS"], rows)
    return rows


def test3_read_noise(cfg):
    """T1-3: read noise is resampled per forward; empirical vs analytic output std."""
    w = torch.randn(128, 256, generator=torch.Generator().manual_seed(2)) * 0.05
    x = torch.randn(16, 256, generator=torch.Generator().manual_seed(3))
    rows = []
    for rr in (0.005, 0.01, 0.02):
        c = PCMConfig(**{**cfg.__dict__, "read_rel": rr})
        a = make_layer(w, c)
        m = nn.Sequential(a)
        instantiate_chip(m, 0)
        a.mat.use_dac_adc = False
        g_snapshot = a.mat.g_pos.clone()
        torch.manual_seed(10)
        ys = torch.stack([m(x) for _ in range(500)])
        pred = rr * torch.sqrt((x * x) @ (a.mat.g_pos ** 2 + a.mat.g_neg ** 2).T) * a.mat.alpha / a.mat.dg
        ratio = ys.std(0) / pred
        distinct = int(torch.unique(ys[:, 0, 0]).numel())
        rows.append([rr, float(ratio.mean()), float(ratio.std()), distinct, 500,
                     bool(torch.equal(g_snapshot, a.mat.g_pos))])
    write_csv(OUT / "t1_3_read_noise.csv",
              ["read_rel", "emp_over_pred_std_mean", "emp_over_pred_std_sd", "distinct_outputs", "n_forwards",
               "programmed_state_unchanged"], rows)
    return rows


def test4_drift(cfg):
    """T1-4: output scale vs time with/without global drift compensation (3-layer MLP)."""
    times = np.logspace(math.log10(cfg.t0), math.log10(TEN_YEARS_S), 9)
    rows = []
    for nu_std in (0.0, 0.3 * cfg.nu):
        torch.manual_seed(5)
        net = nn.Sequential(nn.Linear(128, 256, bias=False), nn.SiLU(), nn.Linear(256, 128, bias=False),
                            nn.SiLU(), nn.Linear(128, 32, bias=False))
        x_cal = torch.randn(512, 128)
        x = torch.randn(256, 128)
        ref = net(x).detach()
        c = PCMConfig(**{**cfg.__dict__, "nu_std": nu_std})
        model = nn.Sequential(*[AnalogLinear(l, c) if isinstance(l, nn.Linear) else l for l in net])
        for seed in CHIP_SEEDS:
            instantiate_chip(model, seed)
            calibrate(model, lambda: model(x_cal))
            for mat in analog_matrices(model):
                mat.read_noise = False
            y0 = model(x).detach()
            for t in times:
                for comp in (False, True):
                    set_drift_time(model, float(t), run_fn=lambda: model(x_cal), compensate=comp)
                    y = model(x).detach()
                    rows.append([nu_std, seed, float(t), comp, float(y.abs().mean() / y0.abs().mean()),
                                 rel_err(y, ref), rel_err(y0, ref)])
    write_csv(OUT / "t1_4_drift.csv",
              ["nu_std", "chip_seed", "t_s", "gdc", "output_scale_vs_t0", "rel_err_vs_digital",
               "rel_err_t0_vs_digital"], rows)
    return rows


def error_budget(cfg, ws):
    """Relative output error of single real layers when non-idealities are switched on cumulatively."""
    rows = []
    for wname, w in ws.items():
        x_cal = torch.randn(256, w.shape[1], generator=torch.Generator().manual_seed(7))
        x = torch.randn(256, w.shape[1], generator=torch.Generator().manual_seed(8))
        ref = x @ w.T
        steps = [
            ("quantization", dict(sigma_prog=0.0, read_rel=0.0), False, cfg.t0),
            ("+prog noise", dict(read_rel=0.0), False, cfg.t0),
            ("+read noise", dict(), False, cfg.t0),
            ("+DAC/ADC 8b", dict(), True, cfg.t0),
            ("+drift 10y +GDC", dict(), True, TEN_YEARS_S),
        ]
        for label, over, dacadc, t in steps:
            c = PCMConfig(**{**cfg.__dict__, **over})
            a = make_layer(w, c)
            m = nn.Sequential(a)
            for seed in CHIP_SEEDS:
                instantiate_chip(m, seed)
                calibrate(m, lambda: m(x_cal))
                a.mat.use_dac_adc = dacadc
                set_drift_time(m, t, run_fn=lambda: m(x_cal))
                torch.manual_seed(100 + seed)
                rows.append([wname, label, seed, rel_err(m(x).detach(), ref)])
    write_csv(OUT / "t1_error_budget.csv", ["weights", "config", "chip_seed", "rel_output_err"], rows)
    return rows


def preview(stats, cfgs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    g = load_trace()
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
    a = ax[0, 0]
    a.plot(range(1, 41), g[:40], "o-", ms=3, label="LTP")
    a.plot(range(41, 81), g[40:], "s-", ms=3, label="LTD")
    a.set(xlabel="pulse #", ylabel="G (uS)", title="(a) measured LTP/LTD")
    a.legend()
    a = ax[0, 1]
    for i, (n, c) in enumerate(cfgs.items()):
        S = np.array(c.states)
        a.eventplot([S], lineoffsets=i, linelengths=0.7, colors=f"C{i}")
    a.eventplot([np.sort(g)], lineoffsets=-1, linelengths=0.7, colors="grey")
    a.set_yticks([-1, 0, 1], ["all 80", f"S 2σ ({len(cfgs['S_2sigma'].states)})", f"S 3σ ({len(cfgs['S_3sigma'].states)})"])
    a.set(xlabel="G (uS)", title="(b) distinguishable states")
    a = ax[0, 2]
    df = pd.read_csv(OUT / "t1_1_quant_nmse_vs_alpha.csv")
    for (s, wn), d in df.groupby(["state_set", "weights"]):
        if s == "S_2sigma":
            a.semilogy(d.alpha_over_maxabs, d.nmse, label=wn.split(":")[-1][:28])
    a.set(xlabel="alpha / max|w|", ylabel="quantization NMSE", title="(T1-1) quant error vs alpha (2σ set)")
    a.legend(fontsize=7)
    a = ax[1, 0]
    df = pd.read_csv(OUT / "t1_2_prog_noise.csv")
    a.plot(df.sigma_set_uS, df.device_err_std_uS, "o", label="device")
    a.plot(df.sigma_set_uS, df.pair_err_std_uS, "s", label="differential pair")
    xs = np.array([0, df.sigma_set_uS.max()])
    a.plot(xs, xs, "k--", lw=0.8)
    a.plot(xs, np.sqrt(2) * xs, "k:", lw=0.8)
    a.set(xlabel="set σ_prog (uS)", ylabel="measured std (uS)", title="(T1-2) programming noise")
    a.legend()
    a = ax[1, 1]
    df = pd.read_csv(OUT / "t1_4_drift.csv")
    for (nus, comp), d in df.groupby(["nu_std", "gdc"]):
        d = d.groupby("t_s").output_scale_vs_t0.mean()
        a.semilogx(d.index, d.values, "o-" if comp else "x--", ms=3,
                   label=f"ν_std={nus:.3f}, {'GDC' if comp else 'no comp.'}")
    a.set(xlabel="t (s)", ylabel="output scale / t0", title="(T1-4) drift & compensation")
    a.legend(fontsize=7)
    a = ax[1, 2]
    df = pd.read_csv(OUT / "t1_error_budget.csv")
    order = list(dict.fromkeys(df.config))
    for wn, d in df.groupby("weights"):
        m = d.groupby("config").rel_output_err.mean().reindex(order)
        a.plot(range(len(order)), m.values, "o-", label=wn.split(":")[-1][:28])
    a.set_xticks(range(len(order)), order, rotation=30, ha="right", fontsize=8)
    a.set(ylabel="relative output error", title="single-layer error budget (2σ set)")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "t1_preview.png", dpi=130)


def main():
    OUT.mkdir(exist_ok=True)
    nom = PCMConfig.from_yaml()
    s3 = PCMConfig(**{**nom.__dict__, "states": select_states(load_trace(), 6 * nom.sigma_prog).tolist()})
    cfgs = {"S_2sigma": nom, "S_3sigma": s3}
    cfgs["S_2sigma"].to_json(OUT / "pcm_config_nominal.json")
    stats = fig1_device(cfgs)
    ws = weight_sets()
    res = {"device": stats, "t1_1": test1_alpha_curves(cfgs, ws)}
    nom = cfgs["S_2sigma"]
    res["t1_2"] = test2_prog_noise(nom)
    res["t1_3"] = test3_read_noise(nom)
    t4 = test4_drift(nom)
    error_budget(nom, ws)
    (OUT / "t1_summary.json").write_text(json.dumps(res, indent=2))
    preview(stats, cfgs)
    print(json.dumps(res["t1_1"], indent=1))
    print("t1_2", res["t1_2"][::3])
    print("t1_3", res["t1_3"])


if __name__ == "__main__":
    main()
