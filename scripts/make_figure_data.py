"""Derived data for figures (figure notebooks only read results/).

1. results/T1/nominal/slice_weight_error.csv  - per crossbar matrix relative weight error |W_eff - W| / |W|
   for DiffCSP and ChargE3Net, single vs dual slice, quantization only vs + programming noise, 3 chips.
2. results/G3/<run>/budget_curves.csv - noise budget per timestep: injected std vs read-noise
   share s(t) for corrector / predictor coordinates and predictor lattice (sigma_a from the G3 calibration);
   runs: smoke20_budget ("budget") and the 1000-structure dual-slice chips ("budget_dual").
usage: python scripts/make_figure_data.py [budget] [budget_dual] [slices] [showcase] [d3_final]
3. results/D3/final/ - D3 nominal tables (chip-averaged density) and the Ge2Sb2Te5 slice for Fig. 3.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pcmsim.analog import AnalogMatrix, instantiate_chip
from pcmsim.convert import to_analog_charge3net, to_analog_diffcsp
from pcmsim.density import charge3net as c3
from pcmsim.device import PCMConfig
from pcmsim.diffcsp import load_mp20_csp

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "data/weights/diffcsp_mp_csp/last.ckpt"


def slice_errors():
    rows = []
    for model_name in ("DiffCSP", "ChargE3Net"):
        for ns in (1, 2):
            for setting, over in (("quantization", dict(sigma_prog=0.0, read_rel=0.0)), ("+prog. noise", {})):
                if model_name == "DiffCSP":
                    m, _ = load_mp20_csp(str(CKPT))
                    to_analog_diffcsp(m, PCMConfig.from_yaml(n_slices=ns, **over))
                else:
                    m = c3.build_model()
                    to_analog_charge3net(m, PCMConfig.from_yaml(n_slices=ns, **over))
                names = {id(mod): n for n, mod in m.named_modules() if isinstance(mod, AnalogMatrix)}
                for chip in (0, 1, 2):
                    instantiate_chip(m, chip)
                    for mod in m.modules():
                        if isinstance(mod, AnalogMatrix):
                            w = mod.weight.detach()
                            err = float((mod.effective_weight() - w).norm() / w.norm())
                            rows.append(dict(model=model_name, n_slices=ns, setting=setting, chip=chip,
                                             matrix=names[id(mod)], n_weights=w.numel(), rel_err=err))
                print(model_name, ns, setting, flush=True)
    out = ROOT / "results/T1/nominal/slice_weight_error.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("->", out)


def budget_curves(run: str = "smoke20_budget"):
    met = json.loads((ROOT / "results/G3" / run / "metrics.json").read_text())
    sa = met["runs"][0]["sigma_a"]
    model, _ = load_mp20_csp(str(CKPT))
    T = model.beta_scheduler.timesteps
    rows = []
    for t in range(T, 0, -1):
        sx = float(np.interp(t, sa["t"], sa["sx"]))
        sl = float(np.interp(t, sa["t"], sa["sl"]))
        sig = model.sigma_scheduler.sigmas
        sigma_x, adj = float(sig[t]), float(sig[t - 1])
        a = model.beta_scheduler.alphas[t]
        ac = model.beta_scheduler.alphas_cumprod[t]
        c0, c1 = float(1 / torch.sqrt(a)), float((1 - a) / torch.sqrt(1 - ac))
        step_c = 1e-5 * (sigma_x / model.sigma_scheduler.sigma_begin) ** 2
        step_p = sigma_x ** 2 - adj ** 2
        rows.append(dict(
            t=t, sigma_a_x=sx, sigma_a_l=sl,
            std_x_corr=float(np.sqrt(2 * step_c)) if t > 1 else 0.0, s_x_corr=step_c * sx,
            std_x_pred=float(np.sqrt(adj ** 2 * (sigma_x ** 2 - adj ** 2) / sigma_x ** 2)) if t > 1 else 0.0,
            s_x_pred=step_p * sx,
            std_l_pred=float(model.beta_scheduler.sigmas[t]) if t > 1 else 0.0, s_l_pred=c0 * c1 * sl))
    df = pd.DataFrame(rows)
    df["analog_corr"] = df.s_x_corr ** 2 <= df.std_x_corr ** 2
    df["analog_pred"] = (df.s_x_pred ** 2 <= df.std_x_pred ** 2) & (df.s_l_pred ** 2 <= df.std_l_pred ** 2)
    out = ROOT / "results/G3" / run / "budget_curves.csv"
    df.to_csv(out, index=False)
    print("->", out, "analog fraction", (df.analog_corr.sum() + df.analog_pred.sum()) / (2 * len(df)))


def showcase_table(gen: str = "digital_seed0"):
    """Per-sample MACE energies for the GST showcase: mixing energy (series) or E - E_enum_min (doped)."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "showcase"))
    from compositions import GST_SERIES, all_compositions
    base = ROOT / "results" / "showcase_test"
    load = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    ok = lambda r: (r.get("status") == "ok" and r.get("converged") and r.get("min_dist", 0) >= 1.0
                    and 0.5 < r.get("vol_ratio", 0) < 2.0)
    summ = json.loads((base / f"summary_{gen}.json").read_text())
    e_gete, e_sbte = summ["E_GeTe_fu"], summ["E_Sb2Te3_fu"]
    refs = {r["file"]: r for r in load(base / "relax" / "refs.jsonl")}
    enum = json.loads((base / "refs" / "enum.json").read_text())
    meta = json.loads((base / "refs" / "refs.json").read_text())
    comps = all_compositions()
    rows = []
    for r in load(base / "relax" / f"gen_{gen}.jsonl"):
        name = r["file"].split("__")[0]
        c = comps[name]
        row = dict(source="generated", composition=name, group=c["group"], dopant=c["dopant"], mode=c["mode"],
                   sample=r["file"], valid=bool(ok(r)), E=r.get("E"))
        rows.append(row)
    for f, m in ((f, m) for m in enum for f in [m["file"]]):
        r = refs.get(f)
        rows.append(dict(source="enumerated", composition=m["composition"], group="doped", dopant=m["dopant"],
                         mode=comps[m["composition"]]["mode"], sample=f, valid=bool(r and ok(r)),
                         E=r.get("E") if r else None))
    from pymatgen.core import Composition
    for m in meta["gst_chemsys"]:
        r = refs.get(f"{m['id']}.cif")
        red = Composition(m["formula"]).reduced_formula
        name = next((k for k, v in comps.items() if v["group"] == "gst_series"
                     and Composition(v["counts"]).reduced_formula == red), None)
        if name:
            rows.append(dict(source="MP", composition=name, group="gst_series", dopant=None, mode=None,
                             sample=m["id"], valid=bool(r and ok(r)), E=r.get("E") if r else None))
    df = pd.DataFrame(rows)
    df["x_GeTe"] = df.composition.map(lambda n: GST_SERIES[n][0] / sum(GST_SERIES[n]) if n in GST_SERIES else np.nan)
    def emix(row):
        if row.group != "gst_series" or row.E is None or not np.isfinite(row.E):
            return np.nan
        m, n = GST_SERIES[row.composition]
        nat = 2 * m + 5 * n
        return (nat * row.E - m * e_gete - n * e_sbte) / nat * 1000
    df["Emix_meV"] = df.apply(emix, axis=1)
    emin = df[(df.source == "enumerated") & df.valid].groupby("composition").E.min()
    df["dE_vs_enum_min_meV"] = (df.E - df.composition.map(emin)) * 1000
    env = {r["composition"]: r.get("dopant_env_gen") for r in summ["rows"]}
    env_e = {r["composition"]: r.get("dopant_env_enum") for r in summ["rows"]}
    df["dopant_env_best_gen"] = df.composition.map(env)
    df["dopant_env_best_enum"] = df.composition.map(env_e)
    out = base / f"showcase_table_{gen}.csv"
    df.to_csv(out, index=False)
    print("->", out, len(df))


D3_IDS = ["mp-32", "mp-104", "mp-19", "mp-938", "mp-1201", "mp-29641", "mp-1224350", "mp-1224375", "mp-1212511"]


def d3_final(tag: str = "gpu_dual_am", slice_id: str = "mp-1224375"):
    """results/D3/final/{nmae.csv, nscf.csv, slices_<id>.npz} for Fig. 3 (D3 nominal: chip-averaged density).

    NMAE: digital and single chips from <tag>_<id>/per_structure.csv, 3-chip mean from <tag>_mean3_lp_<id>/mean.csv.
    NSCF: <tag>_lp (digital, chips, exact grid, atomic superposition) + <tag>_mean3_lp (chip mean).
    Slice: chip-mean slice = average of the three chip slices (same plane).
    """
    den = ROOT / "results/D3/density"
    out = ROOT / "results/D3/final"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mid in D3_IDS:
        r = pd.read_csv(den / f"{tag}_{mid}" / "per_structure.csv").iloc[0]
        m = pd.read_csv(den / f"{tag}_mean3_lp_{mid}" / "mean.csv").iloc[0]
        chips = [r[f"nmae_analog_chip{k}"] for k in range(3)]
        rows.append({"material_id": mid, "formula": r.formula, "n_points": r.n_points, "nmae_digital": r.nmae_digital,
                     **{f"nmae_chip{k}": c for k, c in enumerate(chips)},
                     "nmae_chip_mean": float(np.mean(chips)), "nmae_chip_std": float(np.std(chips)),
                     "nmae_mean3": m.nmae_mean3})
    pd.DataFrame(rows).to_csv(out / "nmae.csv", index=False)
    nd = ROOT / "results/D3/nscf/d3"
    single = pd.read_csv(nd / f"{tag}_lp.csv")
    mean3 = pd.read_csv(nd / f"{tag}_mean3_lp.csv")
    nscf = pd.concat([single, mean3[mean3.init == "analog_mean3"]], ignore_index=True)
    nscf = nscf[nscf.material_id.isin(D3_IDS)].merge(pd.DataFrame(rows)[["material_id", "formula"]], on="material_id")
    nscf.to_csv(out / "nscf.csv", index=False)
    sl = dict(np.load(den / f"{tag}_{slice_id}" / "slices" / f"{slice_id}.npz"))
    sl["analog_mean3"] = np.mean([sl[f"analog_chip{k}"] for k in range(3)], axis=0)
    np.savez_compressed(out / f"slices_{slice_id}.npz", **sl)
    print("->", out, len(rows), "structures,", len(nscf), "NSCF rows")


def d3_groups(tag: str = "gpu_dual_am"):
    """results/D3/final/groups.csv: D3 per structure for the three chip-mean groups (seeds 0-2, 3-5, 6-8).

    Columns per group g0/g1/g2: density NMAE of the 3-chip mean, NSCF gap / Fermi level errors (meV) and total
    energy error (meV/atom); plus mean, std (ddof=0) over groups and the digital reference.
    """
    den = ROOT / "results/D3/density"
    nd = ROOT / "results/D3/nscf/d3"
    groups = {"g0": f"{tag}_mean3_lp", "g1": f"{tag}_g1_mean3_lp", "g2": f"{tag}_g2_mean3_lp"}
    dig = pd.read_csv(nd / f"{tag}_lp.csv").query("init == 'digital'").set_index("material_id")
    cols = {"dgap_meV": "dgap", "defermi_meV": "defermi", "dE_meV_per_atom": "dE"}
    rows = []
    for mid in D3_IDS:
        row = {"material_id": mid}
        for g, t in groups.items():
            row[f"nmae_{g}"] = pd.read_csv(den / f"{t}_{mid}" / "mean.csv").iloc[0].nmae_mean3
            ns = pd.read_csv(nd / f"{t}.csv").query("init == 'analog_mean3'").set_index("material_id").loc[mid]
            for c, short in cols.items():
                row[f"{short}_{g}"] = ns[c]
        for short in ["nmae", *cols.values()]:
            v = [row[f"{short}_{g}"] for g in groups]
            row[f"{short}_mean"], row[f"{short}_std"] = float(np.mean(v)), float(np.std(v))
        for c, short in cols.items():
            row[f"{short}_digital"] = dig.loc[mid, c]
        rows.append(row)
    out = ROOT / "results/D3/final/groups.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("->", out, len(rows), "structures")


if __name__ == "__main__":
    import sys
    torch.set_num_threads(8)
    which = sys.argv[1:] or ["budget", "slices", "showcase"]
    if "d3_final" in which:
        d3_final()
    if "d3_groups" in which:
        d3_groups()
    if "budget" in which:
        budget_curves()
    if "budget_dual" in which:
        for k in range(3):
            budget_curves(f"eval1000_budget_dual_chip{k}")
    if "slices" in which:
        slice_errors()
    if "showcase" in which:
        showcase_table()
