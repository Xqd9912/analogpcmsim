"""Summarise the showcase feasibility test.

env: mace (pymatgen).  usage: python analyze.py --gen digital_seed0
Validity: lattice ok, min_dist0 >= 0.5 A, MACE relax converged, final min_dist >= 1.0 A, 0.5 < V/V0 < 2.
GST series: mixing energy vs GeTe + Sb2Te3 (MACE, lowest MP polymorph of each end member),
            match to MP polymorphs (StructureMatcher defaults, after relaxation).
Doped: E_gen_min - E_enum_min (enumerated substitutions/interstitials on MP Ge2Sb2Te5 hosts),
       dopant coordination (CrystalNN) of the best generated and best enumerated structure.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "results" / "showcase_test"
sys.path.insert(0, str(HERE))
from compositions import ENDPOINTS, GST_SERIES, all_compositions  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def valid(r):
    return (r.get("status") == "ok" and r.get("converged") and r.get("min_dist", 0) >= 1.0
            and 0.5 < r.get("vol_ratio", 0) < 2.0)


def dopant_env(s: Structure, x: str) -> str:
    cnn = CrystalNN()
    out = []
    for i, site in enumerate(s):
        if str(site.specie) == x:
            nn = cnn.get_nn_info(s, i)
            sp = sorted(str(n["site"].specie) for n in nn)
            out.append(f"CN{len(nn)}:" + "".join(f"{e}{sp.count(e)}" for e in sorted(set(sp))))
    return ";".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="digital_seed0")
    a = ap.parse_args()
    comps = all_compositions()
    refs = {r["file"]: r for r in load_jsonl(ROOT / "relax" / "refs.jsonl")}
    gen = load_jsonl(ROOT / "relax" / f"gen_{a.gen}.jsonl")
    rdir_ref = ROOT / "relax" / "refs_relaxed"
    rdir_gen = ROOT / "relax" / f"gen_{a.gen}_relaxed"
    meta = json.loads((ROOT / "refs" / "refs.json").read_text())
    enum = {m["file"]: m for m in json.loads((ROOT / "refs" / "enum.json").read_text())}
    sm = StructureMatcher()

    # --- end members and MP polymorph energies
    mp_E = defaultdict(list)  # reduced formula -> [(E/atom, file)]
    for m in meta["gst_chemsys"]:
        r = refs.get(f"{m['id']}.cif")
        if r and valid(r):
            mp_E[Structure.from_file(rdir_ref / r["file"]).composition.reduced_formula].append((r["E"], r["file"]))
    e_gete = min(mp_E["GeTe"])[0] * 2
    e_sbte = min(mp_E["Sb2Te3"])[0] * 5

    def e_mix(m, n, e_atom):
        nat = 2 * m + 5 * n
        return (nat * e_atom - m * e_gete - n * e_sbte) / nat * 1000  # meV/atom

    by_comp = defaultdict(list)
    for r in gen:
        by_comp[r["file"].split("__")[0]].append(r)

    rows = []
    for name, c in comps.items():
        rs = by_comp.get(name, [])
        ok = [r for r in rs if valid(r)]
        row = {"composition": name, "group": c["group"], "n": len(rs), "n_valid": len(ok)}
        if ok:
            best = min(ok, key=lambda r: r["E"])
            row["E_gen_min"] = best["E"]
            s_best = Structure.from_file(rdir_gen / best["file"])
            row["best_sample"] = best["file"]
        if c["group"] in ("gst_series", "gst_end") and ok:
            m, n = {**GST_SERIES, **ENDPOINTS}[name]
            row["Emix_gen_min_meV"] = e_mix(m, n, best["E"])
            row["Emix_gen_all_meV"] = sorted(round(e_mix(m, n, r["E"]), 1) for r in ok)
            red = s_best.composition.reduced_formula
            if mp_E.get(red):
                e_mp, f_mp = min(mp_E[red])
                row["Emix_mp_min_meV"] = e_mix(m, n, e_mp)
                row["dE_gen_minus_mp_meV"] = (best["E"] - e_mp) * 1000
                mp_structs = [Structure.from_file(rdir_ref / f) for _, f in mp_E[red]]
                n_match = 0
                for r in ok:
                    sg = Structure.from_file(rdir_gen / r["file"])
                    if any(sm.fit(sg, t) for t in mp_structs):
                        n_match += 1
                row["n_match_any_mp_polymorph"] = n_match
        if c["group"] == "doped" and ok:
            x = c["dopant"]
            er = [refs[f] for f, m in enum.items() if m["composition"] == name and f in refs and valid(refs[f])]
            if er:
                eb = min(er, key=lambda r: r["E"])
                row["E_enum_min"] = eb["E"]
                row["n_enum_valid"] = len(er)
                row["dE_gen_minus_enum_meV"] = (best["E"] - eb["E"]) * 1000
                row["n_gen_within_10meV_of_enum"] = sum((r["E"] - eb["E"]) * 1000 <= 10 for r in ok)
                row["n_gen_below_enum"] = sum((r["E"] - eb["E"]) * 1000 < -2 for r in ok)
                s_enum = Structure.from_file(rdir_ref / eb["file"])
                row["best_gen_matches_best_enum"] = bool(sm.fit(s_best, s_enum))
                row["best_enum"] = eb["file"]
                try:
                    row["dopant_env_gen"] = dopant_env(s_best, x)
                    row["dopant_env_enum"] = dopant_env(s_enum, x)
                except Exception as e:  # noqa: BLE001
                    row["dopant_env_error"] = repr(e)[:100]
        rows.append(row)

    out = {"gen": a.gen, "E_GeTe_fu": e_gete, "E_Sb2Te3_fu": e_sbte, "rows": rows}
    (ROOT / f"summary_{a.gen}.json").write_text(json.dumps(out, indent=1, default=float))
    keys = ["composition", "n", "n_valid", "Emix_gen_min_meV", "Emix_mp_min_meV", "dE_gen_minus_mp_meV",
            "n_match_any_mp_polymorph", "n_enum_valid", "dE_gen_minus_enum_meV", "n_gen_within_10meV_of_enum",
            "n_gen_below_enum", "best_gen_matches_best_enum", "dopant_env_gen", "dopant_env_enum"]
    for r in rows:
        print({k: (round(r[k], 1) if isinstance(r.get(k), float) else r.get(k)) for k in keys if k in r})


if __name__ == "__main__":
    main()
