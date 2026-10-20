"""Parse the T2 two-stage SCF test -> CSV + summary JSON + printed table.

  python parse_twostage.py RUNS_DIR OUT_CSV OUT_SUMMARY_JSON

RUNS_DIR = <base>/runs (layout from make_twostage_inputs.py / run_twostage_job.sh).

Per run (single-stage: default, atomaug_prep; two-stage: <init>_<stop>/{A,B}):
  steps        electronic steps in OSZICAR (one ionic step; includes the NELMDL delay steps)
  toten_eV     last 'free  energy   TOTEN' in OUTCAR (stage B / default only are physical; stage A
               energies are non-self-consistent: Harris-Foulkes for ICHARG=12, fixed-density for 11)
  elapsed_s    OUTCAR 'Elapsed time (sec)';  job_wall_s = walltime_s.txt (mpirun incl. start-up)
  converged    'aborting loop because EDIFF is reached', finished normally, and steps < NELM
Two-stage row: steps_A, steps_B, steps_total = A + B, wall_total = elapsed A + B,
  dE_per_atom_eV = |TOTEN(B) - TOTEN(default)| / natoms,
  step_reduction = 1 - steps_total / steps_default (positive = cheaper than single-stage ICHARG=2),
  wall_reduction = 1 - wall_total / elapsed_default,
  checks: B really restarted from WAVECAR (OUTCAR ISTART=1, no 'WAVECAR not read'/ISTART reset),
          A read the CHGCAR (no 'incomplete' warning), A trajectory identical to the atomic A run
          with the same stopping rule (=> ICHARG=11 silently ignored the supplied CHGCAR),
          B first-step energy offset |E_B(1) - TOTEN(B)| (quality of the stage-A orbitals).
For oracle_atomaug the atomaug_prep cost (needed for any grid-only/ML density) is reported separately
(wall_reduction_incl_prep).
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

INITS = ["atomic", "oracle_atomaug", "oracle_trueaug", "oracle_noaug"]
STOPS = ["full", "loose", "fixed8"]
STEP = re.compile(r"^\s*(DAV|RMM|CG|SDA|DIA|ORT|DMP|EDD):\s*(\d+)\s+(\S+)")


def parse_run(d: Path) -> dict:
    r = {"steps": None, "toten_eV": None, "e0_eV": None, "e_first_eV": None, "elapsed_s": None,
         "job_wall_s": None, "converged": False, "finished": False, "efermi_eV": None, "istart": None,
         "icharg": None, "nelm": None, "ediff": None, "natoms": None, "warn_incomplete": False,
         "warn_wavecar": False, "osz_energies": []}
    osz, out, vo = d / "OSZICAR", d / "OUTCAR", d / "vasp.out"
    if osz.exists():
        en = []
        for ln in osz.read_text().splitlines():
            m = STEP.match(ln)
            if m:
                en.append(float(m.group(3)))
            m = re.search(r"F=\s*(\S+)\s+E0=\s*(\S+)", ln)
            if m:
                r["e0_eV"] = float(m.group(2))
        r["steps"] = len(en) if en else None
        r["e_first_eV"] = en[0] if en else None
        r["osz_energies"] = en
    if out.exists():
        t = out.read_text(errors="replace")
        tot = re.findall(r"free  energy   TOTEN  =\s*(\S+)", t)
        r["toten_eV"] = float(tot[-1]) if tot else None
        m = re.search(r"Elapsed time \(sec\):\s*(\S+)", t)
        r["elapsed_s"] = float(m.group(1)) if m else None
        r["finished"] = "General timing and accounting" in t
        for key, pat, typ in [("istart", r"ISTART\s*=\s*(\d+)", int), ("icharg", r"ICHARG\s*=\s*(\d+)", int),
                              ("nelm", r"NELM\s*=\s*(\d+)", int), ("ediff", r"EDIFF\s*=\s*(\S+)", float),
                              ("natoms", r"NIONS\s*=\s*(\d+)", int)]:
            m = re.search(pat, t)
            r[key] = typ(m.group(1)) if m else None
        ef = re.findall(r"E-fermi\s*:\s*(\S+)", t)
        r["efermi_eV"] = float(ef[-1]) if ef else None
        r["warn_incomplete"] = "incomplete" in t.lower()
        # VASP 6.3.2 prints 'because EDIFF is reached' even when it stops at NELM (seen for NELM=1),
        # so also require steps < NELM
        r["converged"] = (("aborting loop because EDIFF is reached" in t) and "EDIFF was not reached" not in t
                          and r["finished"] and r["steps"] is not None and r["nelm"] is not None
                          and r["steps"] < r["nelm"])
    if vo.exists():
        v = vo.read_text(errors="replace")
        r["warn_incomplete"] = r["warn_incomplete"] or ("incomplete" in v.lower())
        r["warn_wavecar"] = bool(re.search(r"WAVECAR not read|ISTART.*reset|could not read WAVECAR|ERROR.*WAVECAR", v, re.I))
    w = d / "walltime_s.txt"
    if w.exists():
        r["job_wall_s"] = float(w.read_text().split()[0])
    return r


def _f(x, n=3):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main(runs: str, out_csv: str, out_json: str) -> None:
    runs = Path(runs)
    rows, by = [], {}
    for sd in sorted(p for p in runs.iterdir() if p.is_dir() and (p / "default").is_dir()):
        mid = sd.name
        dflt = parse_run(sd / "default")
        prep = parse_run(sd / "atomaug_prep")
        if dflt["steps"] is None:
            continue
        nat = dflt["natoms"]
        base = {"material_id": mid, "natoms": nat}
        rows.append(dict(base, run="default", init="default", stop="-", steps_A=0, steps_B=dflt["steps"],
                         steps_total=dflt["steps"], toten_eV=dflt["toten_eV"], wall_A_s=0.0, wall_B_s=dflt["elapsed_s"],
                         wall_total_s=dflt["elapsed_s"], job_wall_total_s=dflt["job_wall_s"],
                         converged_B=dflt["converged"], dE_per_atom_eV=0.0, step_reduction=0.0, wall_reduction=0.0))
        by.setdefault(mid, {})["default"] = rows[-1]
        atomicA = {s: parse_run(sd / f"atomic_{s}" / "A") for s in STOPS}
        for init in INITS:
            for stop in STOPS:
                rd = sd / f"{init}_{stop}"
                if not (rd / "B" / "OSZICAR").exists():
                    continue
                a, b = parse_run(rd / "A"), parse_run(rd / "B")
                if a["steps"] is None or b["steps"] is None:
                    continue
                wa, wb = a["elapsed_s"], b["elapsed_s"]
                wt = (wa + wb) if (wa is not None and wb is not None) else None
                row = dict(base, run=f"{init}_{stop}", init=init, stop=stop, steps_A=a["steps"], steps_B=b["steps"],
                           steps_total=a["steps"] + b["steps"], toten_eV=b["toten_eV"], wall_A_s=wa, wall_B_s=wb,
                           wall_total_s=wt,
                           job_wall_total_s=(a["job_wall_s"] or 0) + (b["job_wall_s"] or 0),
                           converged_A=a["converged"], finished_A=a["finished"], converged_B=b["converged"],
                           icharg_A=a["icharg"], ediff_A=a["ediff"], nelm_A=a["nelm"],
                           efermi_A_eV=a["efermi_eV"], toten_A_eV=a["toten_eV"],
                           A_warn_incomplete=a["warn_incomplete"],
                           A_identical_to_atomic=(init != "atomic" and bool(atomicA[stop]["osz_energies"])
                                                  and a["osz_energies"] == atomicA[stop]["osz_energies"]),
                           B_istart=b["istart"], B_warn_wavecar=b["warn_wavecar"],
                           B_first_step_offset_eV=(abs(b["e_first_eV"] - b["toten_eV"])
                                                   if b["e_first_eV"] is not None and b["toten_eV"] is not None else None),
                           dE_per_atom_eV=(abs(b["toten_eV"] - dflt["toten_eV"]) / nat
                                           if b["toten_eV"] is not None and dflt["toten_eV"] is not None and nat else None),
                           step_reduction=1 - (a["steps"] + b["steps"]) / dflt["steps"],
                           stepB_reduction=1 - b["steps"] / dflt["steps"],
                           wall_reduction=(1 - wt / dflt["elapsed_s"]) if wt and dflt["elapsed_s"] else None,
                           prep_wall_s=prep["elapsed_s"] if init == "oracle_atomaug" else 0.0)
                if init == "oracle_atomaug" and wt and prep["elapsed_s"] and dflt["elapsed_s"]:
                    row["wall_reduction_incl_prep"] = 1 - (wt + prep["elapsed_s"]) / dflt["elapsed_s"]
                rows.append(row)
                by[mid][row["run"]] = row

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def stats(xs):
        xs = [x for x in xs if x is not None]
        return {"mean": st.mean(xs), "median": st.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)} if xs else None

    summ = {"n_structures": len(by), "per_variant": {}, "default": {
        "steps": stats([r["default"]["steps_total"] for r in by.values()]),
        "wall_s": stats([r["default"]["wall_total_s"] for r in by.values()]),
        "n_converged": sum(r["default"]["converged_B"] for r in by.values())}}
    for init in INITS:
        for stop in STOPS:
            name = f"{init}_{stop}"
            rr = [by[m][name] for m in by if name in by[m]]
            if not rr:
                continue
            ok = [r for r in rr if r["converged_B"] and by[r["material_id"]]["default"]["converged_B"]]
            summ["per_variant"][name] = {
                "n_runs": len(rr), "n_converged_B": len(ok), "n_converged_A": sum(bool(r["converged_A"]) for r in rr),
                "steps_A": stats([r["steps_A"] for r in ok]), "steps_B": stats([r["steps_B"] for r in ok]),
                "steps_total": stats([r["steps_total"] for r in ok]),
                "step_reduction_vs_default": stats([r["step_reduction"] for r in ok]),
                "stepB_only_reduction_vs_default": stats([r["stepB_reduction"] for r in ok]),
                "wall_total_s": stats([r["wall_total_s"] for r in ok]),
                "wall_reduction_vs_default": stats([r["wall_reduction"] for r in ok]),
                "wall_reduction_incl_prep": stats([r.get("wall_reduction_incl_prep") for r in ok]),
                "dE_per_atom_eV": stats([r["dE_per_atom_eV"] for r in ok]),
                "n_A_identical_to_atomic": sum(bool(r["A_identical_to_atomic"]) for r in rr),
                "n_A_warn_incomplete": sum(bool(r["A_warn_incomplete"]) for r in rr),
                "n_B_not_restarted": sum((r["B_istart"] != 1) or bool(r["B_warn_wavecar"]) for r in rr),
                "B_first_step_offset_eV": stats([r["B_first_step_offset_eV"] for r in rr])}
    summ["per_structure_steps"] = {m: {k: (v["steps_A"], v["steps_B"]) for k, v in d.items()} for m, d in by.items()}
    summ["notes"] = {
        "steps": "electronic steps of the single ionic step (OSZICAR); two-stage total = steps(A) + steps(B)",
        "wall": "OUTCAR 'Elapsed time (sec)'; two-stage = A + B (WAVECAR write/read included); all runs of a "
                "structure on the same node/DCU in one job",
        "prep": "atomaug_prep (ICHARG=12, NELM=1) cost is not in wall_total; see wall_reduction_incl_prep (oracle_atomaug)",
        "dE": "|TOTEN(B) - TOTEN(default)| per atom; EDIFF=1e-6 eV is a step-to-step criterion"}
    json.dump(summ, open(out_json, "w"), indent=1)

    print(f"{'variant':24s} {'n':>3s} {'stepsA':>7s} {'stepsB':>7s} {'total':>7s} {'dflt':>6s} "
          f"{'red.mean':>8s} {'red.med':>8s} {'wallred':>8s} {'max|dE|/at':>10s} {'A=atomic':>8s}")
    dmean = summ["default"]["steps"]["mean"] if summ["default"]["steps"] else None
    for name, v in summ["per_variant"].items():
        g = lambda k, s="mean": v[k][s] if v[k] else None  # noqa: E731
        de_max = g("dE_per_atom_eV", "max")
        de_txt = f"{de_max:.1e}" if de_max is not None else "-"
        print(f"{name:24s} {v['n_converged_B']:>3d} {_f(g('steps_A'), 1):>7s} {_f(g('steps_B'), 1):>7s} "
              f"{_f(g('steps_total'), 1):>7s} {_f(dmean, 1):>6s} {_f(g('step_reduction_vs_default')):>8s} "
              f"{_f(g('step_reduction_vs_default', 'median')):>8s} {_f(g('wall_reduction_vs_default')):>8s} "
              f"{de_txt:>10s} {v['n_A_identical_to_atomic']:>8d}")


if __name__ == "__main__":
    main(*sys.argv[1:])
