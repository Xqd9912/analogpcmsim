"""T2 two-stage SCF test: VASP inputs (no VASP is run here).

Re-uses the T2 single-stage settings (T2 single-stage inputs, not included): pymatgen MPStaticSet,
user_potcar_functional="PBE_54", ISPIN=1, EDIFF=1E-6, NELM=200, NCORE=1, LREAL=False, LAECHG=F,
LVHAR=F, no MAGMOM, MPStaticSet KPOINTS (reciprocal_density 100).  All other tags = MPStaticSet
defaults (ENCUT 520, PREC Accurate, ISMEAR -5, ALGO Fast, LASPH True, LMAXMIX 4, LORBIT 11, ...).

Runs per structure (directory = runs/<mid>/<run>):
  default                       single stage, ICHARG=2, LCHARG=T  (reference: steps, TOTEN, CHGCAR, NGXF grid)
  atomaug_prep                  ICHARG=12, NELM=1  (only harvests ATOMIC PAW augmentation occupancies)
  <init>_<stop>/A               stage A: fixed density (ICHARG>=10), ISTART=0, LWAVE=T, LCHARG=F
  <init>_<stop>/B               stage B: ISTART=1, ICHARG=0, EDIFF=1E-6, NELM=200, LWAVE=F, LCHARG=T
  init  in {atomic: A ICHARG=12;  oracle_atomaug: A ICHARG=11, CHGCAR = default grid + atomic aug;
            oracle_trueaug: A ICHARG=11, CHGCAR = full converged default CHGCAR}
  stop  in {full: EDIFF=1E-6 NELM=200;  loose: EDIFF=1E-4 NELM=200;  fixed8: NELM=8 NELMIN=8}
  + sanity (Ge2Sb2Te5 = mp-1224375 only): oracle_noaug_full/{A,B}: A ICHARG=11 with grid-only CHGCAR
    (no augmentation block).

Only init-related tags differ between runs (ICHARG, ISTART, LWAVE, LCHARG, and in stage A EDIFF/NELM/
NELMIN); asserted below and written to runs/inputs_manifest.json.  POTCAR is not written here:
POTCAR.spec lists the symbols, the cluster job concatenates $POTDIR/<sym>/POTCAR.

Output: <base>/runs/<mid>/<run>[/A|/B]/{INCAR,KPOINTS,POSCAR,POTCAR.spec}
        <base>/runs/inputs_manifest.json, runs/structures.txt
"""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPStaticSet

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]        # <base> = work dir {scripts/, runs/, structures/}
RUNS = HERE / "runs"
STRUCTS = HERE / "structures"

COMMON = {"ISPIN": 1, "LCHARG": True, "LAECHG": False, "LWAVE": False, "EDIFF": 1e-6, "NELM": 200,
          "NCORE": 1, "MAGMOM": None, "LVHAR": False}
SANITY_MID = "mp-1224375"                          # Ge2Sb2Te5

INITS = {"atomic": 12, "oracle_atomaug": 11, "oracle_trueaug": 11}
STOPS = {"full": dict(EDIFF=1e-6, NELM=200),
         "loose": dict(EDIFF=1e-4, NELM=200),
         # NELMIN=NELM=8 -> exactly 8 electronic steps (incl. the |NELMDL|=5 Davidson delay steps)
         "fixed8": dict(EDIFF=1e-6, NELM=8, NELMIN=8)}
STAGE_A = dict(ISTART=0, LWAVE=True, LCHARG=False)
STAGE_B = dict(ISTART=1, ICHARG=0, EDIFF=1e-6, NELM=200, LWAVE=False, LCHARG=True)   # LCHARG=T as in default (same I/O cost)
ALLOWED_DIFF = {"ICHARG", "ISTART", "LWAVE", "LCHARG", "EDIFF", "NELM", "NELMIN"}


def run_specs(mid: str) -> dict[str, dict]:
    """run name -> INCAR overrides on top of COMMON."""
    sp = {"default": dict(ICHARG=2), "atomaug_prep": dict(ICHARG=12, NELM=1)}
    variants = [(i, s) for i in INITS for s in STOPS]
    if mid == SANITY_MID:
        variants.append(("oracle_noaug", "full"))
    for init, stop in variants:
        ich = INITS.get(init, 11)
        sp[f"{init}_{stop}/A"] = dict(STAGE_A, ICHARG=ich, **STOPS[stop])
        sp[f"{init}_{stop}/B"] = dict(STAGE_B)
    return sp


def main() -> None:
    meta = json.load(open(STRUCTS / "structures.json"))["structures"]
    manifest, mids = {}, []
    for m in meta:
        mid = m["material_id"]
        mids.append(mid)
        st = Structure.from_file(STRUCTS / f"{mid}.cif")
        if (RUNS / mid).exists():
            shutil.rmtree(RUNS / mid)
        ref, diffs = None, {}
        for run, over in run_specs(mid).items():
            vis = MPStaticSet(st, user_potcar_functional="PBE_54", user_incar_settings=dict(COMMON, **over))
            d = RUNS / mid / run
            d.mkdir(parents=True, exist_ok=True)
            incar = vis.incar
            incar.write_file(d / "INCAR")
            vis.kpoints.write_file(d / "KPOINTS")
            vis.poscar.write_file(d / "POSCAR")
            (d / "POTCAR.spec").write_text("\n".join(vis.potcar_symbols) + "\n")
            if ref is None:
                ref = dict(incar)
                kref, pref = str(vis.kpoints), str(vis.poscar)
            assert str(vis.kpoints) == kref and str(vis.poscar) == pref, (mid, run)
            dk = {k for k in set(ref) | set(incar) if ref.get(k) != incar.get(k)}
            assert dk <= ALLOWED_DIFF, (mid, run, dk)
            diffs[run] = {k: incar.get(k) for k in sorted(dk | {"ICHARG"})}
        manifest[mid] = {"formula": m["formula"], "reduced_formula": st.composition.reduced_formula,
                         "nsites": len(st), "e_above_hull_eV_atom": m["e_above_hull_eV_atom"],
                         "potcar_symbols": vis.potcar_symbols, "kpoints": str(vis.kpoints).splitlines()[3],
                         "incar_default": {k: str(v) for k, v in ref.items()},
                         "incar_diff_vs_default": diffs}
        print(mid, m["formula"], len(st), vis.potcar_symbols, manifest[mid]["kpoints"], len(diffs), "runs")
    json.dump(manifest, open(RUNS / "inputs_manifest.json", "w"), indent=1, default=str)
    (RUNS / "structures.txt").write_text("\n".join(mids) + "\n")


if __name__ == "__main__":
    main()
