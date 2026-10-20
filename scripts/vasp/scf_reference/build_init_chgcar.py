"""Build a stage-A (ICHARG=11) CHGCAR from a grid density + optional augmentation occupancies.

  python build_init_chgcar.py GRID_CHGCAR RUN_POSCAR OUT_CHGCAR INFO_JSON [AUG_CHGCAR]

GRID_CHGCAR : CHGCAR whose total-density grid (rho*V, fine grid NGXF x NGYF x NGZF) is used,
              e.g. the converged default run (oracle_*) or, later, an ML-predicted density.
RUN_POSCAR  : POSCAR of the run (header; atom order must match POTCAR).
AUG_CHGCAR  : CHGCAR whose augmentation-occupancy block is appended (atomaug_prep/CHGCAR ->
              ATOMIC occupancies).  Omitted -> grid only (sanity run oracle_noaug).

The grid is written with chgcar_writer (VASP fixed format E17.11), not renormalised (the default
grid already integrates to NELECT; the electron count is recorded in INFO_JSON).  Also checks that
the GRID_CHGCAR / AUG_CHGCAR cells and species equal the run POSCAR.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chgcar_writer as cw  # noqa: E402


def _check_same(a: dict, b: dict, what: str) -> float:
    if list(a["species"]) != list(b["species"]):
        raise SystemExit(f"species/order mismatch: {what}")
    d = float(np.abs(np.asarray(a["lattice"]) - np.asarray(b["lattice"])).max())
    df = np.asarray(a["frac_coords"]) - np.asarray(b["frac_coords"])
    df = float(np.abs(df - np.round(df)).max())
    if d > 1e-5 or df > 1e-5:
        raise SystemExit(f"cell/coords mismatch {what}: dlat={d} dfrac={df}")
    return max(d, df)


def main(grid_chgcar: str, poscar: str, out: str, info_json: str, aug_chgcar: str | None = None) -> dict:
    s_run = cw.read_poscar(poscar)
    s_g, total, _ = cw.read_chgcar(grid_chgcar)
    dev = {"grid_vs_poscar": _check_same(s_g, s_run, "grid CHGCAR vs POSCAR")}
    aug = None
    if aug_chgcar:
        s_a, _, _ = cw.read_chgcar(aug_chgcar)
        dev["aug_vs_poscar"] = _check_same(s_a, s_run, "aug CHGCAR vs POSCAR")
        aug = cw.read_augmentation_text(aug_chgcar)
    info = cw.write_chgcar(out, s_run, total, units="rhoV", comment=f"stageA init: grid={Path(grid_chgcar).parent.name}"
                           f" aug={'atomic(' + Path(aug_chgcar).parent.name + ')' if aug_chgcar else 'none'}", aug_text=aug)
    info.update({"grid_source": str(grid_chgcar), "aug_source": aug_chgcar, "max_struct_dev": dev,
                 "aug_lines": aug.count("\n") if aug else 0})
    json.dump(info, open(info_json, "w"), indent=1)
    print(json.dumps(info))
    return info


if __name__ == "__main__":
    main(*sys.argv[1:])
