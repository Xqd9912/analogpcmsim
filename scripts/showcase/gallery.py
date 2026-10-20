"""S1 structure gallery: lowest-energy analog-generated structure per (GeTe)m(Sb2Te3)n composition (MACE-relaxed).

env: pcm_models.  usage: python scripts/showcase/gallery.py [--seed 0]
Reads the MACE relaxations of the S1 analog runs (3 chips) and of the digital run, picks per composition the lowest
energy (eV/atom) converged structure over the 3 chips, symmetrises it (spglib, symprec 0.1 A) to the conventional
standard cell and compares it with the MP polymorphs of the same composition and with the digital lowest-energy one.
Also counts, per composition and mode, the relaxed structures that are layered: two equal in-plane vectors
<= 4.8 A (LLL-reduced primitive cell) and one atomic plane per atom along the normal (plane gap > 0.6 A).
Output: results/S1/gallery/{<comp>_{analog,digital}.cif (conventional), <comp>_{analog,digital}_prim.cif (LLL-reduced
primitive, used for drawing), gallery.csv}
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from compositions import GST_SERIES  # noqa: E402

RELAX = ROOT / "results" / "showcase_test" / "relax"
REFS = ROOT / "results" / "showcase_test" / "refs"
SYMPREC = 0.1  # Angstrom


def load(stem: str) -> pd.DataFrame:
    df = pd.DataFrame([json.loads(l) for l in open(RELAX / f"{stem}.jsonl")])
    df = df[(df.status == "ok") & df.converged].copy()
    df["composition"] = df.file.str.split("__").str[0]
    df["path"] = [str(RELAX / f"{stem}_relaxed" / f) for f in df.file]
    return df


def lll_primitive(s: Structure) -> Structure:
    try:
        s = s.get_primitive_structure(tolerance=SYMPREC)
    except Exception:  # noqa: BLE001
        pass
    lat = s.lattice.get_lll_reduced_lattice()
    return Structure(lat, s.species, s.cart_coords, coords_are_cartesian=True, to_unit_cell=True)


def is_layered(s: Structure, gap_min: float = 0.6) -> bool:
    p = lll_primitive(s)
    m, abc = p.lattice.matrix, np.array(p.lattice.abc)
    ia, ib, ic = np.argsort(abc)
    if abs(abc[ia] - abc[ib]) > 0.25 or abc[ib] > 4.8:
        return False
    nrm = np.cross(m[ia], m[ib])
    nrm /= np.linalg.norm(nrm)
    period = abs(m[ic] @ nrm)
    h = np.sort((p.cart_coords @ nrm) % period)
    return int((np.diff(np.r_[h, h[0] + period]) > gap_min).sum()) == len(p)


def conventional(s: Structure) -> tuple[Structure, str, int]:
    sga = SpacegroupAnalyzer(s, symprec=SYMPREC)
    return sga.get_conventional_standard_structure(), sga.get_space_group_symbol(), sga.get_space_group_number()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analog_stems", nargs="*", default=None,
                    help="relax jsonl stems to pool for analog (default gen_analog_chip{0,1,2}_seed<seed>)")
    ap.add_argument("--digital_stems", nargs="*", default=None, help="default gen_digital_seed<seed>")
    ap.add_argument("--out", default="gallery", help="output dir under results/S1/")
    a = ap.parse_args()
    od = ROOT / "results" / "S1" / a.out
    od.mkdir(parents=True, exist_ok=True)
    a_stems = a.analog_stems or [f"gen_analog_chip{k}_seed{a.seed}" for k in range(3)]
    d_stems = a.digital_stems or [f"gen_digital_seed{a.seed}"]
    an = pd.concat([load(s).assign(chip=int(re.search(r"chip(\d)", s).group(1))) for s in a_stems])
    dg = pd.concat([load(s) for s in d_stems])
    json.dump({"analog_stems": a_stems, "digital_stems": d_stems}, open(od / "sources.json", "w"), indent=1)
    exact = json.load(open(REFS / "refs.json"))["exact"]
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    rows = []
    for comp, (m, n) in GST_SERIES.items():
        if not ((an.composition == comp).any() and (dg.composition == comp).any()):
            continue   # composition not generated in these runs
        ba = an[an.composition == comp].nsmallest(1, "E").iloc[0]
        bd = dg[dg.composition == comp].nsmallest(1, "E").iloc[0]
        sa, sd = Structure.from_file(ba.path), Structure.from_file(bd.path)
        ca, sg_a, sgn_a = conventional(sa)
        cd, sg_d, _ = conventional(sd)
        ca.to(filename=str(od / f"{comp}_analog.cif"))
        cd.to(filename=str(od / f"{comp}_digital.cif"))
        lll_primitive(sa).to(filename=str(od / f"{comp}_analog_prim.cif"))
        lll_primitive(sd).to(filename=str(od / f"{comp}_digital_prim.cif"))
        nl = {mode: (sum(is_layered(Structure.from_file(p)) for p in df[df.composition == comp].path),
                     int((df.composition == comp).sum())) for mode, df in (("analog", an), ("digital", dg))}
        mp = [r["id"] for r in exact.get(comp, []) if sm.fit(sa, Structure.from_file(REFS / "mp_exact" / f"{r['id']}.cif"))]
        rows.append({"composition": comp, "m": m, "n": n, "x_GeTe": m / (m + n), "natoms": len(sa),
                     "chip": int(ba.chip), "sample": ba.file, "E_analog": ba.E, "E_digital": bd.E,
                     "dE_analog_digital_meV": 1e3 * (ba.E - bd.E), "spg_analog": sg_a, "spg_number_analog": sgn_a,
                     "spg_digital": sg_d, "same_as_digital": bool(sm.fit(sa, sd)),
                     "layered_best_analog": is_layered(sa), "layered_best_digital": is_layered(sd),
                     "n_layered_analog": nl["analog"][0], "n_converged_analog": nl["analog"][1],
                     "n_layered_digital": nl["digital"][0], "n_converged_digital": nl["digital"][1],
                     "n_mp_polymorphs": len(exact.get(comp, [])), "mp_match": ";".join(mp),
                     "a_conv": ca.lattice.a, "c_conv": ca.lattice.c, "Z_conv": len(ca) // len(sa)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(od / "gallery.csv", index=False)
    print("->", od)


if __name__ == "__main__":
    main()
