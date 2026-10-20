"""Enumerate reference doped structures on the MP Ge2Sb2Te5 hosts (9-atom cells).

For each host polymorph and dopant X:
  substitution: X on every symmetry-distinct site of SUB_SITE[X]
  interstitial: X on symmetry-distinct Voronoi-vertex voids with >= DMIN Angstrom to host atoms
env: mace (pymatgen, scipy). Output: showcase_test/refs/enum/*.cif + enum.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial import Voronoi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compositions import DOPANTS, SUB_SITE  # noqa: E402

REFS = HERE.parents[1] / "results" / "showcase_test" / "refs"
HOSTS = ["mp-3534", "mp-1224407", "mp-1224375"]  # MP Ge2Sb2Te5 polymorphs (9 atoms)
DMIN = 1.6  # Angstrom, minimum void-to-atom distance for interstitial candidates
MAX_INT = 8  # keep at most this many distinct interstitial sites per host (largest voids first)


def void_sites(s: Structure) -> list[np.ndarray]:
    """Frac coords of symmetry-distinct voids from Voronoi vertices of the periodic structure."""
    sc = s.copy()
    sc.make_supercell([3, 3, 3])
    vor = Voronoi(sc.cart_coords)
    frac_sc = sc.lattice.get_fractional_coords(vor.vertices)
    inner = np.all((frac_sc >= 1 / 3) & (frac_sc < 2 / 3), axis=1)
    cand = s.lattice.get_fractional_coords(vor.vertices[inner]) % 1.0
    # distance to nearest host atom
    d = s.lattice.get_all_distances(cand, s.frac_coords).min(axis=1)
    order = np.argsort(-d)
    cand, d = cand[order], d[order]
    cand = cand[d >= DMIN]
    # merge near-duplicates (periodic) within 0.5 A
    kept = []
    for c in cand:
        if not kept or s.lattice.get_all_distances([c], kept).min() > 0.5:
            kept.append(c)
    return kept


def dedupe(structs: list[Structure]) -> list[Structure]:
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False, scale=False)
    groups = sm.group_structures(structs)
    return [g[0] for g in groups]


def main():
    out = REFS / "enum"
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for h in HOSTS:
        host = Structure.from_file(REFS / "mp_gst" / f"{h}.cif")
        assert host.composition.reduced_formula == "Ge2Sb2Te5" and len(host) == 9, h
        sym = SpacegroupAnalyzer(host, symprec=0.1).get_symmetrized_structure()
        voids = void_sites(host)
        for x in DOPANTS:
            # substitution
            subs = []
            for eq in sym.equivalent_indices:
                if str(host[eq[0]].specie) == SUB_SITE[x]:
                    s = host.copy()
                    s.replace(eq[0], x)
                    subs.append(s)
            ints = []
            for v in voids:
                s = host.copy()
                s.append(x, v)
                ints.append(s)
            ints = dedupe(ints)[:MAX_INT] if ints else []
            for mode, lst in (("sub", subs), ("int", ints)):
                for k, s in enumerate(lst):
                    tag = f"GST+{x}_{mode}{SUB_SITE[x] if mode == 'sub' else ''}" if mode == "sub" else f"GST+{x}_int"
                    fn = f"{tag}__{h}_{k}.cif"
                    s.to(filename=str(out / fn))
                    meta.append({"file": fn, "composition": tag, "host": h, "dopant": x, "mode": mode, "site": k})
        print(h, "voids:", len(voids), "sym-distinct sites:", len(sym.equivalent_indices))
    (REFS / "enum.json").write_text(json.dumps(meta, indent=1))
    print(len(meta), "enumerated reference structures")


if __name__ == "__main__":
    main()
