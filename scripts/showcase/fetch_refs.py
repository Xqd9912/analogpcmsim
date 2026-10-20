"""Fetch MP reference structures: all Ge-Sb-Te entries (+ GeTe, Sb2Te3 end members)
and any MP entries with exactly the test compositions (novelty check).

env: pcm_models (mp-api). Output: showcase_test/refs/{mp_gst/*.cif, mp_exact/*.cif, refs.json}
"""
import json
import os
import sys
from pathlib import Path

from mp_api.client import MPRester
from pymatgen.core import Composition

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compositions import all_compositions  # noqa: E402

OUT = HERE.parents[1] / "results" / "showcase_test" / "refs"


def main():
    (OUT / "mp_gst").mkdir(parents=True, exist_ok=True)
    (OUT / "mp_exact").mkdir(parents=True, exist_ok=True)
    fields = ["material_id", "formula_pretty", "nsites", "energy_above_hull", "structure", "theoretical"]
    meta = {"gst_chemsys": [], "exact": {}}
    with MPRester(os.environ.get("MP_API_KEY")) as mpr:
        docs = []
        for cs in ["Ge-Sb-Te", "Ge-Te", "Sb-Te"]:
            docs += mpr.materials.summary.search(chemsys=cs, fields=fields)
        for d in docs:
            d.structure.to(filename=str(OUT / "mp_gst" / f"{d.material_id}.cif"))
            meta["gst_chemsys"].append({"id": str(d.material_id), "formula": d.formula_pretty, "nsites": d.nsites,
                                        "e_above_hull": d.energy_above_hull, "theoretical": d.theoretical})
        for name, c in all_compositions().items():
            red = Composition(c["counts"]).reduced_formula
            hits = mpr.materials.summary.search(formula=red, fields=fields)
            meta["exact"][name] = [{"id": str(h.material_id), "nsites": h.nsites,
                                    "e_above_hull": h.energy_above_hull} for h in hits]
            for h in hits:
                h.structure.to(filename=str(OUT / "mp_exact" / f"{h.material_id}.cif"))
    (OUT / "refs.json").write_text(json.dumps(meta, indent=1))
    print(len(meta["gst_chemsys"]), "Ge-Sb-Te chemsys entries")
    for k, v in meta["exact"].items():
        print(f"{k:16s} {Composition(all_compositions()[k]['counts']).reduced_formula:14s} MP hits: "
              + ", ".join(f"{h['id']}({h['nsites']},{h['e_above_hull']:.3f})" for h in v))


if __name__ == "__main__":
    main()
