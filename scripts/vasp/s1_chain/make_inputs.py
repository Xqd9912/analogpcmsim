"""S1 chain: VASP inputs for analog-generated, MACE-relaxed structures (no VASP is run here).

Identical settings to D3 (scripts/vasp/scf_reference/make_twostage_inputs.py): pymatgen MPStaticSet, PBE_54,
ISPIN=1, EDIFF=1E-6, NELM=200, NCORE=1, LREAL=False, LAECHG=F, LVHAR=F, no MAGMOM, MPStaticSet KPOINTS.
Runs per structure (results/S1/vasp_inputs/<id>/<run>/{INCAR,KPOINTS,POSCAR,POTCAR.spec}):
  default               ICHARG=2 SCF (reference density, fine grid, band gap)
  atomaug_prep          ICHARG=12, NELM=1 (atomic PAW augmentation occupancies for the predicted CHGCARs)
  atomic_full/{A,B}     atomic-superposition NSCF (A) + SCF restart (B), as the D3 baseline
  oracle_atomaug_full/A ICHARG=11 template used by d3_nscf/setup.sh for the predicted-density NSCF runs
Copy to <base>/runs/ of the scf_reference work dir on the cluster and run `sbatch scripts/run_twostage_job.sh <id> default atomaug_prep
atomic_full`; then the D3 density / chip-mean / NSCF pipeline applies unchanged.
usage: python scripts/vasp/s1_chain/make_inputs.py
"""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPStaticSet

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/S1/vasp_inputs"
RELAX = ROOT / "results/showcase_test/relax"
# id -> (relaxed CIF, provenance)
STRUCTS = {
    "s1-Ge2Sb2Te5-c2": ("gen_analog_chip2_seed0_relaxed/Ge2Sb2Te5__1.cif",
                        "Ge2Sb2Te5 generated on chip 2 (noise budget), lowest MACE mixing energy -0.2 meV/atom"),
    "s1-GSTC-c1": ("gen_analog_chip1_seed0_relaxed/GST+C_subGe__4.cif",
                   "Ge2Sb2Te5 with C on a Ge site, generated on chip 1, 28 meV/atom below the lowest enumerated configuration"),
}
COMMON = {"ISPIN": 1, "LCHARG": True, "LAECHG": False, "LWAVE": False, "EDIFF": 1e-6, "NELM": 200,
          "NCORE": 1, "MAGMOM": None, "LVHAR": False}
STAGE_A = dict(ISTART=0, LWAVE=True, LCHARG=False, EDIFF=1e-6, NELM=200)
STAGE_B = dict(ISTART=1, ICHARG=0, EDIFF=1e-6, NELM=200, LWAVE=False, LCHARG=True)
RUNS = {"default": dict(ICHARG=2), "atomaug_prep": dict(ICHARG=12, NELM=1),
        "atomic_full/A": dict(STAGE_A, ICHARG=12), "atomic_full/B": dict(STAGE_B),
        "oracle_atomaug_full/A": dict(STAGE_A, ICHARG=11)}
ALLOWED_DIFF = {"ICHARG", "ISTART", "LWAVE", "LCHARG", "EDIFF", "NELM", "NELMIN"}


def main() -> None:
    manifest = {}
    for sid, (cif, note) in STRUCTS.items():
        st = Structure.from_file(RELAX / cif)
        if (OUT / sid).exists():
            shutil.rmtree(OUT / sid)
        ref = None
        for run, over in RUNS.items():
            vis = MPStaticSet(st, user_potcar_functional="PBE_54", user_incar_settings=dict(COMMON, **over))
            d = OUT / sid / run
            d.mkdir(parents=True, exist_ok=True)
            vis.incar.write_file(d / "INCAR")
            vis.kpoints.write_file(d / "KPOINTS")
            vis.poscar.write_file(d / "POSCAR")
            (d / "POTCAR.spec").write_text("\n".join(vis.potcar_symbols) + "\n")
            inc = dict(vis.incar)
            if ref is None:
                ref = inc
            dk = {k for k in set(ref) | set(inc) if ref.get(k) != inc.get(k)}
            assert dk <= ALLOWED_DIFF, (sid, run, dk)
        manifest[sid] = {"source_cif": str(cif), "note": note, "formula": st.composition.formula,
                         "nsites": len(st), "potcar_symbols": vis.potcar_symbols,
                         "kpoints": str(vis.kpoints).splitlines()[3], "incar_default": {k: str(v) for k, v in ref.items()}}
        st.to(filename=str(OUT / sid / f"{sid}.cif"))
        print(sid, st.composition.formula, len(st), vis.potcar_symbols, manifest[sid]["kpoints"])
    json.dump(manifest, open(OUT / "inputs_manifest.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
