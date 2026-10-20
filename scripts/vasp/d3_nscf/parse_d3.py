"""D3: band gap / Fermi level / energy of non-self-consistent runs from predicted densities vs the SCF reference.

Reads local copies (fetch first, see fetch.sh) laid out as
    <root>/d3/<tag>/<mid>/<model>/vasprun.xml        ML-density NSCF runs (ICHARG=11)
    <root>/t2/<mid>/default/vasprun.xml              SCF reference (ICHARG=2)
    <root>/t2/<mid>/oracle_atomaug_full/A/vasprun.xml  NSCF from the exact grid + atomic augmentation
    <root>/t2/<mid>/atomic_full/A/vasprun.xml        NSCF from the atomic superposition density
usage: python parse_d3.py <root> <tag> <out.csv>
"""
import sys
from pathlib import Path

import pandas as pd
from pymatgen.io.vasp import Vasprun


def info(path: Path) -> dict:
    try:
        v = Vasprun(str(path), parse_potcar_file=False, parse_dos=True, parse_projected_eigen=False)
        bs = v.get_band_structure(efermi="smart")
        return {"gap_eV": bs.get_band_gap()["energy"], "efermi_eV": v.efermi, "energy_eV": v.final_energy,
                "natoms": len(v.final_structure)}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:120]}


def main():
    root, tag, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    rows = []
    for mdir in sorted((root / "d3" / tag).iterdir()):
        mid = mdir.name
        ref = info(root / "t2" / mid / "default" / "vasprun.xml")
        runs = {p.name: p / "vasprun.xml" for p in sorted(mdir.iterdir()) if (p / "vasprun.xml").exists()}
        runs["exact_grid"] = root / "t2" / mid / "oracle_atomaug_full" / "A" / "vasprun.xml"
        runs["atomic"] = root / "t2" / mid / "atomic_full" / "A" / "vasprun.xml"
        for name, p in runs.items():
            r = info(p)
            row = {"material_id": mid, "init": name, **r, "gap_scf_eV": ref.get("gap_eV")}
            if "gap_eV" in r and "gap_eV" in ref:
                row["dgap_meV"] = 1000 * (r["gap_eV"] - ref["gap_eV"])
                row["defermi_meV"] = 1000 * (r["efermi_eV"] - ref["efermi_eV"])
                row["dE_meV_per_atom"] = 1000 * (r["energy_eV"] - ref["energy_eV"]) / r["natoms"]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    cols = [c for c in ("material_id", "init", "gap_scf_eV", "gap_eV", "dgap_meV", "defermi_meV", "dE_meV_per_atom")
            if c in df]
    print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
