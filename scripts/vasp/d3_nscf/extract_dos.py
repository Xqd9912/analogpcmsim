"""D3: density of states of the NSCF runs (and SCF reference) for plotting.

VASP's own DOS in these runs is on a 0.1 eV grid (NEDOS = 301), too coarse for band-gap errors of tens of meV, so the
DOS is rebuilt from the eigenvalues (k-point weighted, Gaussian broadening) on a fine grid, with element projections
from the LORBIT = 11 projections. Energies are relative to each run's own VBM (eV); DOS in states / eV / atom.

Layout as in parse_d3.py:
    <root>/d3/<tag_chips>/<mid>/{digital,analog_chip0..2}/vasprun.xml
    <root>/d3/<tag_mean>/<mid>/analog_mean3/vasprun.xml
    <root>/t2/<mid>/default/vasprun.xml               SCF reference
    <root>/t2/<mid>/atomic_full/A/vasprun.xml         NSCF from the atomic superposition density
usage: python extract_dos.py <root> <tag_chips> <tag_mean> <outdir> [--sigma 0.05]
writes <outdir>/dos.csv (long format) and <outdir>/edges.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp import Vasprun

E_GRID = np.arange(-6.0, 4.0 + 1e-9, 0.005)  # eV relative to VBM


def runs_for(root: Path, tag_chips: str, tag_mean: str, mid: str) -> dict:
    r = {"scf": root / "t2" / mid / "default" / "vasprun.xml",
         "atomic": root / "t2" / mid / "atomic_full" / "A" / "vasprun.xml",
         "analog_mean3": root / "d3" / tag_mean / mid / "analog_mean3" / "vasprun.xml"}
    for p in sorted((root / "d3" / tag_chips / mid).iterdir()):
        r[p.name] = p / "vasprun.xml"
    return {k: v for k, v in r.items() if v.exists()}


def dos_of(path: Path, sigma: float) -> tuple[pd.DataFrame, dict]:
    """Gaussian-broadened total and element-projected DOS (states/eV/atom) on E_GRID (eV vs VBM)."""
    v = Vasprun(str(path), parse_potcar_file=False, parse_dos=True, parse_projected_eigen=True)
    bs = v.get_band_structure(efermi="smart")
    vbm, cbm = bs.get_vbm()["energy"], bs.get_cbm()["energy"]
    gap = bs.get_band_gap()["energy"]
    if vbm is None:   # gap closed: reference to the Fermi level instead
        vbm = cbm = v.efermi
    eig = v.eigenvalues[Spin.up][..., 0]                     # (nk, nb)
    w = np.asarray(v.actual_kpoints_weights)                 # sums to 1
    proj = v.projected_eigenvalues[Spin.up].sum(axis=-1)     # (nk, nb, nions), summed over orbitals
    species = [s.symbol for s in v.final_structure.species]
    nat = len(species)
    x = (E_GRID[None, None, :] - (eig - vbm)[..., None]) / sigma
    g = np.exp(-0.5 * x * x) / (sigma * np.sqrt(2 * np.pi)) * w[:, None, None]   # (nk, nb, nE)
    out = {"E_eV": E_GRID, "total": g.sum(axis=(0, 1)) / nat}
    for el in sorted(set(species)):
        idx = [i for i, s in enumerate(species) if s == el]
        out[el] = np.einsum("kb,kbe->e", proj[..., idx].sum(-1), g) / nat
    edges = {"vbm_eV": vbm, "cbm_eV": cbm, "gap_eV": gap, "efermi_eV": v.efermi, "natoms": nat,
             "nkpts": len(w), "formula": v.final_structure.composition.reduced_formula}
    return pd.DataFrame(out), edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("tag_chips")
    ap.add_argument("tag_mean")
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--sigma", type=float, default=0.05, help="Gaussian broadening (eV)")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    dfs, rows = [], []
    for mdir in sorted((a.root / "d3" / a.tag_mean).iterdir()):
        mid = mdir.name
        for init, p in runs_for(a.root, a.tag_chips, a.tag_mean, mid).items():
            d, e = dos_of(p, a.sigma)
            dfs.append(d.assign(material_id=mid, init=init))
            rows.append({"material_id": mid, "init": init, **e, "sigma_eV": a.sigma})
            print(mid, init, f"gap {e['gap_eV']:.4f} eV", flush=True)
    edges = pd.DataFrame(rows)
    ref = edges[edges.init == "scf"].set_index("material_id")["gap_eV"]
    edges["dgap_meV"] = 1000 * (edges.gap_eV - edges.material_id.map(ref))
    edges.to_csv(a.outdir / "edges.csv", index=False)
    cols = ["material_id", "init", "E_eV", "total", "Ge", "Sb", "Te"]
    pd.concat(dfs)[cols].to_csv(a.outdir / "dos.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
