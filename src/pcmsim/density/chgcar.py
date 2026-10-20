"""Minimal, dependency-light (numpy only) VASP CHGCAR reader / writer / regridder.

Used for:
  * T2 'mpinit' runs: MP CHGCAR (pymatgen Chgcar json from the MP S3 bucket) -> our NGXF grid.
  * Later: ML-predicted densities (ChargE3Net / EAC-mp) -> CHGCAR for ICHARG=1.

Conventions (VASP):
  * CHGCAR stores rho(r) * V_cell  (rho in e/Angstrom^3, V in Angstrom^3) on the FINE grid
    NGXF x NGYF x NGZF; sum(values) / Ngrid = number of valence electrons (pseudo + compensation).
  * Values are written with x fastest, then y, then z (Fortran order of an (NX, NY, NZ) array).
  * The header is a POSCAR (Direct coordinates), followed by a blank line and "NX NY NZ".
  * PAW augmentation occupancies (TESTED with VASP 6, T2 runs):
      - grid written in a non-VASP number format  -> "WARNING: chargedensity file is incomplete",
        VASP silently falls back to the atomic density (trajectory identical to ICHARG=2);
      - correct format but NO augmentation block   -> no warning, but the supplied grid is NOT used:
        two different grids (own converged density vs MP density) give bit-identical SCF
        trajectories, which start far from both atomic and converged energies (worse than ICHARG=2);
      - grid + augmentation block of the same POSCAR/POTCAR -> grid is used.
    => always pass `aug_text`.  Leakage-free source: the *atomic* occupancies written by a 1-step
       ICHARG=12 / NELM=1 run (re-reading that CHGCAR with ICHARG=1 reproduces ICHARG=2 exactly).
       ICHARG=2 with NELM=1 or NELM=0 does NOT give atomic occupancies (garbage; one case even
       converged to a wrong state).  See read_augmentation_text().

Python >= 3.8, numpy only, so it runs in any cluster env.

Main API
--------
write_chgcar(path, structure, density, grid_shape=None, units="e/A3", n_electrons=None, ...)
    structure : pymatgen Structure, or a POSCAR file path / POSCAR string (recommended: the
                exact POSCAR used for the VASP run, so atom order matches POTCAR), or a dict
                {"lattice": 3x3, "species": [...], "frac_coords": Nx3}.
    density   : array (NX, NY, NZ) [index order x,y,z] or flat array of length NX*NY*NZ
                in x-fastest order (pass grid_shape).  units "e/A3" or "rhoV".
    grid_shape: (NGXF, NGYF, NGZF) target grid. If density has another shape it is
                Fourier-interpolated onto grid_shape.
    n_electrons: if given, rescale so that sum/Ngrid == n_electrons.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np

__all__ = ["read_poscar", "read_chgcar", "read_augmentation_text", "write_chgcar", "fourier_regrid", "load_mp_chgcar_json",
           "parse_fine_grid_from_outcar", "parse_nelect_from_outcar"]


# --------------------------------------------------------------------------- structure helpers
def _poscar_from_text(text: str) -> dict:
    lines = text.splitlines()
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:  # negative scale = target volume
        lat *= (abs(scale) / abs(np.linalg.det(lat))) ** (1 / 3)
    else:
        lat *= scale
    tok5 = lines[5].split()
    if all(t.lstrip("-").isdigit() for t in tok5):
        raise ValueError("VASP4 POSCAR without element line is not supported")
    symbols = tok5
    counts = [int(x) for x in lines[6].split()]
    i = 7
    if lines[i].strip()[0] in "sS":
        i += 1
    cart = lines[i].strip()[0] in "cCkK"
    n = sum(counts)
    coords = np.array([[float(x) for x in lines[i + 1 + k].split()[:3]] for k in range(n)])
    if cart:
        coords = coords @ np.linalg.inv(lat)
    species = [s for s, c in zip(symbols, counts) for _ in range(c)]
    return {"lattice": lat, "species": species, "frac_coords": coords, "comment": lines[0].strip(),
            "header_end": i + 1 + n}


def read_poscar(src) -> dict:
    p = Path(str(src))
    text = p.read_text() if ("\n" not in str(src) and p.exists()) else str(src)
    return _poscar_from_text(text)


def _as_struct_dict(structure) -> dict:
    if isinstance(structure, dict):
        return {"lattice": np.asarray(structure["lattice"], float),
                "species": list(structure["species"]),
                "frac_coords": np.asarray(structure["frac_coords"], float),
                "comment": structure.get("comment", "")}
    if hasattr(structure, "lattice") and hasattr(structure, "frac_coords"):  # pymatgen Structure
        return {"lattice": np.array(structure.lattice.matrix), "frac_coords": np.array(structure.frac_coords),
                "species": [s.specie.symbol for s in structure], "comment": structure.composition.reduced_formula}
    return read_poscar(structure)


def _poscar_text(s: dict, comment: str | None = None) -> str:
    # group consecutive identical species exactly as given (must match the run's POSCAR/POTCAR order)
    syms, counts = [], []
    for sp in s["species"]:
        if syms and syms[-1] == sp:
            counts[-1] += 1
        else:
            syms.append(sp)
            counts.append(1)
    out = [comment or s.get("comment") or "CHGCAR", "   1.00000000000000"]
    for v in s["lattice"]:
        out.append("  " + " ".join(f"{x:21.16f}" for x in v))
    out.append("   " + "   ".join(syms))
    out.append("   " + "   ".join(str(c) for c in counts))
    out.append("Direct")
    for fc in s["frac_coords"]:
        out.append(" " + " ".join(f"{x:19.16f}" for x in fc))
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- grids
def fourier_regrid(data: np.ndarray, new_shape) -> np.ndarray:
    """Periodic Fourier (band-limited) interpolation of a real 3D array onto new_shape.

    Preserves the mean (hence the integral): sum(new)/N_new == sum(old)/N_old (up to the
    dropped Nyquist plane of an even-sized smaller grid, which is set to zero).
    """
    data = np.asarray(data, float)
    new_shape = tuple(int(x) for x in new_shape)
    if data.shape == new_shape:
        return data.copy()
    F = np.fft.fftn(data)
    G = np.zeros(new_shape, complex)
    sl_src, sl_dst = [], []
    for n, m in zip(data.shape, new_shape):
        L = min(n, m)
        kpos = (L - 1) // 2          # strictly below Nyquist of the smaller grid
        kneg = (L - 1) // 2
        idx_pos = np.arange(0, kpos + 1)
        idx_neg = np.arange(-kneg, 0)
        k = np.concatenate([idx_pos, idx_neg])
        sl_src.append(np.mod(k, n))
        sl_dst.append(np.mod(k, m))
    G[np.ix_(*sl_dst)] = F[np.ix_(*sl_src)]
    out = np.fft.ifftn(G).real * (np.prod(new_shape) / np.prod(data.shape))
    return out


# --------------------------------------------------------------------------- read / write
def read_chgcar(path, read_diff: bool = False):
    """Return (struct_dict, total[NX,NY,NZ] as stored (rho*V), diff or None). Ignores augmentation."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        text = f.read()
    lines = text.splitlines()
    s = _poscar_from_text(text)
    i = s["header_end"]
    while not lines[i].strip():
        i += 1
    shape = tuple(int(x) for x in lines[i].split())
    n = int(np.prod(shape))
    vals, j = [], i + 1
    while len(vals) < n:
        vals.extend(lines[j].split())
        j += 1
    total = np.array(vals[:n], float).reshape(shape, order="F")
    diff = None
    if read_diff:
        while j < len(lines) and lines[j].split() != [str(x) for x in shape]:
            j += 1
        if j < len(lines):
            vals = []
            j += 1
            while len(vals) < n:
                vals.extend(lines[j].split())
                j += 1
            diff = np.array(vals[:n], float).reshape(shape, order="F")
    return s, total, diff


def read_augmentation_text(path) -> str:
    """Return the raw text of the (first / total-density) augmentation-occupancy section of a
    VASP CHGCAR, i.e. everything after the first grid block up to the next grid header or EOF.
    Use it as `aug_text` in write_chgcar for a CHGCAR with the same atoms (order) and POTCARs."""
    lines = Path(path).read_text().splitlines()
    s = _poscar_from_text("\n".join(lines))
    i = s["header_end"]
    while not lines[i].strip():
        i += 1
    shape_line = lines[i].split()
    n = int(np.prod([int(x) for x in shape_line]))
    cnt, j = 0, i + 1
    while cnt < n:
        cnt += len(lines[j].split())
        j += 1
    out = []
    while j < len(lines) and lines[j].split() != shape_line:
        out.append(lines[j])
        j += 1
    # (intended for ISPIN=1 CHGCARs; for ISPIN=2 the per-atom moment line before the 2nd grid
    #  header would be included, so do not use spin-polarised CHGCARs as aug source)
    if not out or not out[0].lstrip().startswith("augmentation"):
        raise ValueError(f"no augmentation occupancies found in {path}")
    return "\n".join(out) + "\n"


def _fortran_e(v: float) -> str:
    """Fortran E17.11 as written by VASP: ' 0.12345678901E+01' / '-.12345678901E+01'.
    VASP reads CHGCAR grids with a FIXED format (5(1X,E17.11)); Python '%E' output such as
    ' 1.23456789012E+00' is NOT accepted (tested: VASP reports 'chargedensity file is incomplete'
    and silently falls back to the atomic density)."""
    if abs(v) < 1e-99:
        v = 0.0
    t = f"{v:.10E}"
    if v >= 0:
        return f"0.{t[0]}{t[2:12]}E{int(t[13:]) + 1:+03d}"
    return f"-.{t[1]}{t[3:13]}E{int(t[14:]) + 1:+03d}"


def _format_block(arr_flat: np.ndarray) -> str:
    vals = [" " + _fortran_e(v) for v in arr_flat.tolist()]
    return "\n".join("".join(vals[k:k + 5]) for k in range(0, len(vals), 5)) + "\n"


def write_chgcar(path, structure, density, grid_shape=None, units: str = "e/A3",
                 n_electrons: float | None = None, comment: str | None = None,
                 clip_negative: bool = False, aug_text: str | None = None) -> dict:
    """Write a VASP-readable CHGCAR (total density; augmentation occupancies only if aug_text given).

    aug_text: augmentation-occupancy section (read_augmentation_text) of a VASP CHGCAR for the SAME
              POSCAR atom order and POTCARs.  Without it VASP 6 ignores the file (see module doc).

    Returns a dict with diagnostics (grid, volume, electrons before/after renormalisation).
    """
    s = _as_struct_dict(structure)
    vol = abs(np.linalg.det(s["lattice"]))
    dens = np.asarray(density, float)
    if dens.ndim == 1:
        if grid_shape is None:
            raise ValueError("flat density needs grid_shape")
        dens = dens.reshape(tuple(grid_shape), order="F")
    if units == "e/A3":
        rhoV = dens * vol
    elif units == "rhoV":
        rhoV = dens.copy()
    else:
        raise ValueError(units)
    info = {"input_shape": list(rhoV.shape), "volume_A3": vol}
    if grid_shape is not None and tuple(grid_shape) != rhoV.shape:
        rhoV = fourier_regrid(rhoV, grid_shape)
        info["regridded"] = True
    else:
        info["regridded"] = False
    if clip_negative:
        rhoV = np.clip(rhoV, 0, None)
    ne_before = rhoV.sum() / rhoV.size
    info["electrons_before"] = float(ne_before)
    if n_electrons is not None:
        rhoV *= n_electrons / ne_before
    info["electrons_after"] = float(rhoV.sum() / rhoV.size)
    info["grid"] = list(rhoV.shape)
    info["min_rhoV"] = float(rhoV.min())
    with open(path, "w") as f:
        f.write(_poscar_text(s, comment))
        f.write("\n")
        f.write(" {:4d} {:4d} {:4d}\n".format(*rhoV.shape))
        f.write(_format_block(rhoV.ravel(order="F")))
        if aug_text:
            f.write(aug_text)
    info["augmentation"] = bool(aug_text)
    return info


HBAR2_2M = 3.80998  # hbar^2 / 2 m_e in eV A^2


def lowpass_sphere(rhoV: np.ndarray, lattice: np.ndarray, gcut: float) -> np.ndarray:
    """Project a periodic grid density onto plane waves with |G| <= gcut (1/A, G includes 2 pi).

    The DFT density built from wavefunctions with kinetic cutoff ENCUT has no components beyond
    2 * sqrt(ENCUT / (hbar^2/2m)); grid noise (e.g. analog read noise, independent per grid point) is
    white and mostly lies outside that sphere. G = 0 is kept, so the electron count is unchanged.
    """
    rec = 2 * np.pi * np.linalg.inv(np.asarray(lattice, float)).T
    k = np.meshgrid(*[np.fft.fftfreq(n, 1.0 / n) for n in rhoV.shape], indexing="ij")
    g2 = sum((k[0] * rec[0, i] + k[1] * rec[1, i] + k[2] * rec[2, i]) ** 2 for i in range(3))
    return np.real(np.fft.ifftn(np.fft.fftn(rhoV) * (g2 <= gcut ** 2)))


def lowpass_chgcar(src, dst, encut_eV: float, factor: float = 2.0) -> dict:
    """Copy a CHGCAR with the grid density projected onto |G| <= factor * sqrt(ENCUT / (hbar^2/2m));
    the augmentation occupancies are copied unchanged."""
    s, rhoV, _ = read_chgcar(src)
    gcut = factor * np.sqrt(encut_eV / HBAR2_2M)
    out = lowpass_sphere(rhoV, s["lattice"], gcut)
    info = write_chgcar(dst, s, out, units="rhoV", comment=f"lowpass |G|<={gcut:.2f}/A of {Path(src).name}",
                        aug_text=read_augmentation_text(src))
    info.update(gcut_inv_A=float(gcut), removed_rel_l1=float(np.abs(out - rhoV).sum() / np.abs(rhoV).sum()))
    return info


# --------------------------------------------------------------------------- MP / OUTCAR helpers
def load_mp_chgcar_json(path):
    """MP S3 'chgcars/<task_id>.json.gz' -> (struct_dict, total rho*V array (NX,NY,NZ), meta)."""
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    c = d["data"] if "data" in d and "poscar" in d["data"] else d
    st = c["poscar"]["structure"]
    lat = np.array(st["lattice"]["matrix"], float)
    species = [site["species"][0]["element"] for site in st["sites"]]
    frac = np.array([site["abc"] for site in st["sites"]], float)
    tot = c["data"]["total"]
    total = np.array(tot["data"] if isinstance(tot, dict) else tot, float)
    meta = {"task_id": d.get("task_id"), "spin_polarized": "diff" in c["data"], "shape": list(total.shape)}
    return {"lattice": lat, "species": species, "frac_coords": frac}, total, meta


def parse_fine_grid_from_outcar(outcar) -> dict:
    text = Path(outcar).read_text(errors="replace")
    m = re.search(r"dimension x,y,z NGXF=\s*(\d+)\s+NGYF=\s*(\d+)\s+NGZF=\s*(\d+)", text)
    m2 = re.search(r"dimension x,y,z NGX =\s*(\d+)\s+NGY =\s*(\d+)\s+NGZ =\s*(\d+)", text)
    out = {}
    if m:
        out["NGXF"], out["NGYF"], out["NGZF"] = map(int, m.groups())
    if m2:
        out["NGX"], out["NGY"], out["NGZ"] = map(int, m2.groups())
    return out


def parse_nelect_from_outcar(outcar) -> float:
    m = re.search(r"NELECT\s*=\s*([\d.]+)", Path(outcar).read_text(errors="replace"))
    return float(m.group(1)) if m else None
