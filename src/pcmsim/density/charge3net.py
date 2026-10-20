"""ChargE3Net (MP e3 model) loading and probe-grid inference.

Upstream code: third_party/charge3net (AIforGreatGood/charge3net @ 18290c6, unmodified).
Adaptations made here instead of in third_party:
  * F.one_hot(...) is cast to float (upstream feeds an int one-hot into a tensor product,
    which fails with e3nn 0.5.8); applied by wrapping the module's `F` namespace.
  * asap3 is optional: fall back to the repo's own ASE neighbour-list wrapper (same interface).
Units: positions in Angstrom, density in e/Angstrom^3.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
C3N = ROOT / "third_party" / "charge3net"
CKPT = C3N / "models" / "charge3net_mp.pt"
CFG = dict(num_interactions=3, num_neighbors=20, mul=500, lmax=4, cutoff=4.0, basis="gaussian", num_basis=20)

if str(C3N) not in sys.path:
    sys.path.insert(0, str(C3N))

import src.charge3net.data.collate as c3n_collate  # noqa: E402
import src.charge3net.models.e3 as c3n_e3  # noqa: E402

_F = c3n_e3.F


class _FloatOneHotF(types.ModuleType):
    def __getattr__(self, name):
        return getattr(_F, name)

    @staticmethod
    def one_hot(*a, **k):
        return _F.one_hot(*a, **k).float()


c3n_e3.F = _FloatOneHotF("F")
try:
    import asap3  # noqa: F401
except ImportError:
    c3n_collate.asap3 = types.SimpleNamespace(FullNeighborList=c3n_collate.AseNeigborListWrapper)


def build_model(device: str | torch.device = "cpu") -> torch.nn.Module:
    model = c3n_e3.E3DensityModel(**CFG)
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"], strict=True)
    return model.eval().to(device)


def atom_graph(atoms, cutoff: float = CFG["cutoff"]):
    """Atom-atom graph (computed once per structure); returns (dict of tensors, neighbourlist, inv_cell_T)."""
    edges, disp, nl, inv_cell_T = c3n_collate.atoms_to_graph(atoms, cutoff)
    dt = torch.get_default_dtype()
    d = {
        "nodes": torch.tensor(atoms.get_atomic_numbers()),
        "atom_edges": torch.tensor(np.concatenate(edges, 0)),
        "atom_edges_displacement": torch.tensor(np.concatenate(disp, 0), dtype=dt),
        "atom_xyz": torch.tensor(atoms.get_positions(), dtype=dt),
        "cell": torch.tensor(np.array(atoms.get_cell()), dtype=dt),
    }
    return d, nl, inv_cell_T


def probe_batch(atoms, cart: np.ndarray, graph, cutoff: float = CFG["cutoff"]) -> dict:
    """Collated batch for probe positions `cart` (n, 3) in Angstrom."""
    d, nl, inv_cell_T = graph
    pe, pd = c3n_collate.probes_to_graph(atoms, cart, cutoff, neighborlist=nl, inv_cell_T=inv_cell_T, batch_size=0)
    dt = torch.get_default_dtype()
    res = dict(d)
    res.update({
        "probe_edges": torch.tensor(np.concatenate(pe, 0)),
        "probe_edges_displacement": torch.tensor(np.concatenate(pd, 0), dtype=dt),
        "probe_xyz": torch.tensor(cart, dtype=dt),
    })
    res["num_nodes"] = torch.tensor(res["nodes"].shape[0])
    res["num_atom_edges"] = torch.tensor(res["atom_edges"].shape[0])
    res["num_probe_edges"] = torch.tensor(res["probe_edges"].shape[0])
    res["num_probes"] = torch.tensor(cart.shape[0])
    return c3n_collate.collate_list_of_dicts([res])


class PeriodicProbeGraph:
    """Vectorised probe->atom neighbour search (replaces the per-probe ASE loop of probes_to_graph,
    which costs ~30 ms per probe). Same conventions as charge3net's probes_to_graph:
    edges (atom_index, probe_index); displacement = integer cell offsets n such that the neighbouring
    atom image sits at atom_pos + n @ cell. Neighbours: all atom images with distance <= cutoff."""

    def __init__(self, atoms, cutoff: float = CFG["cutoff"]):
        from scipy.spatial import cKDTree
        cell = np.array(atoms.get_cell())
        vol = abs(np.linalg.det(cell))
        heights = [vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)]
        reps = [int(np.ceil(cutoff / h)) + 1 for h in heights]
        shifts = np.array(np.meshgrid(*[np.arange(-r, r + 1) for r in reps], indexing="ij")).reshape(3, -1).T
        pos = atoms.get_positions()
        self.img_atom = np.repeat(np.arange(len(atoms)), len(shifts))
        self.img_shift = np.tile(shifts, (len(atoms), 1)).astype(np.float64)
        img_pos = pos[self.img_atom] + self.img_shift @ cell
        self.tree = cKDTree(img_pos)
        self.cutoff = cutoff

    def query(self, cart: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from scipy.spatial import cKDTree
        pairs = self.tree.sparse_distance_matrix(cKDTree(cart), self.cutoff, output_type="ndarray")
        img, probe = pairs["i"], pairs["j"]
        order = np.lexsort((img, probe))
        img, probe = img[order], probe[order]
        edges = np.stack([self.img_atom[img], probe], axis=1)
        return edges, self.img_shift[img]


def probe_batch_fast(atoms, cart: np.ndarray, graph, pgraph: PeriodicProbeGraph) -> dict:
    d = graph[0]
    edges, disp = pgraph.query(cart)
    dt = torch.get_default_dtype()
    res = dict(d)
    res.update({
        "probe_edges": torch.tensor(edges),
        "probe_edges_displacement": torch.tensor(disp, dtype=dt),
        "probe_xyz": torch.tensor(cart, dtype=dt),
    })
    res["num_nodes"] = torch.tensor(res["nodes"].shape[0])
    res["num_atom_edges"] = torch.tensor(res["atom_edges"].shape[0])
    res["num_probe_edges"] = torch.tensor(res["probe_edges"].shape[0])
    res["num_probes"] = torch.tensor(cart.shape[0])
    return c3n_collate.collate_list_of_dicts([res])


def prepare_batches(atoms, cart: np.ndarray, batch_probes: int = 2500) -> list[dict]:
    """Collated CPU batches for all probes (build once, reuse for digital and every chip)."""
    graph = atom_graph(atoms)
    pg = PeriodicProbeGraph(atoms)
    return [probe_batch_fast(atoms, cart[s:s + batch_probes], graph, pg) for s in range(0, len(cart), batch_probes)]


@torch.no_grad()
def predict_batches(model, batches: list[dict], device: str | torch.device = "cpu") -> np.ndarray:
    out, rep = [], None
    for b in batches:
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
        if rep is None:
            rep = model.atom_model(b)
        out.append(model.probe_model(b, rep).reshape(-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_points(model, atoms, cart: np.ndarray, batch_probes: int = 2500,
                   device: str | torch.device = "cpu") -> np.ndarray:
    """Density (e/A^3) at Cartesian probe points; the atom representation is computed once."""
    return predict_batches(model, prepare_batches(atoms, cart, batch_probes), device)


def grid_points(atoms, grid: tuple[int, int, int]) -> np.ndarray:
    """Cartesian coordinates (A) of a VASP-style fine grid (NGXF, NGYF, NGZF), C order."""
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in grid], indexing="ij"), -1).reshape(-1, 3)
    return (ijk / np.array(grid)) @ np.array(atoms.get_cell())


def nmae(pred: np.ndarray, target: np.ndarray) -> float:
    """Normalized MAE sum|pred-target| / sum|target| (uniform grid, so equal to the integral form)."""
    return float(np.abs(pred - target).sum() / np.abs(target).sum())
