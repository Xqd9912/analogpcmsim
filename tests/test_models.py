"""Conversion of DiffCSP and ChargE3Net to analog: coverage and exact digital equivalence."""
from pathlib import Path

import numpy as np
import pytest
import torch

from pcmsim.analog import (AnalogFCTPLinear, AnalogMatrix, analog_matrices, calibrate, digital_mode, instantiate_chip,
                           set_read_seed)
from pcmsim.convert import layer_table, n_analog_params, to_analog_charge3net, to_analog_diffcsp
from pcmsim.device import PCMConfig

ROOT = Path(__file__).resolve().parents[1]
o3 = pytest.importorskip("e3nn.o3")


def test_fctp_linear_equivalence_and_equivariance():
    torch.manual_seed(0)
    tp = o3.FullyConnectedTensorProduct("6x0e+4x1o+2x2e", "1x0e", "5x0e+3x1o+2x2e+1x0o")
    x = torch.randn(9, tp.irreps_in1.dim)
    ones = torch.ones(9, 1)
    ref = tp(x, ones)
    a = AnalogFCTPLinear(tp, PCMConfig.from_yaml(read_rel=0.0))
    m = torch.nn.Sequential(a)
    instantiate_chip(m, 0)
    with digital_mode(m):
        assert torch.allclose(a(x, ones), ref, atol=1e-5)
    for mat in analog_matrices(m):
        mat.use_dac_adc = False
    R = o3.rand_matrix()
    Din, Dout = tp.irreps_in1.D_from_matrix(R), tp.irreps_out.D_from_matrix(R)
    assert torch.allclose(a(x @ Din.T, ones), a(x, ones) @ Dout.T, atol=1e-4)
    assert torch.equal(a.output_mask, tp.output_mask)


@pytest.fixture(scope="module")
def c3n():
    from pcmsim.density import charge3net as c3
    import ase.io
    atoms = ase.io.read(ROOT / "data/inputs/mp-2095/mp-2095_Rb2Te3_chgcar.vasp", format="vasp")
    rng = np.random.default_rng(0)
    cart = rng.random((300, 3)) @ np.array(atoms.get_cell())
    return c3, atoms, cart


def test_charge3net_conversion_digital_equivalence(c3n):
    c3, atoms, cart = c3n
    torch.set_num_threads(8)
    model = c3.build_model()
    ref = c3.predict_points(model, atoms, cart)
    n_params = sum(p.numel() for p in model.parameters())
    conv = to_analog_charge3net(model, PCMConfig.from_yaml())
    kinds = {k for _, k in conv}
    assert kinds == {"fc_layer", "fctp_linear"}  # readout o3.Linear stays digital
    assert not any(n.endswith("readout") for n, _ in conv)
    # atom features stay digital (atom model, probe lin1; decision 2026-10-18)
    # radial-MLP ADCs use the calibration maximum, everything else the MSE full scale (decision 2026-10-18)
    fc = [mt.cfg.range_method for nm, mt in model.named_modules() if isinstance(mt, AnalogMatrix) and '.fc.layer' in nm]
    rest = [mt.cfg.range_method for nm, mt in model.named_modules() if isinstance(mt, AnalogMatrix) and '.fc.layer' not in nm]
    assert fc and set(fc) == {'max'} and set(rest) == {PCMConfig.from_yaml().range_method}
    assert all(n.startswith("probe_model") and not n.endswith("lin1") for n, _ in conv)
    instantiate_chip(model, 0)
    with digital_mode(model):
        assert np.allclose(c3.predict_points(model, atoms, cart), ref, atol=1e-5)
    assert 0.5 * n_params < n_analog_params(model) < 0.6 * n_params  # probe model without lin1 / readout: 54 %
    # analog inference runs and stays close to digital
    calibrate(model, lambda: c3.predict_points(model, atoms, cart[:100]))
    set_read_seed(model, 0)
    p = c3.predict_points(model, atoms, cart)
    assert np.isfinite(p).all() and c3.nmae(p, ref) < 0.5


def test_diffcsp_conversion_digital_equivalence():
    from pcmsim.diffcsp import load_mp20_csp
    model, _ = load_mp20_csp(str(ROOT / "data/weights/diffcsp_mp_csp/last.ckpt"))
    dec = model.decoder
    torch.manual_seed(0)
    nat = torch.tensor([5, 7])
    types = torch.randint(1, 60, (12,))
    n2g = torch.repeat_interleave(torch.arange(2), nat)
    x = torch.rand(12, 3)
    lat = torch.randn(2, 3, 3) + 4 * torch.eye(3)
    t_emb = model.time_embedding(torch.tensor([500, 500]))
    ref = dec(t_emb, types, x, lat, nat, n2g)
    conv = to_analog_diffcsp(model, PCMConfig.from_yaml())
    names = [n for n, _ in conv]
    assert len(conv) == 6 * 4 + 3, names  # 6 CSPLayers x (2 edge + 2 node) + atom_latent_emb + 2 heads
    assert all(k == "linear" for _, k in conv)
    instantiate_chip(model, 0)
    with digital_mode(model):
        out = dec(t_emb, types, x, lat, nat, n2g)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(out, ref))
    assert len(layer_table(model)) == 27


def test_noise_budget_sampler_reduces_to_original():
    """cal=None in digital mode reproduces CSPDiffusion.sample exactly (same RNG order)."""
    from pcmsim.diffcsp import load_mp20_csp
    from pcmsim.diffcsp.noise_budget import sample_noise_budget
    model, _ = load_mp20_csp(str(ROOT / "data/weights/diffcsp_mp_csp/last.ckpt"))
    model.beta_scheduler.timesteps = model.sigma_scheduler.timesteps = 1000
    nat = torch.tensor([2, 3])
    types = torch.tensor([32, 52, 51, 51, 52])
    n2g = torch.repeat_interleave(torch.arange(2), nat)
    to_analog_diffcsp(model, PCMConfig.from_yaml())
    instantiate_chip(model, 0)
    # same converted model, digital mode: isolates the sampler logic (2000 chaotic steps amplify any
    # fp32 difference between nn.Linear's fused addmm and x @ W^T + b)
    with digital_mode(model):
        torch.manual_seed(3)
        ref = model.sample(nat, types, n2g, step_lr=1e-5)
        torch.manual_seed(3)
        out = sample_noise_budget(model, nat, types, n2g, cal=None)
    assert torch.allclose(out["frac_coords"], ref["frac_coords"], atol=1e-5)
    assert torch.allclose(out["lattices"], ref["lattices"], atol=1e-4)
    assert out["analog_fraction"] == 1.0  # naive sampler: every evaluation is (nominally) analog


def test_fast_probe_graph_matches_charge3net(c3n):
    """Vectorised periodic KD-tree probe graph == charge3net's probes_to_graph (edges, offsets, output)."""
    c3, atoms, cart = c3n
    cart = cart[:300]
    graph = c3.atom_graph(atoms)
    ref = c3.probe_batch(atoms, cart, graph)
    fast = c3.probe_batch_fast(atoms, cart, graph, c3.PeriodicProbeGraph(atoms))
    key = lambda b: sorted(map(tuple, np.concatenate([b["probe_edges"][0].numpy(),
                                                      b["probe_edges_displacement"][0].numpy().round()], 1)))
    assert key(ref) == key(fast)
    model = c3.build_model()
    with torch.no_grad():
        rep = model.atom_model(ref)
        a, b = model.probe_model(ref, rep), model.probe_model(fast, rep)
    assert torch.allclose(a, b, atol=1e-6)
