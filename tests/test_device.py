"""T1 unit tests for the device model and analog layers (CLAUDE.md section 9, tests 1-5)."""
import math

import numpy as np
import pytest
import torch
from torch import nn

from pcmsim import mapping
from pcmsim.analog import (AnalogLinear, AnalogO3Linear, analog_matrices, calibrate, convert_to_analog,
                           digital_mode, instantiate_chip, set_drift_time, set_read_seed)
from pcmsim.device import TEN_YEARS_S, PCMConfig, load_trace, trace_stats

o3 = pytest.importorskip("e3nn.o3")


def cfg(**kw):
    """Device-level tests check the statistics of one differential pair (single slice) unless asked otherwise;
    the nominal dual-slice mapping is covered by test_dual_slice_reduces_weight_error."""
    kw.setdefault("n_slices", 1)
    return PCMConfig.from_yaml(**kw)


def gaussian_linear(out=192, inp=300, seed=0, bias=True):
    torch.manual_seed(seed)
    lin = nn.Linear(inp, out, bias=bias)
    nn.init.normal_(lin.weight, std=0.05)
    return lin


# ---------------------------------------------------------------- device data
def test_trace_stats_and_states():
    s = trace_stats(load_trace())
    assert s["g_min_uS"] == pytest.approx(35.33, abs=0.01) and s["g_max_uS"] == pytest.approx(234.44, abs=0.01)
    assert s["ltp_nonmonotonic_steps"] == 3 and s["ltd_nonmonotonic_steps"] == 4
    assert s["ltp_pulse_noise_uS"] == pytest.approx(1.87, abs=0.01)
    assert s["ltd_pulse_noise_uS"] == pytest.approx(2.80, abs=0.01)
    c = cfg()
    assert len(c.states) == 15 and c.g_min == pytest.approx(35.33, abs=0.01) and c.g_max == pytest.approx(234.44, abs=0.01)
    assert np.diff(c.states).min() >= 4 * c.sigma_prog - 1e-9


def test_digital_mode_is_exact():
    lin = gaussian_linear()
    model = nn.Sequential(lin)
    x = torch.randn(7, lin.in_features)
    ref = lin(x)
    convert_to_analog(model, cfg())
    instantiate_chip(model, 0)
    with digital_mode(model):
        assert torch.allclose(model(x), ref, atol=1e-6)


# ------------------------------------------------- 1: quantization vs alpha
def test_1_quantization_error_vs_alpha():
    c = cfg(sigma_prog=0.0, read_rel=0.0)
    lin = gaussian_linear()
    a = AnalogLinear(lin, c)
    S = a.mat.states
    w = lin.weight.detach()
    alpha, ratios, mse = mapping.search_alpha(w, S, return_curve=True)
    i = int(mse.mean(dim=1).argmin())
    assert 0 < i < len(ratios) - 1  # bell-shaped weights: optimum clips the tails
    assert torch.allclose(alpha, ratios[mse.argmin(0)] * w.abs().amax(1))
    model = nn.Sequential(a)
    instantiate_chip(model, 0)
    a.mat.use_dac_adc = False
    x = torch.randn(5, lin.in_features)
    wq = mapping.quantized_weight(w, a.mat.alpha[0], S)
    assert torch.allclose(model(x), x @ wq.T + lin.bias, atol=1e-4)
    gp, gn = a.mat.g_pos, a.mat.g_neg
    assert torch.isin(gp, S).all() and torch.isin(gn, S).all()
    assert ((gp == S[0]) | (gn == S[0])).all()  # one device of every pair at Gmin
    assert torch.equal(lin.weight.detach(), a.mat.weight.detach())  # master weights untouched


# --------------------------------------------- 2: programming noise statistics
def test_2_programming_noise_std():
    c = cfg(read_rel=0.0)
    lin = gaussian_linear(out=256, inp=512)
    a = AnalogLinear(lin, c)
    model = nn.Sequential(a)
    instantiate_chip(model, 123)
    w = lin.weight.detach()
    gp_t, gn_t = mapping.target_conductance(w, a.mat.alpha[0], a.mat.states)
    dp, dn = (a.mat.g_pos - gp_t).flatten(), (a.mat.g_neg - gn_t).flatten()
    assert dp.std().item() == pytest.approx(c.sigma_prog, rel=0.02)
    assert dn.std().item() == pytest.approx(c.sigma_prog, rel=0.02)
    assert abs(torch.corrcoef(torch.stack([dp, dn]))[0, 1]) < 0.02
    err_g = ((a.mat.effective_weight() - mapping.quantized_weight(w, a.mat.alpha[0], a.mat.states))
             * a.mat.dg / a.mat.alpha[0].unsqueeze(1)).flatten()
    assert err_g.std().item() == pytest.approx(math.sqrt(2) * c.sigma_prog, rel=0.02)
    g1 = a.mat.g_pos.clone()
    instantiate_chip(model, 123)
    assert torch.equal(g1, a.mat.g_pos)  # same chip_seed -> same chip
    instantiate_chip(model, 124)
    assert not torch.equal(g1, a.mat.g_pos)


# ---------------------------------------------- 3: read noise per forward
def test_3_read_noise_resampled_each_forward():
    c = cfg()
    lin = gaussian_linear()
    model = nn.Sequential(AnalogLinear(lin, c))
    instantiate_chip(model, 7)
    set_read_seed(model, 0)
    mat = analog_matrices(model)[0]
    mat.use_dac_adc = False
    g_before = mat.g_pos.clone()
    x = torch.randn(4, lin.in_features)
    ys = torch.stack([model(x) for _ in range(400)])
    assert not torch.allclose(ys[0], ys[1])
    assert torch.equal(g_before, mat.g_pos)
    pred = c.read_rel * torch.sqrt((x * x) @ (mat.g_pos ** 2 + mat.g_neg ** 2).T) * mat.alpha[0] / mat.dg
    assert (ys.std(0) / pred).mean().item() == pytest.approx(1.0, abs=0.03)
    # dedicated stream: same read_seed reproduces, and the global RNG is not consumed
    set_read_seed(model, 0)
    torch.manual_seed(5)
    y_a = model(x)
    r_a = torch.rand(1)
    set_read_seed(model, 0)
    torch.manual_seed(5)
    assert torch.equal(model(x), y_a) and torch.equal(torch.rand(1), r_a)


# --------------------------------------------- 4: drift + global compensation
def test_4_drift_compensation():
    c = cfg(read_rel=0.0)
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(64, 128, bias=False), nn.SiLU(), nn.Linear(128, 64, bias=False),
                          nn.SiLU(), nn.Linear(64, 16, bias=False))  # biases are digital and do not drift
    x_cal, x = torch.randn(256, 64), torch.randn(64, 64)
    convert_to_analog(model, c)
    instantiate_chip(model, 0)
    run = lambda: model(x_cal)
    calibrate(model, run)
    y0 = model(x)
    set_drift_time(model, TEN_YEARS_S, compensate=False)
    y_nc = model(x)
    set_drift_time(model, TEN_YEARS_S, run_fn=run)
    y_c = model(x)
    scale = lambda y: (y.abs().mean() / y0.abs().mean()).item()
    assert scale(y_nc) < 0.5
    assert scale(y_c) == pytest.approx(1.0, abs=0.03)
    set_drift_time(model, c.t0)
    assert torch.allclose(model(x), y0, atol=1e-5)


# ------------------------------- 5: o3.Linear equivariant with programming noise only
def test_5_o3_linear_equivariance_programming_noise_only():
    torch.manual_seed(1)
    lin = o3.Linear("8x0e+4x1o+3x2e+2x1e", "6x0e+5x1o+2x2e+3x0o", biases=True)
    with torch.no_grad():
        lin.bias.normal_()
    x = torch.randn(11, lin.irreps_in.dim)
    ref = lin(x)
    model = nn.Sequential(AnalogO3Linear(lin, cfg(read_rel=0.0)))
    instantiate_chip(model, 0)
    with digital_mode(model):
        assert torch.allclose(model(x), ref, atol=1e-5)
    shapes = sorted(tuple(m.weight.shape) for m in analog_matrices(model))
    assert shapes == sorted([(6, 8), (5, 4), (2, 3)])  # one crossbar per irrep path (mul_out, mul_in)
    for m in analog_matrices(model):
        m.use_dac_adc = False
    R = o3.rand_matrix()
    Din, Dout = lin.irreps_in.D_from_matrix(R), lin.irreps_out.D_from_matrix(R)
    y = model(x)
    assert not torch.allclose(y, ref, atol=1e-3)  # programming noise is really on
    assert torch.allclose(model(x @ Din.T), y @ Dout.T, atol=1e-4)


# --------------------------------------------- dual-slice residual mapping (decision point 3)
def test_dual_slice_reduces_weight_error():
    from pcmsim.analog import count_macs
    lin = gaussian_linear(out=256, inp=300)
    w = lin.weight.detach()
    err = {}
    for ns in (1, 2):
        for sp in (0.0, 2.5):
            a = AnalogLinear(lin, cfg(n_slices=ns, sigma_prog=sp, read_rel=0.0))
            m = nn.Sequential(a)
            instantiate_chip(m, 0)
            err[(ns, sp)] = float(((a.mat.effective_weight() - w).norm() / w.norm()))
            if ns == 2 and sp == 2.5:
                a.mat.use_dac_adc = False
                x = torch.randn(6, lin.in_features)
                y = m(x)
                assert torch.allclose(y, x @ a.mat.effective_weight().T + lin.bias, atol=1e-4)
                with count_macs(m) as c:
                    m(x)
                assert c["total"] == 2 * 6 * lin.in_features * lin.out_features
    assert err[(2, 0.0)] < 0.2 * err[(1, 0.0)]   # quantization error largely removed
    assert err[(2, 2.5)] < 0.3 * err[(1, 2.5)]   # slice-1 programming error compensated
