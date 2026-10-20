"""PCM device model from the measured LTP/LTD trace (data/device/1792073641587_R.csv).

Units: conductance in microsiemens (uS), time in seconds (s).
Every non-measured parameter comes from configs/device_nominal.yaml, where its source is recorded.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEVICE_CSV = ROOT / "data" / "device" / "1792073641587_R.csv"
NOMINAL_YAML = ROOT / "configs" / "device_nominal.yaml"
N_LTP = 40  # first 40 pulses LTP, last 40 LTD
TEN_YEARS_S = 10 * 365.25 * 24 * 3600.0


def load_trace(csv_path: str | Path = DEVICE_CSV) -> np.ndarray:
    """Measured 80-pulse conductance trace in uS (CSV column `conductance` is in S)."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 1] * 1e6


def pulse_noise(seg: np.ndarray) -> float:
    """Pulse-to-pulse fluctuation (uS) from second differences: var(d2 x) = 6 s^2 for white noise s."""
    return float(np.diff(seg, 2).std() / math.sqrt(6.0))


def select_states(g: np.ndarray, gap: float) -> np.ndarray:
    """Largest set of measured levels (uS) with adjacent spacing >= gap (uS), always keeping Gmin and Gmax."""
    v = np.sort(np.asarray(g, dtype=float))
    states = [v[0]]
    for x in v[1:]:
        if x - states[-1] >= gap:
            states.append(x)
    gmax = v[-1]
    if states[-1] != gmax:
        while len(states) > 1 and gmax - states[-1] < gap:
            states.pop()
        states.append(gmax)
    return np.array(states)


def trace_stats(g: np.ndarray, sigma_prog: float = 2.5) -> dict:
    """Summary statistics of the measured trace (all conductances in uS)."""
    ltp, ltd = g[:N_LTP], g[N_LTP:]
    out = {
        "g_min_uS": float(g.min()),
        "g_max_uS": float(g.max()),
        "on_off_ratio": float(g.max() / g.min()),
        "ltp_gain_first20_uS": float(ltp[19] - ltp[0]),
        "ltp_gain_last20_uS": float(ltp[39] - ltp[20]),
        "ltd_change_first20_uS": float(ltd[19] - ltd[0]),
        "ltd_change_last20_uS": float(ltd[39] - ltd[20]),
        "ltp_nonmonotonic_steps": int((np.diff(ltp) < 0).sum()),
        "ltd_nonmonotonic_steps": int((np.diff(ltd) > 0).sum()),
        "ltp_pulse_noise_uS": pulse_noise(ltp),
        "ltd_pulse_noise_uS": pulse_noise(ltd),
        "sigma_prog_nominal_uS": sigma_prog,
        "n_levels_35_50uS": int(((g >= 35) & (g < 50)).sum()),
        "n_levels_200_250uS": int(((g >= 200) & (g < 250)).sum()),
        "i_read_max_cell_uA_at_0p1V": float(0.1 * g.max()),
        "i_read_col256_worst_mA": float(256 * 0.1 * g.max() / 1000),
        "i_read_col64_worst_mA": float(64 * 0.1 * g.max() / 1000),
    }
    for k in (2, 3):
        s = select_states(g, 2 * k * sigma_prog)
        n_w = 2 * len(s) - 1
        out[f"states_{k}sigma"] = {
            "criterion": f"adjacent spacing >= 2*{k}*sigma_prog = {2 * k * sigma_prog:.1f} uS",
            "n_states": len(s),
            "states_uS": s.round(4).tolist(),
            "n_weight_levels_differential": n_w,
            "bits_equivalent": float(np.log2(n_w)),
        }
    return out


@dataclass
class PCMConfig:
    """Non-idealities (CLAUDE.md section 5). Conductances in uS, times in s."""

    states: list = field(default_factory=list)  # target state set S (uS), sorted, derived from the CSV
    sigma_prog: float = 2.5        # programming noise per device, static per chip instance (uS)
    read_rel: float = 0.01         # relative read noise std, resampled at every forward
    dac_bits: int = 8
    adc_bits: int = 8
    tile: int = 256                # crossbar rows/cols per tile
    alpha_grid: int = 91           # alpha_j candidates: max|w_j| * linspace(0.1, 1, alpha_grid)
    block_scale: bool = False      # option: block_alpha and block_dac together
    block_alpha: bool = False      # option: alpha_j per row block (tile) instead of per output column
    block_dac: bool = False        # option: DAC full scale per row block (tile) instead of per layer
    range_method: str = "mse"      # DAC/ADC full scale from calibration data: "mse" (min quant+clip MSE) | "max"
    dac_scaling: str = "dynamic"   # "dynamic": per-input-vector abs-max (nominal) | "static": calibrated full scale
    n_slices: int = 2              # 2 = dual-slice residual (write-verify) mapping (nominal since 2026-10-18)
    nu: float = 0.03               # drift exponent (drift disabled in current experiments)
    nu_std: float = 0.0
    t0: float = 20.0               # drift reference time (s)

    @property
    def g_min(self) -> float:
        return float(min(self.states))

    @property
    def g_max(self) -> float:
        return float(max(self.states))

    @classmethod
    def from_yaml(cls, path: str | Path = NOMINAL_YAML, **overrides) -> "PCMConfig":
        """Build from a YAML whose entries are {value, source}; states are derived from the CSV.

        The state set S is selected once with the *nominal* sigma_prog of the YAML and stays fixed;
        overrides (e.g. sigma_prog x0.5 / x2 for G4, or 0 to switch noise off) change only the noise.
        """
        raw = yaml.safe_load(Path(path).read_text())
        vals = {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in raw.items()}
        sel = vals.pop("state_selection")
        csv = vals.pop("device_csv")
        states = select_states(load_trace(ROOT / csv), 2 * sel["k_sigma"] * vals["sigma_prog"])
        vals.update(overrides)
        return cls(states=states.tolist(), **vals)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
