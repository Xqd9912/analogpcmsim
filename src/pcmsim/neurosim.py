"""NeuroSim (DNN+NeuroSim V1.4, C++ hardware estimator) interface for P1 energy estimates.

Upstream: third_party/NeuroSim (github.com/neurosim/NeuroSim, branch 2DInferenceV1.4) is not modified. `build()`
copies its NeuroSIM/ sources into a build directory, sets the user-defined design options in Param.cpp
(each override carries its source in the P1 config) and compiles. Only the C++ estimator is used; accuracy comes
from our own device model (pcmsim.analog), so NeuroSim's Python wrapper is not used.

Each crossbar matrix (AnalogMatrix: one nn.Linear, radial-MLP layer, or one e3nn irrep path) becomes one NeuroSim
layer, written as a 1x1 convolution over an "image" of n_vec x 1 pixels, n_vec = number of input vectors per
network evaluation. Traces follow NeuroSim's wrapper format (utee/hook.py): weight file (in, out) with values in
[-1, 1); input file (in, n_vec * numBitInput) of bits, column = vector * numBitInput + bit, MSB (sign) first,
two's complement (dec2bin). Inputs are normalised per vector by their abs-max before quantisation (our DAC model,
dac_scaling = dynamic); weights are scaled per output column by max|w_j| (per-column alpha_j).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .analog import AnalogMatrix

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/NeuroSim/Inference_pytorch/NeuroSIM"
# Apple clang has no OpenMP (the estimator does not use it) and rejects upstream's chained comparisons
# ("a <= b <= c" in CurrentSenseAmp.cpp) unless -Wparentheses is relaxed; semantics stay as with GCC.
CXXFLAGS = "-O3 -std=c++11 -w -Wno-error=parentheses -Wno-parentheses"


# ------------------------------------------------------------------------------------------- build
def patch_param(src: str, overrides: dict[str, float | int | bool]) -> str:
    """Replace the first `name = ...;` statement at line start (user-defined section of Param.cpp)."""
    for name, val in overrides.items():
        v = ("true" if val else "false") if isinstance(val, bool) else repr(val)
        pat = re.compile(rf"^(\s*){re.escape(name)}\s*=(?!=)[^;]*;", re.M)
        src, n = pat.subn(lambda m: f"{m.group(1)}{name} = {v};  // pcmsim P1 override", src, count=1)
        if n != 1:
            raise KeyError(f"Param.cpp: no assignment for {name}")
    return src


def build(param: dict[str, float | int | bool], build_dir: Path, jobs: int = 8) -> Path:
    """Copy the upstream estimator, apply Param.cpp overrides, compile; returns the `main` binary."""
    build_dir = Path(build_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    for f in list(UPSTREAM.glob("*.cpp")) + list(UPSTREAM.glob("*.h")) + [UPSTREAM / "makefile"]:
        shutil.copy2(f, build_dir / f.name)
    p = build_dir / "Param.cpp"
    p.write_text(patch_param(p.read_text(), param))
    r = subprocess.run(["make", f"-j{jobs}", f"CXXFLAGS={CXXFLAGS}"], cwd=build_dir, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    return build_dir / "main"


# ------------------------------------------------------------------------------------------ traces
def matrix_names(model: torch.nn.Module) -> dict[int, str]:
    return {id(m): n for n, m in model.named_modules() if isinstance(m, AnalogMatrix)}


class TraceRecorder:
    """Forward pre-hooks on every AnalogMatrix: input vectors (n, in) of each call, in first-call order."""

    def __init__(self, model: torch.nn.Module, include=lambda name: True):
        self.names = matrix_names(model)
        self.calls: dict[str, list[torch.Tensor]] = {}
        self.mats: dict[str, AnalogMatrix] = {}
        self._h = []
        for m in model.modules():
            if isinstance(m, AnalogMatrix) and include(self.names[id(m)]):
                self._h.append(m.register_forward_pre_hook(self._hook))

    def _hook(self, m, args):
        name = self.names[id(m)]
        self.mats.setdefault(name, m)
        self.calls.setdefault(name, []).append(args[0].detach().reshape(-1, m.in_features).float().cpu())

    def remove(self):
        for h in self._h:
            h.remove()

    def layers(self, seed: int = 0) -> list[dict]:
        """One entry per matrix (each matrix is called once per network evaluation): weight (out, in), n_vec =
        mean input vectors per call, and n_vec input vectors drawn uniformly from all recorded calls (pooled over
        evaluations, e.g. several diffusion time steps)."""
        rng = np.random.default_rng(seed)
        out = []
        for name, xs in self.calls.items():
            x = torch.cat(xs).numpy()
            n_vec = max(1, int(round(len(x) / len(xs))))
            idx = rng.choice(len(x), size=n_vec, replace=False)
            out.append({"name": name, "weight": self.mats[name].weight.detach().cpu().numpy(),
                        "x": x[idx], "n_vec": n_vec})
        return out


def input_bits(x: np.ndarray, n_bits: int) -> np.ndarray:
    """(n_vec, in) real inputs -> (in, n_vec * n_bits) uint8 bits; per-vector abs-max normalisation, two's
    complement with sign bit first (NeuroSim dec2bin), column = vector * n_bits + bit."""
    sc = np.abs(x).max(axis=1, keepdims=True)
    sc[sc == 0] = 1.0
    q = np.clip(np.round(x / sc * 2 ** (n_bits - 1)), -2 ** (n_bits - 1), 2 ** (n_bits - 1) - 1).astype(np.int64)
    u = np.mod(q, 2 ** n_bits)  # two's complement
    bits = (u[..., None] >> np.arange(n_bits - 1, -1, -1)) & 1  # (n_vec, in, n_bits), MSB first
    return bits.transpose(1, 0, 2).reshape(x.shape[1], -1).astype(np.uint8)


def scaled_weight(w: np.ndarray, synapse_bit: int) -> np.ndarray:
    """(out, in) -> (in, out) in [-1, 1 - 2^-(synapseBit-1)], per output column scaled by max|w_j|."""
    a = np.abs(w).max(axis=1, keepdims=True)
    a[a == 0] = 1.0
    return np.clip(w / a, -1.0, 1.0 - 2.0 ** -(synapse_bit - 1)).T


def toggle_rate(bits: np.ndarray, n_bits: int) -> float:
    """Fraction of bit flips between consecutive serial input bits on each row (NeuroSim `inputtoggle`)."""
    return float((bits[:, 1:] != bits[:, :-1]).mean())


def _write_bits(path: Path, bits: np.ndarray) -> None:
    lut = np.array([b"0", b"1"])
    with open(path, "wb") as f:
        for row in bits:
            f.write(b",".join(lut[row]) + b"\n")


def write_traces(layers: list[dict], out_dir: Path, synapse_bit: int, n_bits: int) -> tuple[Path, list[Path], float]:
    """NetWork.csv + per-layer weight/input files; returns (netcsv, [w1, x1, w2, x2, ...], mean input toggle)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    net, files, tog, wsum = [], [], 0.0, 0
    for i, L in enumerate(layers):
        out_f, in_f = L["weight"].shape
        net.append(f"{L['n_vec']},1,{in_f},1,1,{out_f},0,1")
        wf, xf = out_dir / f"weight{i}.csv", out_dir / f"input{i}.csv"
        np.savetxt(wf, scaled_weight(L["weight"], synapse_bit), delimiter=",", fmt="%.6f")
        b = input_bits(L["x"], n_bits)
        _write_bits(xf, b)
        files += [wf, xf]
        n = b.size
        tog, wsum = tog + toggle_rate(b, n_bits) * n, wsum + n
    netcsv = out_dir / "NetWork.csv"
    netcsv.write_text("\n".join(net) + "\n")
    return netcsv, files, tog / wsum


# --------------------------------------------------------------------------------------------- run
def run(binary: Path, netcsv: Path, files: list[Path], synapse_bit: int, n_bits: int, rows: int,
        rows_parallel: int, log: Path | None = None) -> str:
    cmd = [str(binary), str(netcsv), str(synapse_bit), str(n_bits), str(rows), str(rows_parallel)] + [str(f) for f in files]
    r = subprocess.run(cmd, cwd=Path(binary).parent, capture_output=True, text=True)
    if log is not None:
        Path(log).write_text(r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr else ""))
    if r.returncode != 0 or "Summary" not in r.stdout:
        raise RuntimeError(f"NeuroSim failed ({r.returncode}): {r.stdout[-1500:]} {r.stderr[-1500:]}")
    return r.stdout


_NUM = r"([-+0-9.eE]+|nan|inf)"
_LAYER = {"read_latency_ns": r"readLatency is: {}ns", "read_dynamic_pJ": r"readDynamicEnergy is: {}pJ",
          "leakage_power_uW": r"leakagePower is: {}uW", "leakage_pJ": r"leakageEnergy is: {}pJ",
          "buffer_latency_ns": r"buffer latency is: {}ns", "buffer_dynamic_pJ": r"buffer readDynamicEnergy is: {}pJ",
          "ic_latency_ns": r"ic latency is: {}ns", "ic_dynamic_pJ": r"ic readDynamicEnergy is: {}pJ"}
_BREAK = {"ADC": "ADC", "accum": "Accumulation Circuits", "other": "Other Peripheries"}
_CHIP = {"chip_area_um2": r"ChipArea : {}um\^2", "array_area_um2": r"Chip total CIM array : {}um\^2",
         "ic_area_um2": r"Total IC Area on chip \(Global and Tile/PE local\): {}um\^2",
         "adc_area_um2": r"Total ADC \(or S/As and precharger for SRAM\) Area on chip : {}um\^2",
         "accum_area_um2": r"Total Accumulation Circuits .*? on chip : {}um\^2",
         "other_area_um2": r"Other Peripheries \(e\.g\. decoders, mux, switchmatrix, buffers, pooling and activation units\) : {}um\^2",
         "clk_period_ns": r"Chip clock period is: {}ns",
         "latency_ns": r"Chip (?:pipeline-system-clock-cycle|layer-by-layer readLatency) \(per image\) is: {}ns",
         "read_dynamic_pJ": r"Chip (?:pipeline-system|total) readDynamicEnergy(?: \(per image\))? is: {}pJ",
         "leakage_pJ": r"Chip (?:pipeline-system|total) leakage Energy(?: \(per image\))? is: {}pJ",
         "leakage_power_uW": r"Chip (?:pipeline-system|total) leakage Power(?: \(per image\))? is: {}uW",
         "buffer_dynamic_pJ": r"Chip (?:pipeline-system )?buffer readDynamicEnergy(?: \(per image\))? is: {}pJ",
         "ic_dynamic_pJ": r"Chip (?:pipeline-system )?ic readDynamicEnergy(?: \(per image\))? is: {}pJ",
         "tops_per_w": r"Energy Efficiency TOPS/W \((?:Pipelined|Layer-by-Layer) Process\): {}",
         "throughput_tops": r"Throughput TOPS \((?:Pipelined|Layer-by-Layer) Process\): {}",
         "memory_utilization_pct": r"Memory Utilization of Whole Chip: {} %"}


def parse(stdout: str) -> dict:
    """Chip summary, chip-level ADC/accumulation/other breakdown and per-layer numbers from NeuroSim stdout."""
    f = lambda pat, s: (lambda m: float(m.group(1)) if m else None)(re.search(pat.replace("{}", _NUM), s))
    summ = stdout[stdout.index("Summary"):]
    chip = {k: f(p, stdout if k == "memory_utilization_pct" else summ) for k, p in _CHIP.items()}
    for k, lab in _BREAK.items():
        chip[f"{k}_dynamic_pJ"] = f(rf"-{{11}} {re.escape(lab)}.*?readDynamicEnergy is : {{}}pJ", summ)
        chip[f"{k}_latency_ns"] = f(rf"-{{11}} {re.escape(lab)}.*?readLatency is : {{}}ns", summ)
    layers = []
    for blk in re.split(r"-+ Estimation of Layer \d+ -+", stdout[:stdout.index("Summary")])[1:]:
        d = {k: f(r"layer\d+'s " + p, blk) for k, p in _LAYER.items()}
        for k, lab in _BREAK.items():
            d[f"{k}_dynamic_pJ"] = f(rf"-{{11}} {re.escape(lab)}.*?readDynamicEnergy is : {{}}pJ", blk)
        layers.append(d)
    tiles = re.search(r"# of tile used for each layer -+\n(.*?)\n\n", stdout, re.S)
    if tiles:
        for d, line in zip(layers, tiles.group(1).splitlines()):
            d["tiles"] = float(line.split(":")[1])
    return {"chip": chip, "layers": layers}
