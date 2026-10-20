"""Generate CSP samples for the showcase compositions with the ported DiffCSP, digital or on PCM arrays (S1).

env: pcm_models / pcm.  usage: python generate.py --n 8 --seed 0 [--mode analog --chip 0 --device cuda:0]
Analog mode = G3 nominal: dual-slice mapping (configs/device_nominal.yaml), DAC/ADC calibrated on 32 MP-20 validation
structures (50 strided steps), noise-budget sampler (read noise counted in the injected noise; digital where s^2 > std^2).
Output: results/showcase_test/gen/{digital|analog_chip<k>}_seed{seed}/{name}__{k}.cif + samples.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Element, Lattice, Structure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from compositions import all_compositions  # noqa: E402
from pcmsim.diffcsp import load_mp20_csp, lattices_to_params_shape  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="samples per composition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--mode", choices=["digital", "analog"], default="digital")
    ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--read_seed", type=int, default=1000)
    ap.add_argument("--digital_tail", type=int, default=0, help="analog: last N timesteps digital")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    comps = all_compositions()
    if a.only:
        comps = {k: v for k, v in comps.items() if k in a.only}
    dev = torch.device(a.device)
    model, _ = load_mp20_csp(str(ROOT / "data" / "weights" / "diffcsp_mp_csp" / "last.ckpt"))
    model.to(dev)
    sampler, extra = None, {}
    if a.mode == "analog":
        from pcmsim.analog import calibrate, instantiate_chip, set_read_seed
        from pcmsim.convert import to_analog_diffcsp
        from pcmsim.device import PCMConfig
        from pcmsim.diffcsp.evaluate import batch_inputs, load_subset
        from pcmsim.diffcsp.noise_budget import calibrate_read_noise, sample_noise_budget
        pcm = PCMConfig.from_yaml(ROOT / "configs" / "device_nominal.yaml", n_slices=2)
        to_analog_diffcsp(model, pcm)
        instantiate_chip(model, a.chip)
        model.to(dev)
        cal = load_subset("data/splits/mp20_val_val1000_seed42.csv", "data/mp_20/val.csv", 32)
        nat_c, types_c, n2g_c = batch_inputs([s for _, s in cal], dev)

        def calib_pass():
            torch.manual_seed(12345)
            model.sample(nat_c, types_c, n2g_c, step_lr=1e-5, timesteps=50)

        calibrate(model, calib_pass)
        set_read_seed(model, a.read_seed + a.chip, dev)
        cal_nb = calibrate_read_noise(model, nat_c, types_c, n2g_c, n_t=50, n_rep=16)
        sampler = lambda m, na, ty, ng: sample_noise_budget(m, na, ty, ng, cal_nb, digital_tail=a.digital_tail)

    names, species = [], []
    for name, c in comps.items():
        sp = [el for el, k in sorted(c["counts"].items()) for _ in range(k)]
        for k in range(a.n):
            names.append(f"{name}__{k}")
            species.append(sp)
    nat = torch.tensor([len(s) for s in species], device=dev)
    types = torch.tensor([Element(e).Z for s in species for e in s], device=dev)
    node2graph = torch.repeat_interleave(torch.arange(len(species), device=dev), nat)
    print(f"{len(species)} samples, {int(nat.sum())} atoms total", flush=True)

    torch.manual_seed(a.seed)
    t0 = time.time()
    out = sampler(model, nat, types, node2graph) if sampler else model.sample(nat, types, node2graph, step_lr=1e-5)
    wall = time.time() - t0
    if out.get("analog_fraction") is not None:
        extra["analog_fraction"] = float(out["analog_fraction"])
    lengths, angles = lattices_to_params_shape(out["lattices"].cpu())
    fx = out["frac_coords"].cpu().split(nat.tolist())
    tag = "digital" if a.mode == "digital" else f"analog_chip{a.chip}" + (f"_tail{a.digital_tail}" if a.digital_tail else "")
    od = ROOT / "results" / "showcase_test" / "gen" / f"{tag}_seed{a.seed}"
    od.mkdir(parents=True, exist_ok=True)
    rec = []
    for i, (nm, sp) in enumerate(zip(names, species)):
        r = {"sample": nm, "composition": nm.split("__")[0], "natoms": len(sp)}
        try:
            s = Structure(Lattice.from_parameters(*lengths[i].tolist(), *angles[i].tolist()), sp,
                          fx[i].numpy(), coords_are_cartesian=False)
            dm = s.distance_matrix + np.eye(len(s)) * 1e3
            r.update(valid_lattice=True, vol_per_atom=s.volume / len(s), min_dist=float(dm.min()))
            s.to(filename=str(od / f"{nm}.cif"))
        except Exception as e:  # degenerate lattice
            r.update(valid_lattice=False, error=str(e))
        rec.append(r)
    json.dump({"seed": a.seed, "n_per_comp": a.n, "mode": a.mode, "chip": a.chip if a.mode == "analog" else None,
               "wall_s": wall, **extra, "samples": rec},
              open(od / "samples.json", "w"), indent=1)
    print(f"done in {wall / 60:.1f} min -> {od}")


if __name__ == "__main__":
    main()
