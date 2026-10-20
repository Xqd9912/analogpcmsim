"""S1 denoising trajectory on a PCM chip: snapshots of the noise-budget sampler for one composition (figure filmstrip).

env: pcm_models / pcm.  usage: python scripts/showcase/trajectory.py --comp Ge2Sb2Te5 --n 8 --chip 0 --device cuda:0
Same analog setup as generate.py (G3 nominal: dual slice, DAC/ADC calibrated on 32 MP-20 validation structures,
noise-budget sampler). Records (frac_coords, lattice) of every sample at the snapshot steps and the per-sub-step
analog/digital log.
Output: results/S1/trajectory/<comp>_chip<k>_seed<s>/{traj.npz, substeps.csv, meta.json}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Element

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from compositions import all_compositions  # noqa: E402
from pcmsim.diffcsp import load_mp20_csp  # noqa: E402

SNAP_T = (1000, 900, 800, 700, 600, 500, 400, 300, 200, 150, 100, 70, 50, 30, 20, 15, 10, 5, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="Ge2Sb2Te5")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--read_seed", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    from pcmsim.analog import calibrate, instantiate_chip, set_read_seed
    from pcmsim.convert import to_analog_diffcsp
    from pcmsim.device import PCMConfig
    from pcmsim.diffcsp.evaluate import batch_inputs, load_subset
    from pcmsim.diffcsp.noise_budget import calibrate_read_noise, sample_noise_budget

    model, _ = load_mp20_csp(str(ROOT / "data" / "weights" / "diffcsp_mp_csp" / "last.ckpt"))
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

    counts = all_compositions()[a.comp]["counts"]
    sp = [el for el, k in sorted(counts.items()) for _ in range(k)]
    nat = torch.tensor([len(sp)] * a.n, device=dev)
    types = torch.tensor([Element(e).Z for _ in range(a.n) for e in sp], device=dev)
    n2g = torch.repeat_interleave(torch.arange(a.n, device=dev), nat)
    snaps = {t: None for t in SNAP_T}
    log = []
    torch.manual_seed(a.seed)
    t0 = time.time()
    out = sample_noise_budget(model, nat, types, n2g, cal_nb, log=log, snapshots=snaps)
    wall = time.time() - t0

    od = ROOT / "results" / "S1" / "trajectory" / f"{a.comp}_chip{a.chip}_seed{a.seed}"
    od.mkdir(parents=True, exist_ok=True)
    ts = sorted(snaps, reverse=True)
    np.savez_compressed(od / "traj.npz", t=np.array(ts), species=np.array(sp),
                        frac=np.stack([snaps[t][0].numpy().reshape(a.n, len(sp), 3) for t in ts]),
                        lattice=np.stack([snaps[t][1].numpy() for t in ts]))
    pd.DataFrame(log).to_csv(od / "substeps.csv", index=False)
    json.dump({"comp": a.comp, "n": a.n, "seed": a.seed, "chip": a.chip, "read_seed": a.read_seed + a.chip,
               "device_config": "configs/device_nominal.yaml", "n_slices": 2, "sampler": "budget",
               "analog_fraction": float(out["analog_fraction"]), "wall_s": wall, "device": str(dev),
               "torch": torch.__version__}, open(od / "meta.json", "w"), indent=1)
    print(f"done in {wall / 60:.1f} min, analog fraction {out['analog_fraction']:.4f} -> {od}")


if __name__ == "__main__":
    main()
