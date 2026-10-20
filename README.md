# PCM analog in-memory computing for materials AI (in-silico study)

Code for an EDTM conference paper that evaluates two pretrained materials models on simulated
phase-change-memory (PCM) crossbar arrays whose weight levels come from a measured PCM device:

- **DiffCSP**: diffusion crystal-structure generation (MP-20, CSP mode). About 2000 network
  evaluations per sample reuse the same programmed weights.
- **ChargE3Net**: GNN charge-density prediction. 10⁵–10⁷ grid points per structure reuse the same
  weights. The predicted density goes into a non-self-consistent VASP calculation.

**Everything here is simulation.** No hardware was built or measured. All energy and latency numbers
are **projected** with NeuroSim (V1.4, 22 nm) and compared with energy measured on a GPU (A800).

## Main results (nominal configuration, 3 independently programmed chip instances)

| | digital | PCM arrays (simulated) |
|---|---|---|
| DiffCSP match rate, 1000 MP-20 test structures | 58.0 % | 57.6 ± 0.8 % (noise-budget sampler; 99.2–99.8 % of evaluations on the arrays) |
| ChargE3Net density NMAE, 100 test structures | 0.52 % | 0.96 % (prediction averaged over 3 arrays) |
| Projected energy relative to the A800 GPU | 1 | DiffCSP ≈ 1/98; ChargE3Net ≈ 1/1.9 (Amdahl limit: 48.5 % of its MACs are on the arrays) |

## Simulation model in brief

- **Device**: 80 measured conductance values (one LTP/LTD cycle, `data/device/1792073641587_R.csv`, in S; 35.3–234.4 µS).
  From these we select 15 distinguishable states (adjacent spacing ≥ 4σ_prog), which give 29 levels
  after differential mapping. Programming noise σ_prog = 2.5 µS is drawn once per chip (`chip_seed`).
  Read noise is 1 % of G (assumed) and is redrawn on every forward pass. Drift is not modelled.
- **Mapping**: each weight uses a differential pair, G⁺/G⁻, with one scale per output column chosen
  by an MSE grid search. The nominal mapping uses two slices: the residual left after programming
  slice 1 is written to a second pair. Tiles are 256×256, and partial sums are added in fp32.
  DAC and ADC are 8 bit, with per-vector absolute-max input scaling. The ADC full scale is set on
  calibration data.
- **Noise budget (DiffCSP)**: σ_a(t), the read-noise variance at the network output, is calibrated
  per step. The sampler injects only `sqrt(std² − s²)` and falls back to digital wherever s² > std².
- **ChargE3Net split**: the radial MLPs and the sc/lin2/lin3 linear layers run on the arrays. Tensor
  products, gates, the readout and all atom-feature operations stay digital. Before the NSCF run, the
  predicted densities are projected onto the plane-wave sphere |G| ≤ 2√(ENCUT/(ħ²/2m)).

Implementation: `src/pcmsim/{device,mapping,analog,convert}.py`. Model wrappers:
`src/pcmsim/diffcsp/` (pure-torch port of CSPNet and the noise-budget sampler) and
`src/pcmsim/density/` (ChargE3Net wrapper and CHGCAR I/O). NeuroSim driver: `src/pcmsim/neurosim.py`.

## Layout

```
src/pcmsim/          device model, mapping, analog layers, model conversion, DiffCSP port, ChargE3Net wrapper, NeuroSim driver
configs/             one YAML per experiment (device_nominal.yaml = nominal device)
scripts/             experiment entry points (see table below); showcase/ = GeTe–Sb2Te3 generation chain; vasp/ = cluster jobs
figures/             conference figure notebooks (read results/ only) + style.py
data/device/         measured PCM conductance (S)
data/splits/         fixed evaluation subsets
data/structures/     ChargE3Net calibration CIFs, GST-series structures for D3
tests/               pytest (device statistics, equivariance, model conversion)
```

## Setup

```bash
conda env create -f environment.yml && conda activate pcm   # installs this package in editable mode
bash scripts/setup_third_party.sh       # charge3net, DiffCSP (MP-20 csv), NeuroSim, DiffCSP checkpoint
bash scripts/data/download_mp_chgcar.sh # D1 reference densities (MP CHGCARs, ~2.8 GB)
pytest tests/
```

The upstream repositories are used unmodified. `src/pcmsim/density/charge3net.py` applies a one-line
`one_hot(...).float()` fix at runtime.

## Reproducing the figures

Experiment IDs: T = baseline, G = generation, D = density, S = showcase, P = power. GPU runs used a
single A800 / RTX PRO 6000. Use `--chips k` to run chip instances one at a time.

| Figure panel | Data | Commands |
|---|---|---|
| Fig. 3a projected energy | `results/P1_neurosim/`, `results/P1_gpu/` | `python scripts/run_p1_neurosim.py --config configs/P1_neurosim.yaml`; `python scripts/run_p1_gpu.py --config configs/P1_gpu.yaml --device cuda:0` (idle GPU); `python scripts/run_p1_compare.py` |
| Fig. 3b match rate / RMSE vs read noise | `results/{T3_diffcsp_digital,G1,G3,G4}/` | `python scripts/run_diffcsp.py --config configs/<cfg>.yaml --device cuda:0 --chips 0` for `T3_diffcsp_digital_full`, `G1_dual_full` (naive, 1 %), `G3_dual_full` (budget, 1 %), `G4_naive_read{3,5}pct`, `G4_read{3,5}pct` (and `G4_sigma{0p5x,2x}` for the σ_prog sensitivity) |
| Fig. 3c noise budget + trajectory | `results/G3/eval1000_budget_dual_chip0/budget_curves.csv`, `results/S1/trajectory/` | `python scripts/make_figure_data.py budget_dual`; `python scripts/showcase/trajectory.py --comp Ge2Sb2Te5 --n 8 --chip 0 --device cuda:0` |
| Fig. 3d GeTe–Sb₂Te₃ mixing energies | `results/showcase_test/`, `results/S1/gallery_tail/` | see *S1 chain* below |
| Fig. 4a D1 NMAE | `results/D1/probes300k_dual_am_c012*` | `python scripts/run_charge3net.py --config configs/D1_dual_probes300k.yaml --device cuda:0 --tag probes300k_dual_am_c012 --chips 0 1 2 --shard {0,1} --nshards 2`; `python scripts/d1_hw_floor.py --run probes300k_dual_am_c012` |
| Fig. 4b–c D3 density | `results/D3/density/`, `results/D3/final/` | per structure: `python scripts/run_d3_density.py --ids <mid> --chips 0 1 2 --device cuda:0 --tag gpu_dual_am_<mid> --device_overrides '{"n_slices":2}' --write_chgcar`; then `python scripts/run_d3_chipmean.py --ids <mids...> --src_tag gpu_dual_am`; `python scripts/make_figure_data.py d3_final` |
| Fig. 4d DOS | `results/D3/nscf/` | VASP, see below; `python scripts/vasp/d3_nscf/parse_d3.py results/D3/nscf gpu_dual_am_mean3_lp results/D3/nscf/d3/gpu_dual_am_mean3_lp.csv` (same for `gpu_dual_am_lp`); `python scripts/vasp/d3_nscf/extract_dos.py results/D3/nscf gpu_dual_am_lp gpu_dual_am_mean3_lp results/D3/nscf/dos` |
| page | `figures/out/conf_page/` | run `figures/conf_fig{1,3,4}_*.ipynb`, then `figures/conf_page.ipynb` |

D3 structures (`data/structures/d3_gst/`): mp-32 Ge, mp-104 Sb, mp-19 Te, mp-938 GeTe, mp-1201 Sb₂Te₃,
mp-29641 GeSb₄Te₇, mp-1224350 GeSb₂Te₄, mp-1224375 Ge₂Sb₂Te₅, mp-1212511 Ge₃Sb₂Te₆.
The DFT reference densities are expected at `data/chgcar/gst/<mid>_{default,atomaug}.CHGCAR`.
They are copied from the `default` and `atomaug_prep` runs of the VASP step below.

### S1 chain (generation → MACE relaxation → analysis)

```bash
cd scripts/showcase
python generate.py --mode digital --n 8 --seed 0 --device cuda:0
python generate.py --mode analog --chip {0,1,2} --digital_tail 100 --n 8 --seed 0 --device cuda:0   # budget sampler, last 100 steps digital
python fetch_refs.py                                  # MP references (needs MP_API_KEY), env with mp-api
python relax.py --cifs <gen dir> --out <relax jsonl>  # env: pcm-mace (MACE-MP-0)
python analyze.py --gen analog_chip0_tail100_seed0    # mixing energies vs GeTe + Sb2Te3
python gallery.py --analog_stems ... --out gallery_tail
```

Further compositions use `--only <names> --seed <k>`. Ge₂Sb₂Te₅, GeSb₄Te₇, GeSb₂Te₄ and other GST
phases are in the MP-20 training set, so their generated structures are a qualitative showcase only,
not predictions.

### VASP (cluster, Slurm)

Settings: pymatgen `MPStaticSet`, PBE_54 POTCARs (MP symbol choice), ENCUT 520 eV, ISPIN = 1.
VASP ignores a CHGCAR that has no PAW augmentation occupancies. Each predicted CHGCAR therefore
carries the atomic occupancies taken from an ICHARG = 12, NELM = 1 pre-run.

1. **SCF reference + augmentation pre-run**: set up a work dir `<base>` containing `scripts/` (copy
   `scripts/vasp/scf_reference/*`) and `structures/` (copy `data/structures/d3_gst/*`). Then run
   `python scripts/make_twostage_inputs.py` and `bash scripts/submit_all.sh`. The job script needs
   `VASP_ENV`, `POTDIR` and (optionally) `PY`. Edit the `#SBATCH` lines for your site.
2. **NSCF from predicted densities**: in a second work dir, put the predicted CHGCARs into
   `incoming/<tag>/<mid>_<model>.CHGCAR`. Then run `SCF_BASE=<base> bash scripts/setup.sh <tag> <mid> <models...>`
   followed by `sbatch scripts/run_nscf_job.sh <mid> <tag>` (scripts from `scripts/vasp/d3_nscf/`).
3. Copy the `vasprun.xml` files back to `results/D3/nscf/{d3/<tag>,t2}/...` and run the parsers.

`scripts/vasp/s1_chain/make_inputs.py` writes the same inputs for generated (S1) structures.

## Upstream code and data

- DiffCSP: Jiao et al., NeurIPS 2023, https://github.com/jiaor17/DiffCSP (MP-20 data and checkpoint)
- ChargE3Net: Koker et al., npj Comput. Mater. 2024, https://github.com/AIforGreatGood/charge3net
- NeuroSim: https://github.com/neurosim/NeuroSim (2DInferenceV1.4)
- MACE-MP-0: Batatia et al., https://github.com/ACEsuit/mace
- Materials Project charge densities: Shen et al., Sci. Data 2022 (public S3 bucket `materialsproject-parsed`)

## License / citation

To be added.
