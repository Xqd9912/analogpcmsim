#!/bin/bash
#SBATCH -p xahdnormal
##SBATCH -A <account>        # cluster-specific: edit the #SBATCH lines (partition, GPU/DCU gres) for your site
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=dcu:1
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/slurm-%x-%j.out
# D3: non-self-consistent (ICHARG=11) runs from ML-predicted densities.
# usage (from the NSCF work dir, layout runs/<tag>/<mid>/<model>/): sbatch -J d3_<mid> scripts/run_nscf_job.sh <mid> <tag>
# Inputs per run dir runs/<tag>/<mid>/<model>/: INCAR KPOINTS POSCAR POTCAR (copied from the two-stage stage-A
# of the same structure, LWAVE=.FALSE.) + CHGCAR (predicted grid + atomic augmentation).
set -u
MID=$1; TAG=$2
module purge
source "${VASP_ENV:?export VASP_ENV=<script that puts mpirun and vasp_std on PATH>}"
export OMP_NUM_THREADS=6 HSA_FORCE_FINE_GRAIN_PCIE=1
ulimit -s unlimited
for d in runs/$TAG/$MID/*/; do
  [ -f $d/OUTCAR ] && grep -q "General timing" $d/OUTCAR && { echo "skip $d"; continue; }
  ( cd $d && t0=$(date +%s) && mpirun -np 1 vasp_std > vasp.out 2>&1; echo $(( $(date +%s) - t0 )) > walltime_s.txt )
  rm -f $d/WAVECAR $d/CHG $d/PROCAR
  echo "done $d"
done
echo ALLDONE
