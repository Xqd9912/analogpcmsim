#!/bin/bash
# Submit the T2 two-stage SCF test: one job per structure (9 jobs, 1 DCU each), all runs of a
# structure serial inside its job.  Run on the cluster login node:
#   bash <base>/scripts/submit_all.sh            # all structures in runs/structures.txt
#   bash <base>/scripts/submit_all.sh mp-1224375 # only some
# Re-running is safe: finished runs are skipped by run_twostage_job.sh (FORCE=1 to redo).
set -eu
cd "$(dirname "$0")/.."
mkdir -p slurm_logs
MIDS=${@:-$(cat runs/structures.txt)}
for m in $MIDS; do
  [ -d runs/$m/default ] || { echo "no inputs for $m"; continue; }
  if squeue -u $USER -h -o "%j" | grep -qx "t2ts_$m"; then echo "t2ts_$m already queued/running -- skip"; continue; fi
  sbatch -J t2ts_$m scripts/run_twostage_job.sh $m
done
squeue -u $USER -o "%.10i %.20j %.3t %.10M %R" | grep -E "JOBID|t2ts_" || true
