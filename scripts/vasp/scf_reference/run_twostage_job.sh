#!/bin/bash
#SBATCH -p xahdnormal
##SBATCH -A <account>        # cluster-specific: edit the #SBATCH lines (partition, GPU/DCU gres) for your site
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=dcu:1
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/slurm-%x-%j.out
# T2 two-stage SCF test for ONE structure; all runs of a structure run serially in this job (same
# node/DCU -> comparable wall times).
#   default        ICHARG=2 single stage (reference; converged CHGCAR + NGXF grid)
#   atomaug_prep   ICHARG=12, NELM=1 (atomic PAW augmentation occupancies)
#   <init>_<stop>  stage A (fixed density, ICHARG=11/12, LWAVE=T) -> stage B (ISTART=1, ICHARG=0, SCF)
#                  init: atomic | oracle_atomaug | oracle_trueaug [| oracle_noaug: sanity, mp-1224375]
#                  stop: full | loose | fixed8
# usage (from the work dir <base> = {scripts/,runs/,structures/}):  sbatch -J t2ts_<mid> scripts/run_twostage_job.sh <mid> [runs...]
# A run (or A+B pair) whose OUTCAR already ends normally is skipped (safe to resubmit after a timeout);
# set FORCE=1 to redo.
set -u
MID=$1; shift
BASE=${T2TS_BASE:-$PWD}   # work dir (sbatch from it); T2TS_DRYRUN=1: login-node dry test only
RUN=$BASE/runs/$MID
SCRIPTS=$BASE/scripts
PY=${PY:-python3}      # needs numpy only
POTDIR=${POTDIR:?export POTDIR=<PBE_54 POTCAR library, one <symbol>/POTCAR per element>}
FORCE=${FORCE:-0}
if [ $# -gt 0 ]; then RUNS="$@"; else
  RUNS="default atomaug_prep"
  for i in atomic oracle_atomaug oracle_trueaug; do for s in full loose fixed8; do RUNS="$RUNS ${i}_${s}"; done; done
  [ -d $RUN/oracle_noaug_full ] && RUNS="$RUNS oracle_noaug_full"
fi

module purge
source "${VASP_ENV:?export VASP_ENV=<script that puts mpirun and vasp_std on PATH>}"
ulimit -s unlimited
export OMP_NUM_THREADS=6
export HSA_FORCE_FINE_GRAIN_PCIE=1
[ "${T2TS_DRYRUN:-0}" = 1 ] && source $SCRIPTS/dryrun_stub.sh   # fake mpirun (no VASP) for flow testing
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}  runs: $RUNS"
echo "$(hostname) ${SLURM_JOB_ID:-none}" > $RUN/node.txt

# POTCAR from POTCAR.spec; abort if a library file's TITEL does not match its directory name
mkpot() { : > POTCAR; for s in $(cat POTCAR.spec); do
  t=$(head -1 $POTDIR/$s/POTCAR | awk '{print $2}'); [ "$t" = "$s" ] || { echo "POTCAR $s has TITEL $t -- abort"; exit 2; }
  cat $POTDIR/$s/POTCAR >> POTCAR; done; }
finished() { [ -s "$1/OUTCAR" ] && grep -q "General timing and accounting" "$1/OUTCAR"; }
runvasp() {   # in cwd; wall time of the vasp process -> walltime_s.txt
  rm -f OSZICAR OUTCAR vasp.out walltime_s.txt
  mkpot
  local t0=$(date +%s.%N)
  mpirun -np 1 vasp_std > vasp.out 2>&1
  awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.2f\n", b-a}' > walltime_s.txt
  echo "  [$(basename $(dirname $PWD))/$(basename $PWD)] wall=$(cat walltime_s.txt)s steps=$(grep -cE '^(DAV|RMM|CG|DMP|SDA):' OSZICAR) last='$(tail -1 OSZICAR)'"
  grep -iE "incomplete|WAVECAR|chargedensity" vasp.out | head -5 | sed 's/^/    vasp.out: /'
}

# ---- init CHGCARs for ICHARG=11 (built once from default + atomaug_prep) ----
build_init() {   # $1 = oracle_atomaug | oracle_trueaug | oracle_noaug  -> $RUN/init_chgcar/$1.CHGCAR
  local d=$RUN/init_chgcar; mkdir -p $d
  [ -s $d/$1.CHGCAR ] && return 0
  [ -s $RUN/default/CHGCAR ] || { echo "no default CHGCAR"; return 1; }
  case $1 in
    oracle_trueaug) cp $RUN/default/CHGCAR $d/$1.CHGCAR ;;
    oracle_atomaug|oracle_noaug)
      [ $1 = oracle_atomaug ] && { [ -s $RUN/atomaug_prep/CHGCAR ] || { echo "no atomaug_prep CHGCAR"; return 1; }; }
      $PY $SCRIPTS/build_init_chgcar.py $RUN/default/CHGCAR $RUN/default/POSCAR $d/$1.CHGCAR $d/$1.info.json \
          $([ $1 = oracle_atomaug ] && echo $RUN/atomaug_prep/CHGCAR) || return 1 ;;
  esac
}

for R in $RUNS; do
  case $R in
    default|atomaug_prep)
      if [ $FORCE = 0 ] && finished $RUN/$R && [ -s $RUN/$R/CHGCAR ]; then echo "skip $R"; continue; fi
      cd $RUN/$R || exit 1
      rm -f CHGCAR WAVECAR CHG
      runvasp
      if [ $R = default ]; then
        $PY -c "import sys,json; sys.path.insert(0,'$SCRIPTS'); import chgcar_writer as cw; g=cw.parse_fine_grid_from_outcar('OUTCAR'); g['NELECT']=cw.parse_nelect_from_outcar('OUTCAR'); json.dump(g,open('grid.json','w')); print('  grid', g)"
        finished . && [ -s CHGCAR ] || { echo "default failed -- abort $MID"; exit 3; }
      fi
      ;;
    *_full|*_loose|*_fixed8)
      [ -d $RUN/$R/A ] || { echo "no inputs for $R"; continue; }
      if [ $FORCE = 0 ] && finished $RUN/$R/B; then echo "skip $R"; continue; fi
      INIT=${R%_*}
      # ---- stage A: fixed density, writes WAVECAR ----
      cd $RUN/$R/A || exit 1
      rm -f CHGCAR WAVECAR CHG
      if [ $INIT != atomic ]; then
        build_init $INIT || { echo "$R: init CHGCAR build failed"; continue; }
        cp $RUN/init_chgcar/$INIT.CHGCAR CHGCAR
        grep -q "^ *ICHARG = 11" INCAR || { echo "$R/A: INCAR not ICHARG=11"; continue; }
      fi
      runvasp
      [ -s WAVECAR ] || { echo "$R/A: no WAVECAR -- skip B"; continue; }
      # ---- stage B: ISTART=1, ICHARG=0 (density rebuilt from stage-A orbitals), normal SCF ----
      cd $RUN/$R/B || exit 1
      rm -f CHGCAR WAVECAR CHG
      mv $RUN/$R/A/WAVECAR WAVECAR
      runvasp
      rm -f WAVECAR CHGCAR CHG $RUN/$R/A/CHGCAR $RUN/$R/A/CHG
      ;;
    *) echo "unknown run $R";;
  esac
done
# keep default/CHGCAR (converged reference) and atomaug_prep/CHGCAR (atomic aug. source for ML inits)
rm -f $RUN/*/WAVECAR $RUN/*/*/WAVECAR $RUN/*/CHG $RUN/*/*/CHG $RUN/*/PROCAR $RUN/*/*/PROCAR
rm -f $RUN/init_chgcar/*.CHGCAR
echo ALLDONE
