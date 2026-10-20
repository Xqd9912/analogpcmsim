#!/bin/bash
# usage: bash scripts/setup.sh <tag> <mid> <model> ...   (CHGCARs already uploaded to incoming/<tag>/<mid>_<model>.CHGCAR)
set -eu
TAG=$1; MID=$2; shift 2
SRC=${SCF_BASE:?export SCF_BASE=<scf_reference work dir>}/runs/$MID/oracle_atomaug_full/A
for M in "$@"; do
  d=runs/$TAG/$MID/$M; mkdir -p $d
  cp $SRC/KPOINTS $SRC/POSCAR $SRC/POTCAR $d/
  sed 's/^LWAVE = True/LWAVE = False/' $SRC/INCAR > $d/INCAR
  cp incoming/$TAG/${MID}_${M}.CHGCAR $d/CHGCAR
  grep -q "^ICHARG = 11" $d/INCAR || { echo "INCAR not ICHARG=11"; exit 1; }
done
ls runs/$TAG/$MID
