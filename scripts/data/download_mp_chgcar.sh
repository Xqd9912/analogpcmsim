#!/bin/bash
# D1 reference densities: Materials Project CHGCARs (pymatgen Chgcar.as_dict, json.gz) from the public S3 bucket
# materialsproject-parsed/chgcars/<task_id>.json.gz -> data/chgcar/d1/<task_id>.json.gz  (100 files, ~2.8 GB).
# usage (repository root): bash scripts/data/download_mp_chgcar.sh [data/splits/chg3_test_random100_seed42.csv]
set -eu
LIST=${1:-data/splits/chg3_test_random100_seed42.csv}   # columns: rank,material_id,task_id,bytes
OUT=data/chgcar/d1; mkdir -p $OUT
tail -n +2 "$LIST" | cut -d, -f3 | while read -r tid; do
  f=$OUT/$tid.json.gz
  [ -s "$f" ] && continue
  curl -sS --retry 8 -C - -o "$f.part" "https://materialsproject-parsed.s3.amazonaws.com/chgcars/$tid.json.gz" \
    && gzip -t "$f.part" && mv "$f.part" "$f" && echo "$tid ok" || echo "$tid FAILED"
done
