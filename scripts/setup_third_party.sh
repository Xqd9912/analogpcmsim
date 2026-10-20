#!/bin/bash
# Fetch the upstream code and weights at the exact versions used in the paper (run from the repository root).
#   third_party/charge3net  AIforGreatGood/charge3net @ 18290c6 (includes models/charge3net_mp.pt)
#   third_party/DiffCSP     jiaor17/DiffCSP @ 7121d15 (only its MP-20 csv files are used; the model is ported in src/)
#   third_party/NeuroSim    neurosim/NeuroSim, tag 2DInferenceV1.4 (8a88abf), for P1 only
#   data/weights/diffcsp_mp_csp/{last.ckpt,hparams.yaml}  official DiffCSP MP-20 CSP checkpoint (Google Drive, gdown)
set -eu
mkdir -p third_party data/weights/diffcsp_mp_csp data/mp_20
clone() {  # url dir commit
  [ -d "$2/.git" ] || git clone "$1" "$2"
  git -C "$2" checkout -q "$3"
}
clone https://github.com/AIforGreatGood/charge3net third_party/charge3net 18290c63c0307792ab6da5ff98ee535cdb7376d6
clone https://github.com/jiaor17/DiffCSP third_party/DiffCSP 7121d159826efa2ba9500bf299250d96da37f146
[ -d third_party/NeuroSim/.git ] || git clone --depth 1 --branch 2DInferenceV1.4 https://github.com/neurosim/NeuroSim third_party/NeuroSim
cp third_party/DiffCSP/data/mp_20/{train,val,test}.csv data/mp_20/
# DiffCSP checkpoint (README of jiaor17/DiffCSP -> Google Drive folder 11WOc9lTZN4hkIY7SKLCIrbsTMGy9TsoW, mp_csp/)
if [ ! -s data/weights/diffcsp_mp_csp/last.ckpt ]; then
  pip install -q gdown
  gdown 1pIpTMmqt4vxvwrKM0KwmtwjJDp61ezmO -O data/weights/diffcsp_mp_csp/last.ckpt
  gdown 1X12-mwa3obpAYebYnQ93Nblcn_hDGXda -O data/weights/diffcsp_mp_csp/hparams.yaml
  # the two file ids are recorded without labels; swap if the "checkpoint" turned out to be the small yaml
  if [ $(wc -c < data/weights/diffcsp_mp_csp/last.ckpt) -lt 1000000 ]; then
    mv data/weights/diffcsp_mp_csp/last.ckpt data/weights/diffcsp_mp_csp/tmp
    mv data/weights/diffcsp_mp_csp/hparams.yaml data/weights/diffcsp_mp_csp/last.ckpt
    mv data/weights/diffcsp_mp_csp/tmp data/weights/diffcsp_mp_csp/hparams.yaml
  fi
fi
echo "expected sha256:"
echo "  c06719217c940718823f7c67b58c74a0fcfba4c3e4312b23cdd22b7a15ea6b5f  data/weights/diffcsp_mp_csp/last.ckpt"
echo "  cc2f5a8229472a5747b62966d98550b12ea396feee9fe463034455fdc7191bf8  third_party/charge3net/models/charge3net_mp.pt"
shasum -a 256 data/weights/diffcsp_mp_csp/last.ckpt third_party/charge3net/models/charge3net_mp.pt
