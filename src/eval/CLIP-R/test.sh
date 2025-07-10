#!/bin/bash
# src/eval/CLIP_R/test.sh
data="SKCM"
outpath="examples/${data}"

source /etc/network_turbo
HF_HOME=/root/autodl-tmp/hub \
HF_ENDPOINT=https://hf-mirror.com \
python src/eval/CLIP-R/CLIP_R.py --outpath="${outpath}" --num_neg 99 --seed 42
