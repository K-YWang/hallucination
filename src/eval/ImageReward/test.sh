#!/bin/bash
# src/eval/ImageReward/test.sh
data="SKCM"
outpath="examples/${data}"
source /etc/network_turbo

HF_HOME=/root/autodl-tmp/hub \
HF_ENDPOINT=https://hf-mirror.com \
python src/eval/ImageReward/ImageReward_similarity.py --outpath="${outpath}"
