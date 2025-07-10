#!/bin/bash
data="SKCI"
outpath="examples/${data}"
source /etc/network_turbo
HF_HOME=/root/autodl-tmp/hub \
HF_ENDPOINT=https://hf-mirror.com \
python src/eval/TIFA/TIFA.py --outpath="${outpath}"
