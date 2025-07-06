#!/bin/bash

data="SKCI"
outpath="examples/${data}"
HF_ENDPOINT=https://hf-mirror.com \
python src/eval/PickScore/PickScore.py --outpath="${outpath}"
