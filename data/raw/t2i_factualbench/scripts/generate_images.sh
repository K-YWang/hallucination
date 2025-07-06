#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

torchrun \
  --nproc_per_node=1 \
  --master_port=29501 \
  src/generate_all.py \
  --model sdxl \
  --prompt_dir data/prompts/text_injection \
  --output_dir results/ \
  --level MKCC \
  --knowledge_injection text
