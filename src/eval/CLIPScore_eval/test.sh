data="SKCI"
outpath="examples/${data}"
source /etc/network_turbo
HF_HOME=/root/autodl-tmp/hub \
HF_ENDPOINT=https://hf-mirror.com \
python src/eval/CLIPScore_eval/CLIP_similarity.py --outpath="${outpath}"
