export project_dir="src/eval/BLIPvqa_eval/"
cd $project_dir
data="SKCI"
out_dir="../../../examples/${data}"
source /etc/network_turbo
HF_HOME=/root/autodl-tmp/hub \
HF_ENDPOINT=https://hf-mirror.com \
python BLIP_vqa.py --out_dir=$out_dir
