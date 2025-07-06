export project_dir="src/eval/BLIPvqa_eval/"
cd $project_dir
data="SKCI"
out_dir="../../../examples/${data}"
HF_ENDPOINT=https://hf-mirror.com \ # Set the Hugging Face endpoint（可选）
python BLIP_vqa.py --out_dir=$out_dir
