export project_dir="src/eval/BLIPvqa_eval/"
cd $project_dir
data="p2"
out_dir="../../../examples/${data}"
python BLIP_vqa.py --out_dir=$out_dir
