# src/eval/TIFA/TIFA.py
import os, json, argparse
from tqdm import tqdm
from statistics import mean
from tifascore import (get_llama2_pipeline, get_llama2_question_and_answers,
                       VQAModel, tifa_score_single)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outpath", required=True,
                   help="目录应包含 prompt_images/ 与 sentences.txt")
    p.add_argument("--vqa", default="mplug-large",
                   help="VQA 模型，默认 mplug-large")
    args = p.parse_args()
    return args

def main():
    args = parse_args()
    outpath = args.outpath
    img_dir = os.path.join(outpath, "prompt_images")
    sent_file = os.path.join(outpath, "sentences.txt")

    # 读取文本
    with open(sent_file, "r") as f:
        prompts = [l.strip() for l in f]

    # 加载 QG & VQA
    qg_pipe = get_llama2_pipeline()          # 离线 LLaMA-2 问题生成
    vqa_model = VQAModel(args.vqa)

    results, tifa_scores = [], []

    for idx, prompt in enumerate(tqdm(prompts, desc="TIFA")):
        img_path = os.path.join(img_dir, f"{idx}.png")
        if not os.path.exists(img_path):
            print(f"[Warn] 缺少图片 {img_path}，跳过")
            continue

        # 1) 生成问题
        qas = get_llama2_question_and_answers(qg_pipe, prompt)
        if len(qas) == 0:
            print(f"[Warn] 无法为第 {idx} 条生成 Q&A，跳过")
            continue

        # 2) 计算单条 TIFA
        single_res = tifa_score_single(vqa_model, qas, img_path)
        score = single_res["tifa_score"]
        tifa_scores.append(score)

        results.append({
            "question_id": idx,
            "prompt": prompt,
            "image_name": f"{idx}.png",
            "answer": score,                # 与 PickScore 脚本保持字段命名
            "details": single_res["question_details"]  # 方便诊断，可随意去掉
        })

    # 保存
    save_dir = os.path.join(outpath, "annotation_tifa")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "vqa_result.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    avg = mean(tifa_scores) if tifa_scores else 0.0
    with open(os.path.join(save_dir, "score_avg.txt"), "w") as f:
        f.write(f"TIFA avg: {avg:.4f}\n")
    print(f"TIFA avg: {avg:.4f}")

if __name__ == "__main__":
    main()
