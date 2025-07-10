# src/eval/ImageReward/ImageReward_similarity.py
import os
import json
import argparse
from tqdm import tqdm
import torch
from PIL import Image
import ImageReward as RM       # pip install image-reward

# 设备自动检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 ImageReward v1.0 模型（≈1.8 GB，第一次会自动缓存到 ~/.cache/ImageReward）
model = RM.load("ImageReward-v1.0", device=device)

@torch.no_grad()
def imagereward_score(prompt: str, image_path: str) -> float:
    """
    计算 (prompt, image) 的 ImageReward 分数
    """
    return model.score(prompt, image_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath",
        type=str,
        required=True,
        help="包含 prompt_images/ 与 sentences.txt 的输出目录",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    outpath = args.outpath

    image_folder = os.path.join(outpath, "prompt_images")
    sentence_file = os.path.join(outpath, "sentences.txt")

    # 读取全部 prompt
    with open(sentence_file, "r") as f:
        prompts = [line.strip() for line in f]

    results, scores = [], []

    for i, prompt in enumerate(tqdm(prompts, desc="Scoring")):
        image_path = os.path.join(image_folder, f"{i}.png")
        if not os.path.exists(image_path):
            print(f"[WARN] Image not found: {image_path}")
            continue

        score = imagereward_score(prompt, image_path)
        scores.append(score)
        results.append(
            {
                "question_id": i,
                "prompt": prompt,
                "image_name": f"{i}.png",
                "answer": score,
            }
        )

    savepath = os.path.join(outpath, "annotation_imagereward")
    os.makedirs(savepath, exist_ok=True)

    # 保存逐样本结果
    with open(os.path.join(savepath, "vqa_result.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存平均分
    avg_score = sum(scores) / len(scores) if scores else 0.0
    with open(os.path.join(savepath, "score_avg.txt"), "w") as f:
        f.write(f"ImageReward avg: {avg_score:.4f}\n")

    print("ImageReward avg:", avg_score)

if __name__ == "__main__":
    main()
