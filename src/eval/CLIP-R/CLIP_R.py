# src/eval/CLIP_R/CLIP_R_similarity.py
"""
实现 CLIP-R / R-Precision（Recall@1）
对每张生成图像，从其 prompt 及随机抽取的 N 条干扰 prompt
中检索，看正确 prompt 是否排第 1 名。
"""
import os, json, argparse, random
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ViT-H-14 与 PickScore 同规格；如需更轻量可改成 ViT-B/32
processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
model = AutoModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").eval().to(device)

@torch.no_grad()
def _get_embs(img: Image.Image, texts: list[str]):
    """一次性提特征，避免多余前后处理开销"""
    image_inputs = processor(images=[img], return_tensors="pt").to(device)
    text_inputs  = processor(text=texts,  return_tensors="pt", padding=True, truncation=True).to(device)

    img_emb  = model.get_image_features(**image_inputs)
    txt_embs = model.get_text_features(**text_inputs)

    img_emb  = img_emb  / img_emb.norm(dim=-1, keepdim=True)
    txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)
    return img_emb, txt_embs

def clip_r_rank(prompt_idx: int, prompts: list[str], img_path: str,
                num_neg: int = 99, rng: random.Random | None = None) -> int:
    """返回正例在排序中的名次（1 = 正确 prompt 排首位）"""
    if rng is None:
        rng = random
    # 采样负例索引
    all_indices = list(range(len(prompts)))
    all_indices.remove(prompt_idx)
    neg_indices = rng.sample(all_indices, k=min(num_neg, len(all_indices)))
    cand_texts = [prompts[prompt_idx]] + [prompts[i] for i in neg_indices]

    img = Image.open(img_path).convert("RGB")
    img_emb, txt_embs = _get_embs(img, cand_texts)

    sims = (img_emb @ txt_embs.T).squeeze(0)  # [num_cands]
    # argsort(desc) 返回的是小→大，这里取反
    rank = (sims.argsort(descending=True) == 0).nonzero(as_tuple=False).item() + 1
    return rank  # 1 = top-1 命中

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outpath", type=str, required=True,
                   help="目录下需包含 prompt_images/ 与 sentences.txt")
    p.add_argument("--num_neg", type=int, default=99,
                   help="每张图采用的负例 prompt 数量（默认 99）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args   = parse_args()
    rng    = random.Random(args.seed)

    image_dir = os.path.join(args.outpath, "prompt_images")
    sent_file = os.path.join(args.outpath, "sentences.txt")

    with open(sent_file, "r") as f:
        prompts = [l.strip() for l in f]

    results, hits = [], 0
    for i, prompt in enumerate(tqdm(prompts, desc="Scoring (CLIP-R)")):
        img_path = os.path.join(image_dir, f"{i}.png")
        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_path}")
            continue

        rank = clip_r_rank(i, prompts, img_path, args.num_neg, rng)
        hits += int(rank == 1)
        results.append({
            "question_id": i,
            "prompt": prompt,
            "image_name": f"{i}.png",
            "rank": rank,           # 名次
            "answer": int(rank == 1)  # 与 PickScore 格式保持 answer 字段
        })

    save_dir = os.path.join(args.outpath, "annotation_clip_r")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "vqa_result.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    clip_r = hits / len(results) if results else 0.0
    with open(os.path.join(save_dir, "score_avg.txt"), "w") as f:
        f.write(f"CLIP-R (Recall@1): {clip_r:.4f}\n")
    print("CLIP-R Recall@1 =", round(clip_r, 4))

if __name__ == "__main__":
    main()
