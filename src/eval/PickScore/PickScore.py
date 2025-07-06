# src/eval/PickScore/PickScore_similarity.py
import os
import torch
from PIL import Image
import json
import argparse
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PickScore model and processor
processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)

def pickscore(prompt: str, image_path: str) -> float:
    img = Image.open(image_path).convert("RGB")

    image_inputs = processor(images=[img],
                              return_tensors="pt",
                              truncation=True,
                              max_length=77,
                              padding=True).to(device)

    text_inputs  = processor(text=[prompt],
                              return_tensors="pt",
                              truncation=True,
                              max_length=77,
                              padding=True).to(device)

    with torch.no_grad():
        img_emb = model.get_image_features(**image_inputs)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        txt_emb = model.get_text_features(**text_inputs)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        score = ((txt_emb @ img_emb.T))[0, 0].item()

    return score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outpath", type=str, required=True, help="Path to output folder with prompt_images/ and sentences.txt")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    outpath = args.outpath

    image_folder = os.path.join(outpath, "prompt_images")
    sentence_file = os.path.join(outpath, "sentences.txt")

    # Read prompts
    with open(sentence_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]

    results = []
    scores = []

    for i, prompt in enumerate(tqdm(prompts, desc="Scoring")):
        image_path = os.path.join(image_folder, f"{i}.png")
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        score = pickscore(prompt, image_path)
        scores.append(score)
        results.append({
            "question_id": i,
            "prompt": prompt,
            "image_name": f"{i}.png",
            "answer": score
        })

    savepath = os.path.join(outpath, "annotation_pickscore")
    os.makedirs(savepath, exist_ok=True)

    with open(os.path.join(savepath, "vqa_result.json"), "w") as f:
        json.dump(results, f, indent=2)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    with open(os.path.join(savepath, "score_avg.txt"), "w") as f:
        f.write(f"PickScore avg: {avg_score:.4f}\n")
    print("PickScore avg:", avg_score)

if __name__ == "__main__":
    main()