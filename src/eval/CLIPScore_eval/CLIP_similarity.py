# image and text similarity
# ref https://github.com/openai/CLIP
import os
import torch
import clip
from PIL import Image
import spacy
nlp=spacy.load('en_core_web_sm')

import json
import argparse

from clip.simple_tokenizer import SimpleTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
        required=True,
        help="Path to read samples and output scores"
    )
    parser.add_argument(
        "--complex",
        type=bool,
        default=False,
        help="To evaluate on samples in complex category or not"
    )
    args = parser.parse_args()
    return args





def main():
    args = parse_args()

    outpath=args.outpath

    image_folder=os.path.join(outpath,'prompt_images')
    # file_names = os.listdir(image_folder)
    # file_names.sort(key=lambda x: int(x.split("_")[-1].split('.')[0]))  # sort
    sentence_file = os.path.join(outpath, 'sentences.txt')

    # Read all sentences
    with open(sentence_file, 'r') as f:
        sentences = [line.strip() for line in f.readlines()]

    sim_dict = []
    total = []

    # output annotation.json
    for i, prompt in enumerate(sentences):
        image_path = os.path.join(image_folder, f"{i}.png")


        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)


        if (args.complex):
            doc=nlp(prompt)
            prompt_without_adj=' '.join([token.text for token in doc if token.pos_ != 'ADJ']) #remove adj
            text = clip.tokenize(prompt_without_adj, truncate=True).to(device)
        else:
            text = clip.tokenize(prompt, truncate=True).to(device)







        with torch.no_grad():
            image_features = model.encode_image(image.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)


            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

             # Calculate the cosine similarity between the image and text features
            cosine_similarity = (image_features @ text_features.T).squeeze().item()

        similarity = cosine_similarity
        total.append(similarity)

        sim_dict.append({
            'question_id': i,
            'answer': similarity,
            'prompt': prompt,
            'image_name': f"{i}.png"
        })

        if (i + 1) % 100 == 0:
            print(f"CLIP image-text:{i + 1} prompt(s) have been processed!")

    # Save results
    savepath = os.path.join(outpath, "annotation_clip")
    os.makedirs(savepath, exist_ok=True)

    with open(os.path.join(savepath, 'vqa_result.json'), 'w') as f:
        json.dump(sim_dict, f, indent=2)

    avg_score = sum(total) / len(total) if total else 0.0
    with open(os.path.join(savepath, 'score_avg.txt'), 'w') as f:
        f.write('score avg: ' + str(avg_score))

    print(f"Saved results to {savepath}")
    print("score avg:", avg_score)

    # json_file = json.dumps(sim_dict)
    # savepath = os.path.join(outpath,"annotation_clip") #todo
    # os.makedirs(savepath, exist_ok=True)
    # with open(f'{savepath}/vqa_result.json', 'w') as f:
    #     f.write(json_file)
    # print(f"save to {savepath}")
    
    # # score avg
    # score=0
    # for i in range(len(sim_dict)):
    #     score+=float(sim_dict[i]['answer'])
    # with open(f'{savepath}/score_avg.txt', 'w') as f:
    #     f.write('score avg:'+str(score/len(sim_dict)))
    # print("score avg:", score/len(sim_dict))

if __name__ == "__main__":
    main()






