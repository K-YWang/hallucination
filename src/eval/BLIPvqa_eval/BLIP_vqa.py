import argparse
import os

import torch

from tqdm import tqdm, trange


import json
from tqdm.auto import tqdm
import sys
import spacy

from BLIP.train_vqa_func import VQA_main

def Create_annotation_for_BLIP(image_folder, outpath, np_index=None):
    nlp = spacy.load("en_core_web_sm")

    annotations = []
    # === 修改开始：按照 image_index.png 和 sentences.txt 匹配读取 ===
    sentence_file = os.path.join(os.path.dirname(image_folder), 'sentences.txt')
    with open(sentence_file, 'r') as f:
        sentences = [line.strip() for line in f.readlines()]

    for i, sentence in enumerate(sentences):
        file_name = f"{i}.png"
        image_path = os.path.join(image_folder, file_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        image_dict = {
            'image': image_path,
            'question_id': i,
            'dataset': "color"
        }

        doc = nlp(sentence)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks if chunk.text not in ['top', 'the side', 'the left', 'the right']]

        if np_index is not None and len(noun_phrases) > np_index:
            q_tmp = noun_phrases[np_index]
            image_dict['question'] = f'{q_tmp}?'
        else:
            image_dict['question'] = ''

        annotations.append(image_dict)


    print('Number of Processed Images:', len(annotations))

    json_file = json.dumps(annotations)
    with open(f'{outpath}/vqa_test.json', 'w') as f:
        f.write(json_file)

def parse_args():
    parser = argparse.ArgumentParser(description="BLIP vqa evaluation.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        required=True,
        help="Path to output BLIP vqa score",
    )
    parser.add_argument(
        "--np_num",
        type=int,
        default=8,
        help="Noun phrase number, can be greater or equal to the actual noun phrase number",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    np_index = args.np_num #how many noun phrases

    answer = []
    sample_num = len(os.listdir(os.path.join(args.out_dir,"prompt_images")))
    reward = torch.zeros((sample_num, np_index)).to(device='cuda')

    out_dir = args.out_dir

    order="_blip" #rename file
    for i in tqdm(range(np_index)):
        print(f"start VQA{i+1}/{np_index}!")
        os.makedirs(f"{out_dir}/blipvqa/annotation{i + 1}{order}", exist_ok=True)
        os.makedirs(f"{out_dir}/blipvqa/annotation{i + 1}{order}/VQA/", exist_ok=True)
        Create_annotation_for_BLIP(
            f"{out_dir}/prompt_images",
            f"{out_dir}/blipvqa/annotation{i + 1}{order}",
            np_index=i,
        )
        answer_tmp = VQA_main(f"{out_dir}/blipvqa/annotation{i + 1}{order}/",
                              f"{out_dir}/blipvqa/annotation{i + 1}{order}/VQA/")
        answer.append(answer_tmp)

        with open(f"{out_dir}/blipvqa/annotation{i + 1}{order}/VQA/result/vqa_result.json", "r") as file:
            r = json.load(file)
        with open(f"{out_dir}/blipvqa/annotation{i + 1}{order}/vqa_test.json", "r") as file:
            r_tmp = json.load(file)
        for k in range(len(r)):
            if(r_tmp[k]['question']!=''):
                reward[k][i] = float(r[k]["answer"])
            else:
                reward[k][i] = 1
        print(f"end VQA{i+1}/{np_index}!")
    reward_final = reward[:,0]
    for i in range(1,np_index):
        reward_final *= reward[:,i]

    #output final json
    with open(f"{out_dir}/blipvqa/annotation{i + 1}{order}/VQA/result/vqa_result.json", "r") as file:
        r = json.load(file)
    reward_after=0
    for k in range(len(r)):
        r[k]["answer"] = '{:.4f}'.format(reward_final[k].item())
        reward_after+=float(r[k]["answer"])
    os.makedirs(f"{out_dir}/blipvqa/annotation{order}", exist_ok=True)
    with open(f"{out_dir}/blipvqa/annotation{order}/vqa_result.json", "w") as file:
        json.dump(r, file)

    # calculate avg of BLIP-VQA as BLIP-VQA score
    print("BLIP-VQA score:", reward_after/len(r),'!\n')
    with open(f"{out_dir}/blipvqa/annotation{order}/blip_vqa_score.txt", "w") as file:
        file.write("BLIP-VQA score:"+str(reward_after/len(r)))



if __name__ == "__main__":
    main()
