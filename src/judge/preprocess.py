# preprocess.py
"""
{"img": "000123.png", "prompt": "...", "label": 1}
"""
import json, argparse, pathlib, random, tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='原始标注文件 (.json/.tsv)')
    ap.add_argument('--dst', default='data.jsonl')
    ap.add_argument('--img_dir', default='.')
    args = ap.parse_args()

    dst = open(args.dst, 'w', encoding='utf-8')
    for line in tqdm.tqdm(open(args.src, encoding='utf-8')):
        parts = line.strip().split('\t')
        img, prompt, major_off = parts[0], parts[1], int(parts[4])
        label = 1 if major_off else 0           # 1 = 有幻觉
        obj = {'img': f'{args.img_dir}/{img}', 'prompt': prompt, 'label': label}
        dst.write(json.dumps(obj, ensure_ascii=False) + '\n')
    dst.close()

if __name__ == '__main__':
    main()