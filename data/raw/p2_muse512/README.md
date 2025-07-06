---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
dataset_info:
  features:
  - name: Prompt
    dtype: string
  - name: Category
    dtype: string
  - name: Challenge
    dtype: string
  - name: Note
    dtype: string
  - name: images
    dtype: image
  - name: model_name
    dtype: string
  - name: seed
    dtype: int64
  splits:
  - name: train
    num_bytes: 128701081.568
    num_examples: 1632
  download_size: 127769152
  dataset_size: 128701081.568
---
# Dataset Card for "muse_512"

```py
```py
from PIL import Image  
import torch
from muse import PipelineMuse, MaskGiTUViT
from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value, load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = PipelineMuse.from_pretrained(
    transformer_path="valhalla/research-run",
    text_encoder_path="openMUSE/clip-vit-large-patch14-text-enc",
    vae_path="openMUSE/vqgan-f16-8192-laion",
).to(device)

pipe.transformer = MaskGiTUViT.from_pretrained("valhalla/research-run-finetuned-journeydb", revision="06bcd6ab6580a2ed3275ddfc17f463b8574457da", subfolder="ema_model").to(device)
pipe.tokenizer.pad_token_id = 49407

if device == "cuda":
    pipe.transformer.enable_xformers_memory_efficient_attention()
    pipe.text_encoder.to(torch.float16)
    pipe.transformer.to(torch.float16)


import PIL


def main():
    print("Loading dataset...")
    parti_prompts = load_dataset("nateraw/parti-prompts", split="train")

    print("Loading pipeline...")
    seed = 0

    device = "cuda"
    torch.manual_seed(0)

    ckpt_id = "openMUSE/muse-512"

    scale = 10

    print("Running inference...")
    main_dict = {}
    for i in range(len(parti_prompts)):
        sample = parti_prompts[i]
        prompt = sample["Prompt"]

        image = pipe(
            prompt,
            timesteps=16,
            negative_text=None,
            guidance_scale=scale,
            temperature=(2, 0),
            orig_size=(512, 512),
            crop_coords=(0, 0),
            aesthetic_score=6,
            use_fp16=device == "cuda",
            transformer_seq_len=1024,
            use_tqdm=False,
        )[0]

        image = image.resize((256, 256), resample=PIL.Image.Resampling.LANCZOS)
        img_path = f"/home/patrick/muse_images/muse_512_{i}.png"
        image.save(img_path)
        main_dict.update(
            {
                prompt: {
                    "img_path": img_path,
                    "Category": sample["Category"],
                    "Challenge": sample["Challenge"],
                    "Note": sample["Note"],
                    "model_name": ckpt_id,
                    "seed": seed,
                }
            }
        )

    def generation_fn():
        for prompt in main_dict:
            prompt_entry = main_dict[prompt]
            yield {
                "Prompt": prompt,
                "Category": prompt_entry["Category"],
                "Challenge": prompt_entry["Challenge"],
                "Note": prompt_entry["Note"],
                "images": {"path": prompt_entry["img_path"]},
                "model_name": prompt_entry["model_name"],
                "seed": prompt_entry["seed"],
            }

    print("Preparing HF dataset...")
    ds = Dataset.from_generator(
        generation_fn,
        features=Features(
            Prompt=Value("string"),
            Category=Value("string"),
            Challenge=Value("string"),
            Note=Value("string"),
            images=ImageFeature(),
            model_name=Value("string"),
            seed=Value("int64"),
        ),
    )
    ds_id = "diffusers-parti-prompts/muse512"
    ds.push_to_hub(ds_id)


if __name__ == "__main__":
    main()
```