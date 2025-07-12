# model.py
import torch, torch.nn as nn
from transformers import AutoModel, AutoProcessor

class HalluDetector(nn.Module):
    def __init__(self, clip_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                 embed_dim=640, hidden=512, dropout=0.1):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(clip_name)
        self.clip = AutoModel.from_pretrained(clip_name)
        for p in self.clip.parameters():   # 全冻结
            p.requires_grad_(False)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)           # logits
        )

    @torch.no_grad()
    def encode(self, imgs, texts, device):
        inp_img = self.processor(images=imgs, return_tensors='pt').to(device)
        inp_txt = self.processor(text=texts, return_tensors='pt',
                                 padding=True, truncation=True).to(device)
        img_e = self.clip.get_image_features(**inp_img)
        txt_e = self.clip.get_text_features(**inp_txt)
        img_e = img_e / img_e.norm(dim=-1, keepdim=True)
        txt_e = txt_e / txt_e.norm(dim=-1, keepdim=True)
        return img_e, txt_e

    def forward(self, imgs, texts):
        device = next(self.parameters()).device
        with torch.no_grad():
            img_e, txt_e = self.encode(imgs, texts, device)
        feat = torch.cat([img_e, txt_e, (img_e - txt_e).abs()], dim=-1)
        return self.mlp(feat).squeeze(-1)   # [B]
