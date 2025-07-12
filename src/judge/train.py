# train.py
import argparse, json, random, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from PIL import Image
from model import HalluDetector

class JsonlSet(Dataset):
    def __init__(self, path):
        self.items = [json.loads(l) for l in open(path)]
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

def collate(batch):
    imgs   = [Image.open(b['img']).convert('RGB') for b in batch]
    prompts= [b['prompt'] for b in batch]
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.float32)
    return imgs, prompts, labels

def run_epoch(model, loader, opt=None):
    logits_all, y_all = [], []
    for imgs, txts, y in tqdm(loader, leave=False):
        logits = model(imgs, txts)
        if opt:                               # 训练
            loss = F.binary_cross_entropy_with_logits(logits, y.to(logits.device))
            opt.zero_grad(); loss.backward(); opt.step()
        logits_all.append(logits.detach().cpu())
        y_all.append(y)
    logits_all = torch.cat(logits_all); y_all = torch.cat(y_all)
    probs = torch.sigmoid(logits_all)
    preds = (probs > 0.5).float()
    return {
        'acc': accuracy_score(y_all, preds),
        'f1':  f1_score(y_all, preds),
        'auroc': roc_auc_score(y_all, probs)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True, help='data/evalmuse.jsonl')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr',    type=float, default=1e-4)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    # split 80/10/10
    data = JsonlSet(args.jsonl)
    random.shuffle(data.items)
    n = len(data); n_train = int(0.8*n); n_val = int(0.9*n)
    train_set = torch.utils.data.Subset(data, range(0, n_train))
    val_set   = torch.utils.data.Subset(data, range(n_train, n_val))
    test_set  = torch.utils.data.Subset(data, range(n_val, n))

    model = HalluDetector().to(args.device)
    opt   = torch.optim.AdamW(model.mlp.parameters(), lr=args.lr)

    for ep in range(1, args.epochs+1):
        print(f'-- Epoch {ep} --')
        train_stats = run_epoch(model, DataLoader(train_set, shuffle=True,
                             batch_size=args.batch, collate_fn=collate), opt)
        print(' Train', train_stats)
        val_stats   = run_epoch(model, DataLoader(val_set, batch_size=args.batch,
                             collate_fn=collate))
        print('  Val ', val_stats)

    print('\n=== Final Test ===')
    test_stats = run_epoch(model, DataLoader(test_set, batch_size=args.batch,
                         collate_fn=collate))
    print(' Test', test_stats)

if __name__ == '__main__':
    main()
