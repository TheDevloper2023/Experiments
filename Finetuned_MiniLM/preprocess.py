"""
Preprocesses everything for the model
"""
import csv
import torch
from transformers import AutoTokenizer, AutoModel
import hparam as hp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(hp.MODEL)
model = AutoModel.from_pretrained(hp.MODEL).to(device).eval()

def mean_pool(last_hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

texts, labels = [], []
with open('gotrainadd.csv', newline='') as f:
    for row in csv.DictReader(f):
        texts.append(row['text'])
        labels.append([float(row[e]) / 10.0 for e in hp.EMOTIONS])   # 0..10 → 0..1


embeds = []
with torch.no_grad():
    for i in range(0, len(texts), hp.BATCH_SIZE):
        enc = tokenizer(
            texts[i:i+hp.BATCH_SIZE],
            padding=True, truncation=True, max_length=hp.MAX_LEN,
            return_tensors='pt'
        ).to(device)
        out = model(**enc)
        embeds.append(mean_pool(out.last_hidden_state, enc['attention_mask']).cpu())

X = torch.cat(embeds, 0)                                # (N, 384) raw pooled — no normalize
Y = torch.tensor(labels, dtype=torch.float32)           # (N, 28) in [0, 1]

assert X.shape[0] == Y.shape[0] == len(texts)
torch.save(
    {'X': X, 'Y': Y, 'texts': texts, 'emotions': hp.EMOTIONS},
    'minilm_eqn.pt'
)
print(f'saved {X.shape[0]} rows → minilm_eqn.pt')