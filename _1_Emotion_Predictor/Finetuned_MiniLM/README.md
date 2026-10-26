# Lunar Emotion Predictor

Partially fine-tuned MiniLM-L6 on GoEmotions. Unlike most emotion classifiers, 
this one gives you **intensity scores** for each of 28 emotions — not just a 
single label. It also preserves **mixed emotions** ("sad but happy" fires both).

## Results

33-case diagnostic battery (baseline, co-occurrence, OOD, neutral):

| Category | Frozen | **Fine-tuned** |
|---|---|---|
| Baseline (20) | 75% | **85%** |
| Co-occurrence (5) | 5/5 | **5/5** |
| OOD (5) | 60% | **80%** |
| Neutral (3) | 3/3 | **3/3** |
| **Total** | **75.8%** | **87.9%** |

On "I'm sad but also happy" → `sadness=0.94, joy=0.38` — bittersweet preserved 
(naive fine-tune gives `joy=0.21`, which collapses the emotion).

## How

- **Focal loss (γ=1.0)** on soft `[0,1]` targets — lifts rare classes without noise
- **Co-occurrence penalty (λ=0.1)** — auxiliary loss that stops the model from 
  collapsing mixed emotions into a single dominant label
- **Partial fine-tune** — freeze embedding + bottom 3 blocks, train top 3 + 
  hybrid linear/MLP head (5.6M / 23M trainable params)

## Usage

```python
import torch
from transformers import AutoTokenizer
from model import EmotionClassifierHybrid

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = EmotionClassifierHybrid(28, 384, minilm='sentence-transformers/all-MiniLM-L6-v2')
model.load_state_dict(torch.load('saved/emotion_head_final_v1.pt', map_location=device))
model.to(device).eval()

with torch.inference_mode():
    enc = tokenizer("I'm sad but also happy", truncation=True,
                    max_length=256, return_tensors='pt').to(device)
    probs = torch.sigmoid(model(enc['input_ids'], enc['attention_mask']))
    # probs: 28-d vector in [0, 1]