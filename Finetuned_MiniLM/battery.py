"""
battery.py — diagnostic battery for frozen vs fine-tuned models.

Usage:
    python battery.py                        # tests emotion_head_best.pt
    python battery.py frozen.pt              # tests specific checkpoint
    python battery.py frozen.pt ft.pt        # side-by-side comparison
"""
import sys
import os
from pathlib import Path
import torch
import hparam as hp

from model import EmotionClassifierHybrid
from transformers import AutoTokenizer


# ─── Battery ────────────────────────────────────────────────────
CASES = [
    # (category, expected, text) — expected: str | 'a|b' | None
    ('baseline', 'joy',            "I'm so happy!"),
    ('baseline', 'sadness',        "I feel terrible, everything is falling apart."),
    ('baseline', 'anger',          "I am furious right now."),
    ('baseline', 'fear',           "I'm terrified."),
    ('baseline', 'love',           "I love you more than anything."),
    ('baseline', 'gratitude',      "Thank you so much, I really appreciate it."),
    ('baseline', 'admiration',     "You're an incredible person, I truly admire you."),
    ('baseline', 'amusement',      "That was hilarious, I can't stop laughing."),
    ('baseline', 'embarrassment',  "I'm so embarrassed, I want to disappear."),
    ('baseline', 'nervousness',    "I'm really nervous about the interview tomorrow."),
    ('baseline', 'desire',         "I really want this more than anything."),
    ('baseline', 'neutral',        "The meeting is at 3 PM in room 4."),
    ('baseline', 'excitement',     "I'm so excited for the trip!"),
    ('baseline', 'grief',          "I miss them so much it hurts."),
    ('baseline', 'pride',          "I'm so proud of what I accomplished today."),
    ('baseline', 'disappointment', "I'm so let down by what happened."),
    ('baseline', 'relief',         "Thank god it's finally over."),
    ('baseline', 'optimism',       "I'm sure things are going to get better."),
    ('baseline', 'remorse',        "I deeply regret hurting you."),
    ('baseline', 'caring',         "Please take care of yourself, I'm worried about you."),

    ('cooccur',  'sadness|joy',            "I'm sad but also happy."),
    ('cooccur',  'nervousness|excitement', "I'm nervous and excited about tomorrow."),
    ('cooccur',  'gratitude|sadness',      "I'm grateful but heartbroken."),
    ('cooccur',  'anger|disappointment',   "I'm angry and disappointed in you."),
    ('cooccur',  'love|fear',              "I love you but I'm scared of losing you."),

    ('ood',      'desire',     "I'm horny."),
    ('ood',      'desire',     "I'm craving your touch."),
    ('ood',      'joy',        "This is lit."),
    ('ood',      'excitement', "I'm stoked."),
    ('ood',      'anger',      "I'm pissed off."),

    ('neutral',  'neutral', "The table is brown."),
    ('neutral',  'neutral', "I went to the store."),
    ('neutral',  'neutral', "Water boils at 100°C."),
]


# ─── Load (your functions, unchanged) ───────────────────────────
def load_classifier(checkpoint, device):
    tokenizer = AutoTokenizer.from_pretrained(hp.MODEL)
    model = EmotionClassifierHybrid(
        n_emotions=hp.NUM_EMOTIONS,
        input_dim=hp.input_dim,
        dropout=0.25,
        minilm=hp.MODEL,
        n_keep=0,   # all frozen for inference; weights still load
    )
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return tokenizer, model


@torch.inference_mode()
def predict(texts, tokenizer, model, device, top_k=5):
    encoded = tokenizer(
        texts, padding=True, truncation=True,
        max_length=hp.MAX_LEN, return_tensors="pt",
    ).to(device)
    logits = model(encoded["input_ids"], encoded["attention_mask"])
    probabilities = torch.sigmoid(logits).cpu()
    k = min(max(top_k, 1), len(hp.EMOTIONS))

    return [
        [(hp.EMOTIONS[i], float(v)) for v, i in
         zip(*[t.tolist() for t in torch.topk(row, k=k)])]
        for row in probabilities
    ]


# ─── Check logic ────────────────────────────────────────────────
def check(category, expected, top_results):
    names = [e for e, _ in top_results]
    if category == 'baseline':
        return (names[0] == expected, f'top1={names[0]}')
    if category == 'cooccur':
        needed = expected.split('|')
        hits = [e for e in needed if e in names]
        return (len(hits) == len(needed), f'hits={hits}')
    if category == 'ood':
        return (expected in names, f'top3={names[:3]}')
    if category == 'neutral':
        return (names[0] == 'neutral', f'top1={names[0]}')
    return (None, '')


# ─── Runner ─────────────────────────────────────────────────────
def evaluate(checkpoint, device, top_k=5):
    print(f'\n{"="*70}')
    print(f'CHECKPOINT: {checkpoint}')
    print(f'{"="*70}')

    tokenizer, model = load_classifier(checkpoint, device)
    print(f'  loaded {len(model.state_dict())} params')

    texts = [t for _, _, t in CASES]
    top_results = predict(texts, tokenizer, model, device, top_k=top_k)

    total, passed = 0, 0
    per_cat = {}
    for (cat, expected, text), top in zip(CASES, top_results):
        ok, note = check(cat, expected, top)
        mark = {True: '[✓]', False: '[✗]', None: '[ ]'}[ok]
        tops_str = '  '.join(f'{e}:{v:.2f}' for e, v in top[:3])
        print(f'{mark} [{cat:8s}] "{text}"')
        print(f'       {tops_str}   {note}')

        if ok is not None:
            total += 1
            passed += int(ok)
            per_cat.setdefault(cat, [0, 0])
            per_cat[cat][0] += int(ok)
            per_cat[cat][1] += 1

    print(f'\n── Per-category ──')
    for cat, (p, t) in per_cat.items():
        print(f'  {cat:10s}: {p}/{t}  ({100*p/t:.0f}%)')
    print(f'\nTOTAL: {passed}/{total} ({100*passed/total:.1f}%)')

    return per_cat


def diff(r1, r2, name1, name2):
    print(f'\n{"="*70}')
    print(f'DELTA: {name2} − {name1}')
    print(f'{"="*70}')
    for cat in sorted(set(r1) | set(r2)):
        p1, t1 = r1.get(cat, (0, 1))
        p2, t2 = r2.get(cat, (0, 1))
        delta = (p2 / t2) - (p1 / t1)
        arrow = '↑' if delta > 0.01 else '↓' if delta < -0.01 else '='
        print(f'  {cat:10s}: {p1}/{t1} → {p2}/{t2}  ({100*delta:+.0f}% {arrow})')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    ckpts = sys.argv[1:3] if len(sys.argv) > 1 else ['emotion_head_best.pt']
    missing = [c for c in ckpts if not os.path.exists(c)]
    if missing:
        print(f'ERROR: missing checkpoints: {missing}')
        sys.exit(1)

    results = {}
    for ckpt in ckpts:
        results[ckpt] = evaluate(ckpt, device)

    if len(results) == 2:
        diff(results[ckpts[0]], results[ckpts[1]], ckpts[0], ckpts[1])