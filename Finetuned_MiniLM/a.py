"""
analyze_emotions.py — rare-class analysis for GoEmotions soft-label dataset

Usage:
    python analyze_emotions.py                 # uses minilm_eqn.pt
    python analyze_emotions.py path/to/blob.pt # custom path
"""
import sys
import torch
import hparam as hp


def analyze(blob_path='minilm_eqn.pt', 
            freq_threshold=0.05,      # "rare" if < 5% of samples have this emotion
            intensity_threshold=0.10, # ... or if mean intensity < 0.10
            top_k_hard=0.50):         # ... or if top-k threshold needs to be > 0.5
    blob = torch.load(blob_path, map_location='cpu')
    Y = blob['Y'].float()                    # (N, 28) in [0,1]
    emotions = blob['emotions']              # list of 28 names
    N = Y.shape[0]

    print(f'Loaded {blob_path}: {N} samples, {len(emotions)} classes\n')

    # ---- per-class stats ---------------------------------------------------
    # "positives" defined multiple ways — each tells you something different
    pos_any    = (Y > 0.0).float().mean(0)         # fraction with *any* signal
    pos_half   = (Y > 0.5).float().mean(0)         # fraction "strongly" positive
    mean_int   = Y.mean(0)                         # average intensity
    std_int    = Y.std(0)                          # intensity variance
    p95_int    = Y.quantile(0.95, dim=0)           # 95th percentile
    max_int    = Y.max(0).values                   # max seen
    pos_mass   = Y.sum(0)                          # total "positive mass" per class
    neg_mass   = N - pos_mass                      # for pos_weight

    # ---- rarity flags ------------------------------------------------------
    rare_by_freq = pos_half < freq_threshold       # < 5% of samples > 0.5
    rare_by_int  = mean_int < intensity_threshold  # mean intensity < 0.10
    rare_by_tail = p95_int < top_k_hard            # 95th percentile still weak

    # composite: rare if ≥ 2 of the 3
    is_rare = (rare_by_freq.long() + rare_by_int.long() + rare_by_tail.long()) >= 2

    # ---- sorted table ------------------------------------------------------
    order = torch.argsort(mean_int, descending=True)
    header = (f'{"emotion":15s} {"mean":>6s} {"std":>6s} '
              f'{"P(>0)":>7s} {"P(>0.5)":>8s} {"P95":>6s} {"max":>6s} '
              f'{"pw":>7s}  flags')
    print(header)
    print('-' * len(header))

    for i in order.tolist():
        name = emotions[i]
        flags = []
        if rare_by_freq[i]: flags.append('freq')
        if rare_by_int[i]:  flags.append('int')
        if rare_by_tail[i]: flags.append('tail')
        flag_str = f'  [{"|".join(flags)}]' if flags else ''
        # pos_weight cap preview
        pw = (neg_mass[i] / max(pos_mass[i].item(), 1.0)).item()
        print(f'{name:15s} '
              f'{mean_int[i]:>6.3f} '
              f'{std_int[i]:>6.3f} '
              f'{pos_any[i]:>7.3f} '
              f'{pos_half[i]:>8.3f} '
              f'{p95_int[i]:>6.3f} '
              f'{max_int[i]:>6.3f} '
              f'{pw:>7.2f}{flag_str}')

    # ---- summary -----------------------------------------------------------
    print()
    print(f'Rare classes (≥2 of 3 criteria): {is_rare.sum().item()} / {len(emotions)}')
    for i in is_rare.nonzero(as_tuple=True)[0].tolist():
        print(f'  {emotions[i]:15s}  '
              f'P(>0.5)={pos_half[i]:.4f}  '
              f'mean={mean_int[i]:.3f}  '
              f'P95={p95_int[i]:.3f}')

    # ---- suggested pos_weight (capped) -------------------------------------
    print()
    print('Suggested pos_weight (capped at 8, rare classes only):')
    print('RARE = {', ', '.join(f"'{emotions[i]}'" for i in is_rare.nonzero(as_tuple=True)[0].tolist()), '}')
    print()
    print('pw = torch.ones(28)')
    print('for i, e in enumerate(hp.EMOTIONS):')
    print('    if e in RARE:')
    print(f'        pw[i] = (neg_mass[i] / pos_mass[i].clamp(min=1)).clamp(max=8)')

    # ---- per-class variance diagnostic -------------------------------------
    # Classes with near-zero std are effectively constant predictors
    # (the model can't learn them from features if the training labels don't vary)
    print()
    print('Low-variance classes (std < 0.05) — model can barely learn these:')
    low_var = (std_int < 0.05).nonzero(as_tuple=True)[0].tolist()
    if not low_var:
        print('  (none)')
    for i in low_var:
        print(f'  {emotions[i]:15s}  std={std_int[i]:.4f}')

    # ---- co-occurrence snapshot --------------------------------------------
    # How often do pairs co-occur? Useful for the WDL / focal decision.
    print()
    print('Top co-occurring pairs (both intensities > 0.3):')
    strong = (Y > 0.3).float()
    cooc = strong.T @ strong                                   # (28, 28)
    diag = torch.diag(cooc).clone()
    cooc.fill_diagonal_(0)
    flat = cooc.flatten()
    topk = torch.topk(flat, k=10)
    for val, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        i, j = idx // len(emotions), idx % len(emotions)
        if i < j and val > 0:
            print(f'  {emotions[i]:15s} + {emotions[j]:15s}  count={int(val):5d}')

    return {
        'pos_any': pos_any, 'pos_half': pos_half, 'mean_int': mean_int,
        'std_int': std_int, 'p95_int': p95_int,
        'is_rare': is_rare, 'emotions': emotions,
    }


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'minilm_eqn.pt'
    analyze(path)