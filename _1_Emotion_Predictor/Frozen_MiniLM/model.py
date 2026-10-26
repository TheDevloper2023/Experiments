import torch
import torch.nn as nn
import torch.nn.functional as F
import _1_Emotion_Predictor.Frozen_MiniLM.hparam as hp

class EmotionClassifierV1(nn.Module):
    """
    Sanity check to see if the idea can even work
    """
    def __init__(self, n_emotions, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_emotions)

    def forward(self, x):
        return self.fc(x)


class EmotionClassifierV2(nn.Module):
    def __init__(self, n_emotions, input_dim, hidden=None, dropout=0.15):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, n_emotions)   # raw logits

    def forward(self, x):
        return self.head(self.trunk(x))


# Vibe code this because
# V1 can do things that V2 cannot and V2 is richer within rarer emotions, this should fix it?
class EmotionClassifierHybrid(nn.Module):
    def __init__(self, n_emotions, input_dim, hidden=None, dropout=0.15):
        super().__init__()
        h = hidden or input_dim

        # Linear path — preserves per-class independence, calibrated
        self.linear = nn.Linear(input_dim, n_emotions)

        # Nonlinear path — adds capacity for shared/rare signal
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mlp_head = nn.Linear(h, n_emotions)

        # CRITICAL: zero-init the MLP head so training starts at fc behavior.
        # At step 0, output == fc(x) exactly. Nonlinearity only "turns on"
        # if it reduces loss.
        nn.init.zeros_(self.mlp_head.weight)
        nn.init.zeros_(self.mlp_head.bias)

    def forward(self, x):
        return self.linear(x) + self.mlp_head(self.trunk(x))