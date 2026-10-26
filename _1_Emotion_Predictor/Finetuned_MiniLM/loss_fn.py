import torch
import torch.nn as nn
import torch.nn.functional as F



# Test this
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, pos_weight=None):
        super().__init__()
        self.gamma, self.pos_weight = gamma, pos_weight
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight)
        pt = targets * p + (1 - targets) * (1 - p)
        return ((1 - pt).pow(self.gamma) * ce).mean()



# I ask deepseek about fixing some predictions where I need happy and sad at the same time.
# The Co-occurrence Penalty (Interspeech 2022) fixes he said


def compute_cooccurrence_matrix(Y, eps=1e-6):
    """Y: (N, C) labels. Returns (C, C) matrix of P(j|i)."""
    Y_bin = (Y > 0.5).float()
    co_pos = Y_bin.T @ Y_bin
    pos_counts = Y_bin.sum(0)
    return co_pos / (pos_counts.unsqueeze(1) + eps)

def cooccurrence_penalty(logits, true_cooc_matrix, eps=1e-6):
    """Penalizes deviation from true co-occurrence structure."""
    probs = torch.sigmoid(logits)
    probs_norm = probs / (probs.sum(0, keepdim=True) + eps)
    batch_cooc = probs.T @ probs_norm
    return F.mse_loss(batch_cooc, true_cooc_matrix)