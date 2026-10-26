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



# Focal fixes some problems with co-occurence (allowes sad but happy and so on) but is a bit nosiy
# BCE has issues but is somewhat sharper
# What if I combine it?


class BCEFocalMixed(nn.Module):
    def __init__(self, gamma=1.0, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        # Standard BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal = BCE × (1 - p_t)^gamma
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal = (1 - pt).pow(self.gamma) * bce
        
        return (self.alpha * bce + (1 - self.alpha) * focal).mean()
    