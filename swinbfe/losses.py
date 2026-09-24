"""Loss terms. Every run in the paper uses binary cross-entropy plus a
Tversky term with alpha = beta = 0.5, at which setting the Tversky index
equals the soft Dice coefficient (Eq. 9). The reference-protocol runs use
binary cross-entropy alone.
"""

import torch
import torch.nn.functional as F

EPS = 1e-6


def tversky_loss(p, g, alpha=0.5, beta=0.5):
    tp = (p * g).sum()
    fp = (p * (1 - g)).sum()
    fn = ((1 - p) * g).sum()
    return 1 - (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)


def compute_losses(out, batch, alpha=0.5, beta=0.5, tversky_weight=1.0):
    """Return a dict with 'bce', 'tversky' and their weighted 'total'."""
    g = batch["mask"]
    logits = out["logits"]
    p = torch.sigmoid(logits)
    L = {
        "bce": F.binary_cross_entropy_with_logits(logits, g),
        "tversky": tversky_loss(p, g, alpha, beta),
    }
    L["total"] = L["bce"] + tversky_weight * L["tversky"]
    return L
