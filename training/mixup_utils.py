
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F  # type: ignore
import numpy as np  # type: ignore

def soft_cross_entropy(pred, soft_targets):
    log_probs = F.log_softmax(pred, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()

def mixup_criterion(pred, y_a, y_b, lam, num_classes):
    y_a = F.one_hot(y_a.long(), num_classes=num_classes).float()
    y_b = F.one_hot(y_b.long(), num_classes=num_classes).float()
    soft_targets = lam * y_a + (1 - lam) * y_b
    return soft_cross_entropy(pred, soft_targets)

def mixup_data(x, y, skin_vec, alpha=0.2, epoch=0, warmup_epochs=5):
    if x.ndim != 4 or skin_vec.ndim != 2:
        raise ValueError(f"Expected x [B,C,H,W] and skin_vec [B,D], got {x.shape} and {skin_vec.shape}")

    if epoch < warmup_epochs:
        lam = 1.0
    else:
        lam = float(np.clip(np.random.beta(alpha, alpha), 0.3, 0.7))

    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_skin = lam * skin_vec + (1 - lam) * skin_vec[index]
    y_a, y_b = y, y[index]

    #print(f"[Mixup] λ = {lam:.3f} | epoch: {epoch}")
    return mixed_x, mixed_skin, y_a, y_b, lam