import os
import copy
import torch # type: ignore
import numpy as np # type: ignore
from sklearn.metrics import confusion_matrix # type: ignore
import matplotlib.pyplot as plt # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore  
from models.efficientnet_with_attention import EfficientNetWithAttention
import timm # type: ignore

def get_output_channels(model_name):
    if model_name in ["resnet18", "resnet50v2", "resnet101v2"]:
        return 512 if model_name == "resnet18" else 2048
    elif model_name in ["mobilenet_v2", "googlenet"]:
        return 1280
    elif model_name == "alexnet":
        return 4096
    elif model_name.startswith("efficientnet_b"):
        tf_variant_map = {
            "efficientnet_b4": "tf_efficientnet_b4_ns",
            "efficientnet_b5": "tf_efficientnet_b5_ns",
            "efficientnet_b6": "tf_efficientnet_b6_ns",
            "efficientnet_b7": "tf_efficientnet_b7_ns",
        }
        tf_variant = tf_variant_map.get(model_name, model_name)
        backbone = timm.create_model(tf_variant, pretrained=True, num_classes=0)
        return backbone.num_features
    elif model_name.startswith("vgg"):
        return 4096
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_model_with_attention(model_name, num_classes, attention_type="none", pretrained=True,
                             fold=None, weights_root=None, resume=True, **kwargs):
    use_film_before = kwargs.get("use_film_before", False)
    use_film_in_cbam = kwargs.get("use_film_in_cbam", False)
    use_triplet_embedding = kwargs.get("use_triplet_embedding", False)
    triplet_embedding_dim = kwargs.get("triplet_embedding_dim", 128)
    include_skin_vec = kwargs.get("include_skin_vec", True)

    # ✅ Remap tf-efficientnet variants for b4–b7
    tf_variant_map = {
        "efficientnet_b4": "tf_efficientnet_b4_ns",
        "efficientnet_b5": "tf_efficientnet_b5_ns",
        "efficientnet_b6": "tf_efficientnet_b6_ns",
        "efficientnet_b7": "tf_efficientnet_b7_ns",
    }
    tf_variant = tf_variant_map.get(model_name, model_name)

    # Get output channels dynamically
    C = get_output_channels(model_name)
    if model_name.startswith("efficientnet_b"):
        model = EfficientNetWithAttention(
            num_classes=num_classes,
            attention_type=attention_type,
            pretrained=pretrained,
            use_film_before=use_film_before,
            use_film_in_cbam=use_film_in_cbam,
            use_triplet_embedding=use_triplet_embedding,
            triplet_embedding_dim=triplet_embedding_dim,
            include_skin_vec=include_skin_vec,
            efficientnet_variant=tf_variant  # 👈 Use tf variant
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 2. Load weights if available
    if resume and weights_root and fold is not None:
        subdir = f"{model_name}_{attention_type}"
        model_dir = os.path.join(weights_root, subdir)
        checkpoint_path = os.path.join(model_dir, "checkpoints", f"{subdir}_fold{fold}_checkpoint.pth")
        best_weights_path = os.path.join(model_dir, "weights", f"{subdir}_fold{fold}_best.pth")

        def safe_load(path):
            try:
                state = torch.load(path, map_location="cpu")
                if isinstance(state, dict) and "model_state" in state:
                    model.load_state_dict(state["model_state"])
                else:
                    model.load_state_dict(state)
                print(f"✅ Successfully loaded weights from {path}")
            except RuntimeError as e:
                print(f"❌ Failed to load weights due to shape mismatch in {path}:\n{e}")

        if os.path.isfile(checkpoint_path):
            print(f"💾 Resuming from checkpoint: {checkpoint_path}")
            safe_load(checkpoint_path)
        elif os.path.isfile(best_weights_path):
            print(f"💾 Loading best weights: {best_weights_path}")
            safe_load(best_weights_path)
        elif pretrained:
            print(f"📦 No saved weights found — using pretrained weights for {model_name.upper()}")
        else:
            print(f"⚠️ No pretrained or saved weights found for {model_name.upper()}")

    return model

def compute_classwise_alpha(
    y_true,
    y_pred,
    num_classes=4,
    normalize=True,
    clip_range=(0.1, 3.0),
    prev_alpha=None,
    beta=0.9,
    smoothing=True
):
    """
    Compute smoothed, capped alpha weights for Focal Loss based on inverse recall.

    Args:
        y_true (array): Ground truth labels.
        y_pred (array): Predicted labels.
        num_classes (int): Number of classes.
        normalize (bool): Whether to normalize alpha to sum to num_classes.
        clip_range (tuple): Min and max values to clip alpha.
        prev_alpha (np.ndarray or torch.Tensor): Previous epoch's alpha for smoothing.
        beta (float): Smoothing factor for EMA.
        smoothing (bool): Whether to apply exponential smoothing.

    Returns:
        torch.Tensor: Alpha weights.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    recalls = cm.diagonal() / (cm.sum(axis=1) + 1e-6)  # Avoid division by zero
    alphas = 1.0 / (recalls + 1e-6)

    # Clip alpha to avoid extreme weights
    alphas = np.clip(alphas, clip_range[0], clip_range[1])

    # Smooth with EMA using previous alpha
    if smoothing and prev_alpha is not None:
        if isinstance(prev_alpha, torch.Tensor):
            prev_alpha = prev_alpha.detach().cpu().numpy()
        alphas = beta * prev_alpha + (1 - beta) * alphas


    # Normalize to keep total scale constant
    if normalize:
        alphas = alphas / alphas.sum() * num_classes

    print("🔍 Dynamic Alpha (inverse recall):", np.round(alphas, 4))
    return torch.tensor(alphas, dtype=torch.float32)

class GradualUnfreezer:
    def __init__(self, model, base_lr=0.001, start_epoch=None, unfreeze_every=1, max_blocks=None, weight_decay=1e-4):
        self.model = model
        self.base_lr = base_lr
        self.unfreeze_every = unfreeze_every
        self.weight_decay = weight_decay

        self.children = list(model.base.children())
        self.total_blocks = len(self.children)
        self.max_blocks = max_blocks if max_blocks is not None else self.total_blocks
        self.blocks_to_unfreeze = min(self.total_blocks, self.max_blocks)
        self.start_epoch = start_epoch or 0
        self.next_unfreeze_epoch = self.start_epoch
        self.current_block = self.total_blocks  # Start frozen

    def step(self, optimizer, epoch):
        if self.current_block <= self.total_blocks - self.blocks_to_unfreeze:
            return  # Done unfreezing target number of blocks

        if epoch >= self.next_unfreeze_epoch:
            self.current_block -= 1
            block = self.children[self.current_block]
            for param in block.parameters():
                param.requires_grad = True

            # ✅ Add block parameters with weight decay
            optimizer.add_param_group({
                'params': block.parameters(),
                'lr': self.base_lr * 0.01,
                'weight_decay': self.weight_decay
            })

            print(f"🔥 GradualUnfreezer: Unfroze block {self.current_block}/{self.total_blocks}")
            self.next_unfreeze_epoch += self.unfreeze_every

class PostWarmupLRScheduler:
    def __init__(self, optimizer, base_lr=0.001, rise_epochs=3, weight_decay=1e-4):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.rise_epochs = rise_epochs
        self.epoch_count = 0
        self.weight_decay = weight_decay

    def step(self):
        if self.epoch_count >= self.rise_epochs:
            return

        new_lr = self.base_lr * (self.epoch_count + 1) / self.rise_epochs
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = new_lr
        self.epoch_count += 1

        print(f"📈 LR Increase: Set LR to {new_lr:.6f}")

def setup_directories(base_path, model_name, fold=None, attention_type=None):
    """
    Create directory structure:
    base_path/fold_{N}_{model_name}_{attention_type}/[checkpoints, weights, graphs, predictions]
    """
    attention_str = str(attention_type).lower() if attention_type else "none"
    fold_str = f"fold_{fold}" if fold is not None else "fold_None"
    tag = f"{fold_str}_{model_name}_{attention_str}"

    model_root = os.path.join(base_path, tag)  # ✅ Use tag as subdirectory

    checkpoint_dir = os.path.join(model_root, "checkpoints")
    weights_dir = os.path.join(model_root, "weights")
    graph_dir = os.path.join(model_root, "graphs")
    predictions_dir = os.path.join(model_root, "predictions")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"{tag}_checkpoint.pth")
    best_weights_path = os.path.join(weights_dir, f"{tag}_best.pth")

    return checkpoint_path, best_weights_path, graph_dir, predictions_dir

'''def compute_classwise_alpha(y_true, y_pred, num_classes=4, normalize=True):
    """
    Compute alpha weights for Focal Loss based on per-class recall.
    Classes with low recall get higher weights.

    Args:
        y_true (list or array): Ground truth class indices.
        y_pred (list or array): Predicted class indices.
        num_classes (int): Total number of classes.
        normalize (bool): Whether to normalize weights to sum to num_classes.

    Returns:
        torch.Tensor: Tensor of alpha weights for each class.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    recalls = cm.diagonal() / (cm.sum(axis=1) + 1e-6)  # Avoid divide-by-zero
    alphas = 1.0 / (recalls + 1e-6)  # Inverse recall: lower recall → higher weight

    if normalize:
        alphas = alphas / alphas.sum() * num_classes  # Normalize to keep scale stable
    
    print("🔍 Dynamic Alpha (inverse recall):", np.round(alphas, 3))

    return torch.tensor(alphas, dtype=torch.float32)'''

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        """
        Args:
            alpha (Tensor, Matrix or None): Class weights or class × MST weights of shape (C,) or (C, M).
            gamma (float): Focusing parameter.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Can be 1D or 2D
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mst_groups=None):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)

        targets = targets.view(-1, 1)
        logpt = logpt.gather(1, targets).squeeze(1)
        pt = pt.gather(1, targets).squeeze(1)

        if self.alpha is not None:
            if self.alpha.dim() == 2 and mst_groups is not None:
                # α per (class, mst_group)
                indices = torch.stack([targets.squeeze(), mst_groups], dim=1)
                at = self.alpha.to(inputs.device)[indices[:, 0], indices[:, 1]]
            else:
                # standard per-class α
                at = self.alpha.to(inputs.device).gather(0, targets.squeeze())
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def compute_class_mst_alpha_matrix(y_true, y_pred, mst_bins, num_classes=7, num_mst_bins=10, normalize=True):
    """
    Compute alpha weights per (class, MST bin) based on inverse recall.
    Low recall = high alpha.

    Args:
        y_true: list or array of true class indices (B,)
        y_pred: list or array of predicted class indices (B,)
        mst_bins: list or array of MST bin indices (B,)
        num_classes: number of classes
        num_mst_bins: number of MST bins (default 10 for full Monk scale)
        normalize: normalize alphas per class row to sum to 1

    Returns:
        alpha_matrix: Tensor of shape [num_classes, num_mst_bins]
    """
    alpha_matrix = np.ones((num_classes, num_mst_bins), dtype=np.float32)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mst_bins = np.array(mst_bins)

    for cls in range(num_classes):
        for mst in range(num_mst_bins):
            mask = (y_true == cls) & (mst_bins == mst)
            if mask.sum() == 0:
                alpha_matrix[cls, mst] = 1.0  # fallback
                continue

            y_true_subset = y_true[mask]
            y_pred_subset = y_pred[mask]

            recall = np.sum(y_pred_subset == cls) / (len(y_true_subset) + 1e-6)
            alpha_matrix[cls, mst] = 1.0 / (recall + 1e-6)  # inverse recall

    if normalize:
        # Normalize each row (per class) to sum to num_mst_bins
        alpha_matrix = alpha_matrix / alpha_matrix.sum(axis=1, keepdims=True) * num_mst_bins

    print("📊 Alpha Matrix (class × MST):")
    print(np.round(alpha_matrix, 2))

    return torch.tensor(alpha_matrix, dtype=torch.float32)

def freeze_backbone(model):
    for param in model.base.parameters():
        param.requires_grad = False
    print("🧊 Backbone frozen.")

def safe_criterion_call(criterion, outputs, labels, mst_groups=None):
    try:
        return criterion(outputs, labels, mst_groups)
    except TypeError:
        return criterion(outputs, labels)

def plot_alpha_trends(alpha_history, num_classes, save_path=None):
    alpha_array = torch.stack(alpha_history).cpu().numpy()
    for class_idx in range(num_classes):
        plt.plot(alpha_array[:, class_idx], label=f"Class {class_idx}")
    plt.xlabel("Epoch")
    plt.ylabel("Alpha Weight")
    plt.title("Focal Loss Alpha Trend (Inverse Recall)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"📊 Alpha plot saved to: {save_path}")
    else:
        plt.show()