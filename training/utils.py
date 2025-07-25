import os
import copy
import torch # type: ignore
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

def unfreeze_backbone(model, last_n_layers=None, base_lr=0.001):
    children = list(model.base.children())
    total = len(children)
    param_groups = []

    for i, child in enumerate(children):
        if last_n_layers is None or i >= total - last_n_layers:
            for param in child.parameters():
                param.requires_grad = True
            param_groups.append({'params': child.parameters(), 'lr': base_lr * 0.01})  # Lower LR for base
        else:
            for param in child.parameters():
                param.requires_grad = False

    print(f"🔥 Unfroze last {last_n_layers if last_n_layers else 'all'} EfficientNet blocks.")
    return param_groups

class GradualUnfreezer:
    def __init__(self, model, base_lr=0.001, start_block=None):
        self.model = model
        self.base_lr = base_lr
        self.children = list(model.base.children())
        self.total_blocks = len(self.children)
        self.current_block = start_block if start_block is not None else self.total_blocks

    def step(self, optimizer, epoch):
        if self.current_block <= 0:
            return  # All blocks unfrozen

        self.current_block -= 1
        block = self.children[self.current_block]
        for param in block.parameters():
            param.requires_grad = True
        optimizer.add_param_group({'params': block.parameters(), 'lr': self.base_lr * 0.01})
        print(f"🔥 GradualUnfreezer: Unfroze block {self.current_block}/{self.total_blocks}")

class PostWarmupLRScheduler:
    def __init__(self, optimizer, base_lr, rise_epochs=3, max_lr=None):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr if max_lr else base_lr
        self.rise_epochs = rise_epochs
        self.epoch = 0

    def step(self):
        if self.epoch < self.rise_epochs:
            factor = (self.epoch + 1) / self.rise_epochs
            new_lr = self.base_lr * factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = min(new_lr, self.max_lr)
            print(f"📈 LR Increase: Set LR to {new_lr:.6f}")
            self.epoch += 1

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

def compute_classwise_alpha(y_true, y_pred, num_classes=4, normalize=True):
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

    return torch.tensor(alphas, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        """
        Args:
            alpha (Tensor or None): Class weights to handle imbalance (e.g., tensor([0.8, 1.2, 1.0, ...])).
                                   If None, all classes are weighted equally.
            gamma (float): Focusing parameter. Higher values down-weight easy examples more.
            reduction (str): Specifies reduction mode: 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Model output logits of shape (B, C) — unnormalized scores.
            targets (Tensor): Ground-truth class indices of shape (B,) — values in [0, C-1].
        Returns:
            Focal loss value (scalar or tensor depending on reduction).
        """

        # Apply log-softmax to get log-probabilities for each class
        logpt = F.log_softmax(inputs, dim=1)

        # Convert log-probabilities to actual probabilities
        pt = torch.exp(logpt)

        # Reshape targets to (B, 1) for proper indexing
        targets = targets.view(-1, 1)

        # Gather the log probability corresponding to the true class for each sample
        logpt = logpt.gather(1, targets).squeeze(1)
        pt = pt.gather(1, targets).squeeze(1)

        # If alpha is provided, apply class-specific weights
        if self.alpha is not None:

            # Gather the alpha weight for each sample's true class
            at = self.alpha.to(inputs.device).gather(0, targets.squeeze())
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        # Apply the specified reduction (mean, sum, or none)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
def freeze_backbone(model):
    for param in model.base.parameters():
        param.requires_grad = False
    print("🧊 Backbone frozen.")
