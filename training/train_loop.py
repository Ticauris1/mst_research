import os
import copy
import gc
from tqdm import tqdm # type: ignore
from contextlib import nullcontext
import time
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
from sklearn.utils.class_weight import compute_class_weight # type: ignore
from torch.cuda.amp import autocast as autocast_cuda # type: ignore
from training.lr_scheduler import HybridLRScheduler # type: ignore
from training.utils import GradualUnfreezer, PostWarmupLRScheduler, freeze_backbone, FocalLoss
from training.utils import setup_directories, compute_classwise_alpha # Removed plot_alpha_trends as it's likely a plotting function
from training.mixup_utils import mixup_data, mixup_criterion # type: ignore
#from evaluation.plot_utils import plot_training_curves # type: ignore
from contextlib import nullcontext
import sys, time
import traceback


# === Main Training Loop ===
def local_train(
    train_loader, model, device, num_epochs=10, lr=0.003,
    val_loader=None, save_model_path=None, model_name="model",
    fold=None, resume_path=None, alpha=0.2, mixup_enabled=True,
    warmup_epochs=4, num_classes=4, attention_type="none",
    log_lr_each_epoch=True, y_train=None
):
    """
    Executes a local training loop for a given model.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        model (nn.Module): The neural network model to train.
        device (torch.device): The device (CPU, CUDA, MPS) to train on.
        num_epochs (int): Total number of training epochs.
        lr (float): Initial learning rate.
        val_loader (DataLoader, optional): DataLoader for the validation set. Defaults to None.
        save_model_path (str, optional): Base path to save model checkpoints and weights. Defaults to None.
        model_name (str, optional): Name of the model for directory structuring. Defaults to "model".
        fold (int, optional): Current fold number in k-fold cross-validation. Defaults to None.
        resume_path (str, optional): Path to a checkpoint file to resume training from. Defaults to None.
        alpha (float): Initial alpha value for FocalLoss. Defaults to 0.2.
        mixup_enabled (bool): Whether to use Mixup data augmentation. Defaults to True.
        warmup_epochs (int): Number of epochs for learning rate warmup. Defaults to 4.
        num_classes (int): Number of output classes for the model.
        attention_type (str, optional): Type of attention mechanism used. Defaults to "none".
        log_lr_each_epoch (bool): Whether to print learning rate each epoch. Defaults to True.
        y_train (list): True labels of the training set (used for class weighting).

    Returns:
        tuple: A tuple containing:
            - model (nn.Module): The best trained model based on validation accuracy.
            - dict: A dictionary containing training history (train/val loss, train/val accuracy, LRs).
    """

    # Optimizer initialization
    optimizer = torch.optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': lr},
        {'params': model.skin_mlp.parameters(), 'lr': lr},
    ], weight_decay=1e-4)

    # Add attention module parameters to optimizer if present and trainable
    if hasattr(model, 'attn') and not isinstance(model.attn, nn.Identity):
        optimizer.add_param_group({'params': model.attn.parameters(), 'lr': lr, 'weight_decay': 1e-4})

    # === Initialize History Lists (moved up to avoid re-initialization) ===
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    lrs_history = []

    # === Class weighting ===
    # Compute balanced class weights for the loss function based on training data
    class_weights_np = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    weights = torch.tensor(class_weights_np, dtype=torch.float, device=device)

    # Define loss functions
    ce_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    focal_criterion = FocalLoss(alpha=weights) # Initial alpha for focal loss
    criterion = ce_criterion # Default criterion is CrossEntropyLoss

    # === Initialize alpha tensor for focal loss smoothing ===
    # This 'alpha' is used by compute_classwise_alpha, not the initial FocalLoss alpha parameter
    alpha = weights.clone().detach()  # shape = (num_classes,)

    # Learning rate scheduler
    scheduler = HybridLRScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=num_epochs,
        mode='cosine', # Cosine annealing
        plateau_patience=10, # Patience for ReduceLROnPlateau component
        plateau_factor=0.5,  # Factor for ReduceLROnPlateau component
        min_lr=1e-6          # Minimum learning rate
    )

    # Gradient scaler for mixed precision training (if CUDA is available)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Setup directories for saving results (checkpoints, best weights, graphs, predictions)
    checkpoint_path, best_weights_path, graph_dir, predictions_dir = setup_directories(
        base_path=save_model_path,
        model_name=model_name,
        fold=fold,
        attention_type=attention_type or "none"
    )

    best_val_accuracy = 0.0
    best_model_weights = copy.deepcopy(model.state_dict()) # Store best model weights
    early_stop_counter = 0
    early_stop_patience = 12 # How many epochs to wait for improvement
    start_epoch = 0 # Starting epoch for the training loop

    # === Resume checkpoint ===
    # Attempt to load checkpoint if a resume_path is provided and valid
    if resume_path and os.path.isfile(resume_path):
        try:
            print(f"Attempting to resume training from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Successfully resumed training from checkpoint at Epoch {start_epoch}")
        except Exception as e:
            # If loading fails (e.g., corrupted file, parameter group mismatch)
            print(f"⚠️ Failed to load checkpoint from {resume_path}: {e}")
            print("Starting training from scratch (Epoch 0) instead.")
            # Reset states to ensure a fresh start
            best_val_accuracy = 0.0
            start_epoch = 0
            early_stop_counter = 0
    else:
        print(f"No valid checkpoint found at {resume_path}. Starting training from scratch (Epoch 0).")

    # === Initial backbone freezing ===
    # Freeze the backbone layers of the model during early warmup epochs
    if start_epoch < warmup_epochs:
        freeze_backbone(model)

    # === Gradual unfreezing and post-warmup LR adjustment tools ===
    unfreezer = GradualUnfreezer(
        model,
        base_lr=lr,
        start_epoch=warmup_epochs,
        unfreeze_every=5, # Unfreeze a block every 5 epochs after warmup
        max_blocks=2,    # Unfreeze up to 2 blocks
        weight_decay= 1e-4
    )
    lr_riser = PostWarmupLRScheduler(optimizer, base_lr=lr, rise_epochs=3)

    # Main training loop
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for dataset if it handles epoch-dependent augmentations
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)

        # Trigger unfreezing and LR adjustment AFTER warmup phase
        if epoch == warmup_epochs:
            print(f"🧊 Warmup complete — beginning gradual unfreezing.")
        unfreezer.step(optimizer, epoch)
        lr_riser.step()

        # Append current learning rate to history
        # Using get_lr() for modern PyTorch schedulers, [0] for the first param group
        lrs_history.append(scheduler.get_lr()[0]) # Use get_lr() instead of get_last_lr()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # === Update focal loss alpha dynamically (after warmup) ===
        if epoch >= warmup_epochs and val_loader:
            mixup_enabled = False # Disable mixup after warmup, or as per your logic
            y_true_val, y_pred_val = [], []
            model.eval() # Set model to evaluation mode for alpha calculation
            with torch.no_grad():
                for batch in val_loader:
                    v_images = batch[0].to(device)
                    v_labels = batch[1].to(device)
                    v_skin = batch[2].to(device) if model.include_skin_vec else None
                    v_triplet = batch[5].to(device) if model.use_triplet_embedding else None

                    out = model(v_images, v_skin, v_triplet)
                    y_pred_val.extend(out.argmax(dim=1).cpu().numpy())
                    y_true_val.extend(v_labels.cpu().numpy())

            # Compute dynamic alpha weights based on validation performance
            dynamic_alpha = compute_classwise_alpha(
                y_true_val, y_pred_val,
                num_classes=num_classes,
                normalize=True,
                clip_range=(0.1, 3.0), # Clip alpha values to prevent extreme weights
                prev_alpha=alpha,      # Use previous alpha for smoothing
                beta=0.9,              # Smoothing factor
                smoothing=True
            )
            alpha = dynamic_alpha.clone().detach() # Update alpha for the next epoch's focal loss
            #alpha_history.append(alpha) # Store alpha history (optional, for plotting alpha trends)
            focal_criterion = FocalLoss(alpha=dynamic_alpha) # Update FocalLoss with dynamic alpha
            criterion = focal_criterion # Switch to FocalLoss after warmup

        model.train() # Set model back to training mode
        total_loss, correct, total = 0.0, 0, 0

        # Training loop for the current epoch
        for batch in train_loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            skin_vecs = batch[2].to(device) if model.include_skin_vec else None
            triplet = batch[5].to(device) if model.use_triplet_embedding else None

            # === Mixup Data Augmentation ===
            # Apply mixup based on mixup_enabled flag and class conditions
            if mixup_enabled and not ((labels == 3).all() or (labels == 4).all()):
                mix_ratio = max(1.0 - epoch / num_epochs, 0.1) # Mixup ratio decreases over epochs
                images, skin_vecs, y_a, y_b, lam = mixup_data(
                    images, labels, skin_vecs, alpha * mix_ratio, epoch, warmup_epochs
                )
                use_mixup = True
            else:
                y_a, y_b = labels, labels # No mixup, labels remain as is
                lam = 1.0
                use_mixup = False

            optimizer.zero_grad() # Zero gradients before backward pass

            # === Forward Pass, Loss Calculation, and Backward Pass ===
            try:
                if scaler: # Use mixed precision if scaler is available (CUDA)
                    with autocast_cuda():
                        out = model(images, skin_vecs, triplet)
                        assert out.dim() == 2 and out.size(1) == num_classes, f"Bad logits: {out.shape}"
                        loss = mixup_criterion(out, y_a, y_b, lam, num_classes=num_classes) if use_mixup else criterion(out, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else: # Standard full precision training
                    out = model(images, skin_vecs, triplet)
                    assert out.dim() == 2 and out.size(1) == num_classes, f"Bad logits: {out.shape}"
                    loss = mixup_criterion(out, y_a, y_b, lam, num_classes=num_classes) if use_mixup else criterion(out, labels)
                    loss.backward()
                    optimizer.step()

            except Exception as e: # Catch errors during a training step
                print("\n🚨 STEP FAILURE")
                print(f"type: {type(e)}\nmsg: {e}")
                print("---- shapes at failure ----")
                print(f"images: {images.shape}, labels: {labels.shape}")
                if skin_vecs is not None: print(f"skin_vecs: {skin_vecs.shape}")
                if triplet is not None: print(f"triplet: {triplet.shape}")
                try:
                    print(f"logits (if computed): {out.shape}")
                except:
                    print("logits not computed")
                print("---------------------------")
                traceback.print_exc()
                raise # Re-raise the exception after printing debug info

            total_loss += loss.item() * labels.size(0)
            correct += (out.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        # Calculate and record training metrics for the current epoch
        train_acc = correct / total
        avg_train_loss = total_loss / total
        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_train_loss:.4f} — Train Acc: {train_acc:.4f}")
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_acc) # <--- Append train acc to history

        # === Validation Phase ===
        val_acc = 0.0
        avg_val_loss = 0.0
        if val_loader:
            model.eval() # Set model to evaluation mode
            val_correct, val_total = 0, 0
            val_loss_total = 0.0
            with torch.no_grad(): # No gradient calculation during validation
                for batch in val_loader:
                    v_images = batch[0].to(device)
                    v_labels = batch[1].to(device)
                    v_skin = batch[2].to(device) if model.include_skin_vec else None
                    v_triplet = batch[5].to(device) if model.use_triplet_embedding else None

                    out = model(v_images, v_skin, v_triplet)
                    assert out.dim() == 2 and out.size(1) == num_classes, f"Bad logits (val): {out.shape}"
                    loss = criterion(out, v_labels) # Use current criterion (CE or Focal)
                    val_loss_total += loss.item() * v_labels.size(0)
                    val_correct += (out.argmax(dim=1) == v_labels).sum().item()
                    val_total += v_labels.size(0)

            # Calculate and record validation metrics for the current epoch
            avg_val_loss = val_loss_total / val_total
            val_acc = val_correct / val_total
            val_loss_history.append(avg_val_loss)
            val_acc_history.append(val_acc) # <--- Append val acc to history
            print(f"Validation Accuracy: {val_acc:.4f} — Val Loss: {avg_val_loss:.4f}")
            scheduler.step(val_acc) # Step the LR scheduler based on validation accuracy

        # === Save best model weights ===
        # Save model if current validation accuracy is the best seen so far
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            early_stop_counter = 0 # Reset early stopping counter
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break # Stop training if no improvement for 'patience' epochs

        # === Save checkpoint ===
        # Periodically save a training checkpoint
        if checkpoint_path:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_accuracy": best_val_accuracy
            }, checkpoint_path)

    # Load the best model weights back into the model at the end of training
    model.load_state_dict(best_model_weights)
    # Save the best model weights to a dedicated path
    if best_weights_path:
        torch.save(best_model_weights, best_weights_path)
        print(f"✅ Saved best model weights to: {best_weights_path}")

    # Clear CUDA cache and run garbage collection to free memory
    torch.cuda.empty_cache()
    gc.collect()

    # Return the best model and the collected training history
    return model, {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_acc": train_acc_history,
        "val_acc": val_acc_history,
        "lrs": lrs_history
    }






