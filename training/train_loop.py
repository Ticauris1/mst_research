import os
import copy
import gc
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
from sklearn.utils.class_weight import compute_class_weight # type: ignore
from torch.cuda.amp import autocast as autocast_cuda # type: ignore
from training.lr_scheduler import HybridLRScheduler # type: ignore
from training.utils import GradualUnfreezer, PostWarmupLRScheduler, freeze_backbone, FocalLoss
from training.utils import setup_directories, freeze_backbone, compute_classwise_alpha # type: ignore
from training.mixup_utils import mixup_data, mixup_criterion # type: ignore
from evaluation.plot_utils import plot_training_curves 
# === Main Training Loop ===
def local_train(
    train_loader, model, device, num_epochs=10, lr=0.003,
    val_loader=None, save_model_path=None, model_name="model",
    fold=None, resume_path=None, alpha=0.2, mixup_enabled=True,
    warmup_epochs=5, num_classes=4, attention_type="none",
    log_lr_each_epoch=True, y_train=None
):
    # === Optimizer: Do NOT include model.base.parameters() here if using GradualUnfreezer
    optimizer = torch.optim.Adam([
        {'params': model.classifier.parameters(), 'lr': lr},
        {'params': model.skin_mlp.parameters(), 'lr': lr},
    ], weight_decay=1e-4)

    # Add attention module if present
    if hasattr(model, 'attn') and not isinstance(model.attn, nn.Identity):
        optimizer.add_param_group({'params': model.attn.parameters(), 'lr': lr})


    # === Class weighting ===
    class_weights_np = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    weights = torch.tensor(class_weights_np, dtype=torch.float, device=device)

    ce_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    focal_criterion = FocalLoss(alpha=weights)
    criterion = ce_criterion

    scheduler = HybridLRScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=num_epochs,
        plateau_patience=5,
        plateau_factor=0.5,
        min_lr=1e-6
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    checkpoint_path, best_weights_path, graph_dir, predictions_dir = setup_directories(
        base_path=save_model_path,
        model_name=model_name,
        fold=fold,
        attention_type=attention_type or "none"
    )

    best_val_accuracy = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    early_stop_counter = 0
    early_stop_patience = 5
    start_epoch = 0
    alpha_history = []
    train_loss_history = []
    val_loss_history = []

    # === Resume checkpoint ===
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"🔁 Resumed training from checkpoint at Epoch {start_epoch}")

    # === Initial freeze ===
    if start_epoch < warmup_epochs:
        freeze_backbone(model)

    for epoch in range(start_epoch, num_epochs):
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)

    # === Gradual unfreezing tools (defined once, not per epoch!)
    unfreezer = GradualUnfreezer(model, base_lr=lr)
    lr_riser = PostWarmupLRScheduler(optimizer, base_lr=lr, rise_epochs=3)

    for epoch in range(start_epoch, num_epochs):
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)

        # === Trigger unfreezing AFTER warmup
        if epoch == warmup_epochs:
            print(f"🧊 Warmup complete — beginning gradual unfreezing.")
        if epoch > warmup_epochs:
            unfreezer.step(optimizer, epoch)
            lr_riser.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


        # === Update focal loss ===
        if epoch >= warmup_epochs:
            mixup_enabled = False
            if val_loader:
                y_true, y_pred = [], []
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        v_images = batch[0].to(device)
                        v_labels = batch[1].to(device)
                        v_skin = batch[2].to(device) if model.include_skin_vec else None
                        v_triplet = batch[5].to(device) if model.use_triplet_embedding else None

                        out = model(v_images, v_skin, v_triplet)
                        y_pred.extend(out.argmax(dim=1).cpu().numpy())
                        y_true.extend(v_labels.cpu().numpy())

                dynamic_alpha = compute_classwise_alpha(y_true, y_pred, num_classes=num_classes).to(device)
                if alpha_history:
                    dynamic_alpha = 0.7 * alpha_history[-1] + 0.3 * dynamic_alpha
                alpha_history.append(dynamic_alpha)
                focal_criterion = FocalLoss(alpha=dynamic_alpha)
                criterion = focal_criterion

        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            skin_vecs = batch[2].to(device) if model.include_skin_vec else None
            triplet = batch[5].to(device) if model.use_triplet_embedding else None

            # === Mixup ===
            if mixup_enabled and not ((labels == 1).all() or (labels == 2).all()):
                mix_ratio = max(1.0 - epoch / num_epochs, 0.1)
                images, skin_vecs, y_a, y_b, lam = mixup_data(
                    images, labels, skin_vecs, alpha * mix_ratio, epoch, warmup_epochs
                )
                use_mixup = True
            else:
                y_a, y_b = labels, labels
                lam = 1.0
                use_mixup = False

            optimizer.zero_grad()

            if scaler:
                with autocast_cuda():
                    out = model(images, skin_vecs, triplet)
                    loss = mixup_criterion(out, y_a, y_b, lam, num_classes=num_classes) if use_mixup else criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(images, skin_vecs, triplet)
                loss = mixup_criterion(out, y_a, y_b, lam, num_classes=num_classes) if use_mixup else criterion(out, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"📚 Epoch {epoch+1}/{num_epochs} — Train Loss: {total_loss:.4f} — Train Acc: {train_acc:.4f}")
        train_loss_history.append(total_loss)

        # === Validation ===
        val_acc = 0.0
        avg_val_loss = 0.0
        if val_loader:
            model.eval()
            val_correct, val_total = 0, 0
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    v_images = batch[0].to(device)
                    v_labels = batch[1].to(device)
                    v_skin = batch[2].to(device) if model.include_skin_vec else None
                    v_triplet = batch[5].to(device) if model.use_triplet_embedding else None

                    out = model(v_images, v_skin, v_triplet)
                    loss = criterion(out, v_labels)
                    val_loss_total += loss.item()
                    val_correct += (out.argmax(dim=1) == v_labels).sum().item()
                    val_total += v_labels.size(0)

            avg_val_loss = val_loss_total / len(val_loader)
            val_acc = val_correct / val_total
            val_loss_history.append(avg_val_loss)
            print(f"🧪 Validation Accuracy: {val_acc:.4f} — Val Loss: {avg_val_loss:.4f}")
            scheduler.step(val_acc)

        # === Save best ===
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("⏹️ Early stopping triggered.")
                break

        # === Save checkpoint ===
        if checkpoint_path:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_accuracy": best_val_accuracy
            }, checkpoint_path)

        if log_lr_each_epoch:
            print(f"🔁 Current LR: {scheduler.get_lr()[0]:.6f}")

    if graph_dir and train_loss_history:
        training_plot_path = os.path.join(graph_dir, f"{model_name}_training_curves.png")
        plot_training_curves(train_loss_history, val_loss_history, scheduler.lr_history, save_path=training_plot_path)
        print(f"📉 Training curves saved to: {training_plot_path}")


    model.load_state_dict(best_model_weights)
    if best_weights_path:
        torch.save(best_model_weights, best_weights_path)
        print(f"✅ Saved best model weights to: {best_weights_path}")

    torch.cuda.empty_cache()
    gc.collect()
    return model