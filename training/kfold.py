from collections import Counter
import os
import gc
import time
import numpy as np # type: ignore
from config import  RESULTS_DIR, DEVICE
import torch # type: ignore
import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import StratifiedKFold # type: ignore
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  # type: ignore
from dataset.custom_dataset import CustomDataset  # Assuming you have a CustomDataset class defined
from dataset.transforms import standard_transform  # Assuming you have a standard_transform defined
from training.utils import setup_directories, get_model_with_attention
from evaluation.plot_utils import  plot_fairness, plot_probability_distributions
from utils.utils import compute_fairness_by_group 
from training.train_loop import local_train
from training.eval import evaluate_model

def kfold_cross_validation(
    X, y, label_encoder, model_names, attention_types, num_classes,
    transform, num_folds=5, num_epochs=10, batch_size=32,
    save_root=RESULTS_DIR, triplet_embedding_dict=None
):
    device = DEVICE
    seed = np.random.randint(0, 99999)
    print(f"🎲 Using random_state = {seed} for this k-fold trial")
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)



    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔁 Fold {fold + 1}/{num_folds}")

        X_tr = [X[i] for i in train_idx]
        y_tr_orig = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val_orig = [y[i] for i in val_idx]

        # 🔁 Re-map class labels to a contiguous range for each fold
        fold_classes = sorted(set(y_tr_orig + y_val_orig))
        fold_num_classes = len(fold_classes)
        class_mapping = {label: idx for idx, label in enumerate(fold_classes)}
        y_tr = [class_mapping[lbl] for lbl in y_tr_orig]
        y_val = [class_mapping[lbl] for lbl in y_val_orig]

        print(f"Fold {fold+1} Classes: {fold_classes} → Remapped to: {list(class_mapping.values())}")

        train_class_counts = Counter(y_tr)
        val_class_counts = Counter(y_val)
        print(f"📊 Fold {fold+1} Class Distribution (Train): {dict(train_class_counts)}")
        print(f"📊 Fold {fold+1} Class Distribution (Val): {dict(val_class_counts)}")

        y_tr = [class_mapping[lbl] for lbl in y_tr_orig]
        '''
        train_dataset = CustomDataset(
            X_tr,
            y_tr,
            transform=transform,
            include_skin_vec=True,
            triplet_embedding_dict=triplet_embedding_dict,  # ✅ Add this line
        )

        val_dataset = CustomDataset(
            X_val,
            y_val,
            transform=standard_transform,
            include_skin_vec=True,
            triplet_embedding_dict=triplet_embedding_dict,  # ✅ Add this line
        )'''


        CONFUSED_CLASSES = {1, 3, 4}        # East Asian, Latino_Hispanic, Middle Eastern
        UNDERREPRESENTED_CLASSES = {2, 5}  # Indian, Southeast Asian
        DOMINANT_CLASSES = {0, 6}          # Black, White

        MST_POLICY_MAP = {
            "MST_3": "specific_transform",
            "MST_4": "specific_transform",
            "MST_10": "specific_transform"
        }


        # Train dataset (epoch-based augmentation)
        train_dataset = CustomDataset(
            image_paths=X_tr,
            labels=y_tr,
            transform=None,  # ⛔️ Keep None to enable dynamic class+MST schedule
            include_skin_vec=True,
            triplet_embedding_dict=triplet_embedding_dict,
            confused_classes=CONFUSED_CLASSES,
            underrepresented_classes=UNDERREPRESENTED_CLASSES,
            dominant_classes=DOMINANT_CLASSES,
        )

        train_dataset.aug_schedule.mst_policy_map = MST_POLICY_MAP
        train_dataset.set_epoch(0)  # Set epoch at init or in training loop


        val_dataset = CustomDataset(
            X_val,
            y_val,
            transform=standard_transform,
            include_skin_vec=True,
            triplet_embedding_dict=triplet_embedding_dict,  # ✅ Add this line
        )



        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for attn_type in attention_types:
            for model_name in model_names:
                run_name = f"{model_name}_{attn_type}"
                print(f"\n🧪 Training: {run_name.upper()} — Fold {fold + 1}")

                checkpoint_path, best_weights_path, graph_dir, predictions_dir = setup_directories(
                    base_path=save_root,
                    model_name=model_name,
                    fold=fold + 1,
                    attention_type=attn_type
                )
                try:
                    start_time = time.time()  # ⏱️ Start timer

                    model = get_model_with_attention(
                        model_name=model_name,
                        num_classes=fold_num_classes,
                        attention_type=attn_type,
                        pretrained=True,
                        fold=fold + 1,
                        weights_root=save_root,
                        resume=True,
                        use_film_before=True,
                        use_film_in_cbam=False,
                        use_triplet_embedding=True,  # ✅ added
                        triplet_embedding_dim=512    # ✅ consistent with your model
                    ).to(device)

                    model = local_train(
                        train_loader=train_loader,
                        model=model,
                        device=device,
                        num_epochs=num_epochs,
                        lr=0.0001,
                        val_loader=val_loader,
                        save_model_path=save_root,
                        model_name=model_name,
                        fold=fold + 1,
                        resume_path=checkpoint_path,
                        alpha=0.2,
                        mixup_enabled=True,
                        warmup_epochs=3,
                        num_classes=fold_num_classes,
                        attention_type=attn_type,
                        y_train=y_tr
                    )


                      # Choose appropriate Grad-CAM layer based on model type
                    if model_name.lower() == "alexnet":
                        gradcam_layer = model.features[-1]  # or model.attn if CBAM goes after features
                    elif model_name.lower().startswith("resnet"):
                        gradcam_layer = model.layer4[-1]
                    elif model_name.lower().startswith("googlenet"):
                        gradcam_layer = model.inception5b if hasattr(model, "inception5b") else model.backbone[-1]
                    elif model_name.lower().startswith("efficientnet"):
                        gradcam_layer = model.backbone.conv_head  # ✅ Safe for timm-efficientnet_b0
                    elif model_name.lower().startswith("mobilenet"):
                        gradcam_layer = model.backbone[-1] # ✅ for MobileNetV2
                    else:
                        raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")

                    evaluate_model(
                        model=model,
                        test_loader=val_loader,
                        device=device,
                        label_encoder=label_encoder,
                        save_dir=predictions_dir,  # ✅ prediction CSV + report
                        model_name=f"{model_name}_{attn_type}_fold{fold+1}",
                        mst_bins=[sample[3] for sample in val_dataset],
                        skin_groups=[sample[4] for sample in val_dataset],
                        visualize_gradcam=True,
                        gradcam_layer=None,
                        graph_dir=graph_dir  # ✅ plots (confusion, tsne, etc.)
                    )
                    #print(classification_report(y_true, y_pred, digits=2))
                    elapsed = time.time() - start_time  # ⏱️ End timer
                    print(f"Training and evaluation time for {run_name.upper()} — Fold {fold+1}: {elapsed:.2f} seconds")

                except RuntimeError as e:
                    print(f"OOM Error — Skipping {run_name.upper()} (Fold {fold + 1}): {e}")
                finally:
                    if 'model' in locals():
                        del model
                    torch.cuda.empty_cache()
                    gc.collect()

def summarize_kfold_results(
    base_dir,
    model_names,
    attention_types,
    num_folds,
    label_encoder,
    save_root=None
):
    import time

    class_names = [str(c) for c in label_encoder.classes_]

    for model_name in model_names:
        for attention_type in attention_types:
            all_preds, all_trues = [], []
            all_probs, all_mst, all_groups = [], [], []
            timing_log = []

            for fold in range(1, num_folds + 1):
                tag = f"fold_{fold}_{model_name}_{attention_type}"
                pred_path = os.path.join(
                    base_dir, tag, "predictions",
                    f"{model_name}_{attention_type}_fold{fold}_predictions.csv"
                )

                start_time = time.time()

                if not os.path.exists(pred_path):
                    print(f"⚠️ Missing predictions for fold {fold}: {pred_path}")
                    continue

                df = pd.read_csv(pred_path)

                try:
                    all_trues.extend(df["True_Label"].map(lambda x: class_names.index(str(x))))
                    all_preds.extend(df["Predicted_Label"].map(lambda x: class_names.index(str(x))))
                except ValueError as e:
                    print(f"Label mapping error in fold {fold}: {e}")
                    continue

                all_probs.extend(df[[c for c in df.columns if c.startswith("Prob_")]].values.tolist())
                all_mst.extend(df["MST"].tolist())
                all_groups.extend(df["Skin_Group"].tolist())

                elapsed = time.time() - start_time
                timing_log.append([fold, model_name, attention_type, elapsed])

            if len(all_trues) == 0:
                print(f"No predictions loaded for {model_name} + {attention_type}")
                continue

            y_true = np.array(all_trues)
            y_pred = np.array(all_preds)
            y_probs = np.array(all_probs)

            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=class_names)

            print("\n📉 Confusion Matrix:")
            print(pd.DataFrame(cm, index=[f"True_{c}" for c in class_names],
                                  columns=[f"Pred_{c}" for c in class_names]))


            print(f"\n{model_name.upper()} + {attention_type.upper()} — Average Accuracy: {acc:.4f}")
            print("\n📋 Classification Report:")
            print(report)

            if save_root:
                save_path = os.path.join(save_root, f"{model_name}_{attention_type}_fold_avg")
                os.makedirs(save_path, exist_ok=True)

                # === Confusion Matrix ===
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names)
                plt.title(f"{model_name.upper()} + {attention_type.upper()} — Aggregated Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                cm_out = os.path.join(save_path, f"{model_name}_{attention_type}_avg_confusion.png")
                plt.savefig(cm_out, dpi=300)
                plt.close()
                #print(f"✅ Saved confusion matrix: {cm_out}")

                # === Fairness Plot ===
                fairness_df = compute_fairness_by_group(
                    y_true, y_probs, class_names, skin_groups=all_groups
                )
                fairness_out = os.path.join(save_path, f"{model_name}_{attention_type}_avg_fairness.png")
                plot_fairness(fairness_df, save_path=fairness_out)
                #print(f"✅ Fairness plot saved to: {fairness_out}")

                # === Probabilities Plot ===
                prob_plot_out = os.path.join(save_path, f"{model_name}_{attention_type}_avg_prob_dist.png")
                plot_probability_distributions(
                    y_true, y_probs, len(class_names), class_names, save_path=prob_plot_out
                )
                #print(f"📊 Probability distribution plot saved to: {prob_plot_out}")

                # === Report ===
                report_path = os.path.join(save_path, f"{model_name}_{attention_type}_avg_report.txt")
                with open(report_path, "w") as f:
                    f.write(f"Accuracy: {acc:.4f}\n\n")
                    f.write("Classification Report:\n")
                    f.write(report)
                print(f"📄 Saved report: {report_path}")

                # === Timing Log ===
                time_log_path = os.path.join(save_path, f"{model_name}_{attention_type}_timing_log.csv")
                df_timing = pd.DataFrame(
                    timing_log, columns=["Fold", "Model", "Attention", "Seconds"]
                )
                df_timing.to_csv(time_log_path, index=False)
                print(f"⏱️ Saved timing log: {time_log_path}")


def run_multiple_kfold_trials(
    X, y, label_encoder, model_names, attention_types, num_classes,
    transform, num_trials=3, num_folds=5, num_epochs=10, batch_size=32,
    save_root="results_dir", triplet_embedding_dict=None
):

    seeds = np.random.randint(0, 99999, size=num_trials)
    trial_tags = []

    for trial_num, seed in enumerate(seeds, 1):
        print(f"\n🚀 Starting Trial {trial_num}/{num_trials} with random seed {seed}")

        # Adjusted save_root to nest results by trial
        trial_root = f"{save_root}/trial_{trial_num}_seed_{seed}"
        trial_tags.append(trial_root)

        kfold_cross_validation(
            X=X,
            y=y,
            label_encoder=label_encoder,
            model_names=model_names,
            attention_types=attention_types,
            num_classes=num_classes,
            transform=transform,
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            save_root=trial_root,
            triplet_embedding_dict=triplet_embedding_dict  # ✅ added here
        )


    return trial_tags


def summarize_multiple_trials(
    trial_folders,
    model_names,
    attention_types,
    label_encoder,
    output_path="trial_summary"
):
    """
    Summarizes performance metrics across multiple k-fold trial folders.

    Args:
        trial_folders (list): Paths to trial root folders.
        model_names (list): List of model names used.
        attention_types (list): List of attention mechanisms used.
        label_encoder (LabelEncoder): Fitted encoder for class names.
        output_path (str): Path to save the aggregate summary.
    """
    os.makedirs(output_path, exist_ok=True)
    class_names = [str(c) for c in label_encoder.classes_]
    summary_records = []

    for model_name in model_names:
        for attention_type in attention_types:
            accs = []
            for trial in trial_folders:
                report_path = os.path.join(
                    trial, f"{model_name}_{attention_type}_fold_avg",
                    f"{model_name}_{attention_type}_avg_report.txt"
                )
                if os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.lower().startswith("accuracy"):
                                acc = float(line.strip().split(":")[-1])
                                accs.append(acc)
                                break
                else:
                    print(f"⚠️ Missing report: {report_path}")

            if accs:
                summary_records.append({
                    "Model": model_name,
                    "Attention": attention_type,
                    "Mean Accuracy": np.mean(accs),
                    "Std Accuracy": np.std(accs),
                    "Trials": len(accs)
                })

    # Save results to CSV
    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(output_path, "multiple_trial_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"✅ Saved trial summary: {summary_csv}")
