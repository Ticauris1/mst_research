from collections import Counter
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
from utils.utils import bin_mst_to_skin_group, compute_fairness_by_group
import cv2 # type: ignore
from sklearn.manifold import TSNE # type: ignore
import numpy as np # type: ignore
import os
from sklearn.metrics import (auc, roc_curve) # type: ignore 
from sklearn.preprocessing import label_binarize  # type: ignore 

def plot_class_skin_group_counts(y, z, group_fn=bin_mst_to_skin_group, save_path=None):

    combo_counts = Counter()
    for label, skin_vec in zip(y, z):
        if not isinstance(skin_vec, dict) or "MST" not in skin_vec:
            continue
        group = group_fn(skin_vec["MST"])
        combo_counts[(label, group)] += 1

    df = pd.DataFrame([
        {"Class": k[0], "Skin_Group": k[1], "Count": v}
        for k, v in combo_counts.items()
    ])

    pivot = df.pivot(index="Class", columns="Skin_Group", values="Count").fillna(0)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Samples per (Class, Skin Group)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📊 Saved class-group heatmap to {save_path}")
    else:
        plt.show()

def plot_mst_distribution_by_class(y_true, mst_bins, class_names, save_path=None):
    df = pd.DataFrame({
        'Class': [class_names[y] for y in y_true],
        'MST': mst_bins
    })

    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Class', hue='MST', palette='Spectral')
    plt.title("MST Distribution by Class")
    plt.xticks(rotation=45)
    plt.legend(title="MST Bin", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ MST distribution plot saved to: {save_path}")
    plt.close()

def plot_combo_histogram(combo_map, min_per_combo):

    counts = [len(v) for v in combo_map.values()]
    plt.hist(counts, bins=20, color="skyblue")
    plt.axvline(min_per_combo, color='red', linestyle='--', label=f'min_per_combo={min_per_combo}')
    plt.xlabel("Samples per (class, skin_group) combo")
    plt.ylabel("Count of combos")
    plt.title("Distribution of Combo Sizes")
    plt.legend()
    plt.savefig("combo_histogram.png")
    plt.close()

def overlay_heatmap(heatmap, image_path, alpha=0.4):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(heatmap_colored, alpha, image, 1 - alpha, 0)
    return overlayed

def plot_loss_curve(train_loss, val_loss, save_path=None, title="Training and Validation Loss"):
    """
    Plots training and validation loss curves.

    Args:
        train_loss (list): List of training losses per epoch.
        val_loss (list): List of validation losses per epoch.
        save_path (str, optional): If provided, saves the plot to this path.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label="Training Loss", linewidth=2)
    plt.plot(val_loss, label="Validation Loss", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"📈 Saved loss plot to {save_path}")
    else:
        plt.show()

def plot_tsne(features, labels, class_names, perplexity=30, save_path=None, show=False):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced[:, 0], y=reduced[:, 1],
        hue=[class_names[i] for i in labels],
        palette="tab10", alpha=0.6
    )
    plt.title("t-SNE of Feature Embeddings")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()

def plot_tsne_class_confusion(features, y_true, y_pred, class_names, target_class=2, perplexity=30, save_path=None):
    """
    Plot t-SNE with class confusion emphasis.

    Args:
        features (np.ndarray): Feature matrix of shape [N x D].
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Model predicted labels.
        class_names (list): List of class names (index-aligned).
        target_class (int): Class index to highlight confusion for.
        perplexity (int): t-SNE perplexity (should be < n_samples).
        save_path (str, optional): Path to save the plot image.
    """
    # Safety check for perplexity vs sample size
    n_samples = len(features)
    perplexity = min(perplexity, max(2, (n_samples - 1) // 3))

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(features)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct_mask = (y_true == target_class) & (y_pred == target_class)
    misclass_mask = (y_true == target_class) & (y_pred != target_class)

    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    # Plot other background classes
    for cls in np.unique(y_true):
        if cls == target_class:
            continue
        idx = y_true == cls
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f"Other: {class_names[cls]}", alpha=0.2, s=20)

    # Plot correct predictions
    plt.scatter(reduced[correct_mask, 0], reduced[correct_mask, 1],
                label=f"Correct {class_names[target_class]}", c='green', s=40)

    # Plot misclassifications
    plt.scatter(reduced[misclass_mask, 0], reduced[misclass_mask, 1],
                label=f"Misclassified {class_names[target_class]}", c='red', s=40, marker='x')

    plt.title(f"t-SNE — Confusion Analysis for Class {target_class} ({class_names[target_class]})")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

def plot_tsne_combined(features, y_true, y_pred, class_names, target_class=2, perplexity=30, save_path=None):
    sns.set(style="whitegrid")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sns.scatterplot(
        x=reduced[:, 0], y=reduced[:, 1],
        hue=[class_names[i] for i in y_pred],
        palette="tab10", alpha=0.6,
        ax=axes[0]
    )
    axes[0].set_title("t-SNE — All Predictions (by predicted class)")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    correct_mask = (y_true == target_class) & (y_pred == target_class)
    misclass_mask = (y_true == target_class) & (y_pred != target_class)

    for cls in np.unique(y_true):
        if cls == target_class:
            continue
        idx = y_true == cls
        axes[1].scatter(reduced[idx, 0], reduced[idx, 1], label=f"Other: {class_names[cls]}", alpha=0.15, s=20)

    axes[1].scatter(reduced[correct_mask, 0], reduced[correct_mask, 1], label=f"Correct {class_names[target_class]}", c='green', s=40)
    axes[1].scatter(reduced[misclass_mask, 0], reduced[misclass_mask, 1], label=f"Misclassified {class_names[target_class]}", c='red', s=40, marker='x')

    axes[1].set_title(f"t-SNE — Confusion Focus: {class_names[target_class]}")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)

def plot_fairness(fairness_df, save_path=None):
    def group_sort_key(g):
        try:
            return int(g.split("_")[-1])
        except:
            return 999
    fairness_df["sort_key"] = fairness_df["Skin Group"].apply(group_sort_key)
    fairness_df = fairness_df.sort_values("sort_key")
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    cmap = plt.colormaps.get_cmap('tab10').resampled(len(fairness_df))
    for i, row in enumerate(fairness_df.itertuples()):
        x, y, group = row.Accuracy, row.F1, row._1  # or row._2 depending on column order
        plt.scatter(x, y, label=group, color=cmap(i))
        plt.text(x + 0.01, y, group, fontsize=9, color=cmap(i))
    plt.title("Accuracy vs F1 Score by Skin Group")
    plt.xlabel("Accuracy")
    plt.ylabel("F1 Score")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Fairness plot saved to: {save_path}")
        csv_path = os.path.splitext(save_path)[0] + ".csv"
        fairness_df.drop(columns=["sort_key"], errors='ignore').to_csv(csv_path, index=False)
        print(f"📄 Fairness data saved to: {csv_path}")
    plt.close()

def plot_probability_distributions(y_true, y_probs, num_classes, class_names=None, save_path=None):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    if class_names is not None:
        class_names = list(class_names)

    fig, axes = plt.subplots(num_classes, 1, figsize=(9, 3 * num_classes), sharex=True)
    colors = plt.cm.tab10.colors
    for class_idx in range(num_classes):
        class_probs = y_probs[y_true == class_idx, class_idx]
        color = colors[class_idx % len(colors)]
        label = class_names[class_idx] if class_names and len(class_names) > class_idx else f"Class {class_idx}"
        axes[class_idx].hist(class_probs, bins=20, range=(0, 1), alpha=0.8, color=color)
        axes[class_idx].set_title(f"True Class: {label}")
        axes[class_idx].set_ylabel("Frequency")
    axes[-1].set_xlabel("Predicted Probability")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Probability distribution plot saved to: {save_path}")
    plt.close()

def plot_training_curves(train_loss, val_loss=None, lrs=None, save_path="training_plot.png"):
    epochs = list(range(1, len(train_loss) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(epochs, train_loss, label="Train Loss", color="blue", linewidth=2)
    if val_loss:
        ax1.plot(epochs, val_loss, label="Val Loss", color="orange", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    if lrs:
        ax2 = ax1.twinx()
        ax2.plot(epochs, lrs, label="LR", color="green", linestyle="--")
        ax2.set_ylabel("Learning Rate")
        ax2.legend(loc="upper left")

    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig(save_path)

def plot_evaluation_results(
    model_name, y_true, y_pred, y_probs, confusion,
    class_names, skin_vecs, mst_bins, skin_groups, output_dir
):
    print(f"\n📍 Entered plot_evaluation_results() for {model_name}")
    os.makedirs(output_dir, exist_ok=True)

    # === Confusion Matrix ===
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")
    cm_path = os.path.join(output_dir, f'{model_name}_confusion.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to: {cm_path}")
    plt.close()

    # === Probability Distribution ===
    prob_plot_path = os.path.join(output_dir, f'{model_name}_prob_dist.png')
    plot_probability_distributions(y_true, y_probs, len(class_names), class_names, save_path=prob_plot_path)

    # === ROC Curve with Robust Class Handling ===
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        num_probs = y_probs.shape[1]

        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(num_probs):
            if i >= y_true_bin.shape[1]:
                print(f"⚠️ Skipping ROC for class {i} — missing from y_true.")
                continue
            if y_true_bin[:, i].sum() == 0:
                print(f"⚠️ Skipping ROC for class {i} — no true positives.")
                continue
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        if roc_auc:
            plt.figure(figsize=(10, 7))
            for i in roc_auc:
                plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} ROC Curve')
            plt.legend(loc="lower right")
            roc_path = os.path.join(output_dir, f'{model_name}_roc.png')
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            print(f"✅ ROC curve saved to: {roc_path}")
            plt.close()
        else:
            print("⚠️ ROC curve skipped — no valid class AUCs to plot.")
    except Exception as e:
        print(f"❌ ROC plot failed: {e}")

    # === Per-sample CSV (Probabilities + Metadata) ===
    try:
        output_data = []
        for i in range(len(y_true)):
            row = {
                "True_Label": class_names[y_true[i]],
                "Predicted_Label": class_names[y_pred[i]],
                "MST": mst_bins[i],
                "Skin_Group": skin_groups[i]
            }
            for j in range(min(len(class_names), y_probs.shape[1])):
                row[f"Prob_{class_names[j]}"] = y_probs[i][j]
            output_data.append(row)
        df = pd.DataFrame(output_data)
    except Exception as e:
        print(f"❌ Failed to build prediction DataFrame: {e}")
        df = pd.DataFrame()

    # === Fairness Plot ===
    try:
        fairness_df = compute_fairness_by_group(y_true, y_probs, class_names, skin_groups=skin_groups)
        fairness_plot_path = os.path.join(output_dir, f"{model_name}_fairness.png")
        plot_fairness(fairness_df, save_path=fairness_plot_path)
        print(f"✅ Saved fairness plot to: {fairness_plot_path}")
    except Exception as e:
        print(f"❌ Fairness plot failed: {e}")

def plot_training_curves(train_loss, val_loss=None, lrs=None, save_path="training_plot.png"):
    epochs = list(range(1, len(train_loss) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(epochs, train_loss, label="Train Loss", color="blue", linewidth=2)
    if val_loss:
        ax1.plot(epochs, val_loss, label="Val Loss", color="orange", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    if lrs:
        ax2 = ax1.twinx()
        ax2.plot(epochs, lrs, label="LR", color="green", linestyle="--")
        ax2.set_ylabel("Learning Rate")
        ax2.legend(loc="upper left")

    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig(save_path)