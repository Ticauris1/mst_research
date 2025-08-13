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

def plot_mst_distribution_by_class(y_true, mst_bins, class_names, save_path=None, show_percent=False):

    # --- Debugging Prints --- (Keep these for now, they are useful)
    print(f"\n--- Debugging plot_mst_distribution_by_class ---")
    print(f"y_true (first 20 elements, if available): {y_true[:20]}")
    print(f"Full y_true length: {len(y_true)}")
    print(f"class_names: {class_names}")
    print(f"len(class_names): {len(class_names)}")
    if len(y_true) > 0:
        max_y_true = np.max(y_true)
        if max_y_true >= len(class_names):
            print(f"🚨 Potential Mismatch: max(y_true) ({max_y_true}) is >= len(class_names) ({len(class_names)})")
    else:
        print("y_true is empty.")
    print(f"---------------------------------------------")
    # --- End Debugging Prints ---

    # Ensure y_true is a list of integers, not a NumPy array if that's causing subtle issues.
    # Also, ensure class_names is treated as categories for plotting consistency.
    y_true_list = [int(y) for y in y_true] # Ensure elements are standard Python ints
    
    # Use pandas Categorical type for 'Class' column with explicit categories from class_names
    # This ensures seaborn understands the order and labels correctly.
    df = pd.DataFrame({
        'Class': pd.Categorical([class_names[y] for y in y_true_list], categories=class_names),
        'MST': mst_bins
    })


    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df, x='Class', hue='MST', palette='Spectral', order=class_names) # Explicitly set order
    plt.title("MST Distribution by Class")
    plt.xticks(rotation=45)
    plt.legend(title="MST Bin", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # === Annotate counts or percentages on bars ===
    # Get counts for each class directly from the DataFrame
    totals_by_class = df.groupby('Class').size().to_dict()


    for p in ax.patches:
        height = p.get_height()
        if height == 0:
            continue
        # Get x-coordinate for annotation
        x = p.get_x() + p.get_width() / 2
        y = height

        # Ensure class_idx is safely obtained
        # Use try-except for robust annotation in case `get_xticklabels` fails or is out of sync.
        try:
            class_idx = int(p.get_x() // p.get_width()) # This needs to match the overall plot structure
            class_name = ax.get_xticklabels()[class_idx].get_text() # This gets the label from the tick
        except IndexError:
            # Fallback if ax.get_xticklabels() is out of sync or class_idx is invalid
            class_name = "Unknown_Class" # Or log a warning
            #print(f"⚠️ Warning: Could not determine class name for annotation patch at x={p.get_x()}")
            continue # Skip annotation for this patch


        total = totals_by_class.get(class_name, 1) # Use the totals derived directly from df


        label = f"{height}"
        if show_percent:
            pct = 100 * height / total
            label = f"{height}\n({pct:.1f}%)"

        ax.annotate(label, (x, y), ha='center', va='bottom', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    plt.plot(train_loss, label="Train Loss", linewidth=2)
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

def plot_training_curves(
    train_loss, val_loss=None,
    train_acc=None, val_acc=None,
    lrs=None, save_path="training_plot.png"
):
    epochs = list(range(1, len(train_loss) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # === Plot Loss ===
    ax1.plot(epochs, train_loss, label="Train Loss", color="blue", linewidth=2)
    if val_loss is not None:
        ax1.plot(epochs, val_loss, label="Val Loss", color="orange", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # === Plot Accuracy ===
    if train_acc is not None or val_acc is not None:
        ax2 = ax1.twinx()
        if train_acc is not None:
            ax2.plot(epochs, train_acc, label="Train Acc", color="green", linestyle="--", linewidth=2)
        if val_acc is not None:
            ax2.plot(epochs, val_acc, label="Val Acc", color="red", linestyle="--", linewidth=2)
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="upper right")

    # === Plot Learning Rate ===
    if lrs is not None:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))  # Shift right for LR
        ax3.plot(epochs, lrs, label="LR", color="purple", linestyle=":", linewidth=2)
        ax3.set_ylabel("Learning Rate")
        ax3.legend(loc="lower right")

    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training plot saved to: {save_path}")
    plt.close()

def plot_evaluation_results(
    model_name,
    y_true,
    y_pred,
    y_probs,
    confusion,
    class_names,
    skin_vecs,
    mst_bins,
    skin_groups,
    output_dir,
    features=None,  # 🆕 Optional t-SNE input
    save_training_curves=False,
    training_curves_data=None,  # Dict with 'train_loss', 'val_loss', etc.
    target_confused_class=2      # 🆕 Highlight t-SNE confusion for this class
):
    print(f"\n📍 Entered plot_evaluation_results() for {model_name}")
    os.makedirs(output_dir, exist_ok=True)

    # === Confusion Matrix ===
    cm_path = os.path.join(output_dir, f'{model_name}_confusion.png')
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to: {cm_path}")

    # === Probability Distribution ===
    prob_plot_path = os.path.join(output_dir, f'{model_name}_prob_dist.png')
    plot_probability_distributions(y_true, y_probs, len(class_names), class_names, save_path=prob_plot_path)

    # === ROC Curve ===
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(y_probs.shape[1]):
            if y_true_bin[:, i].sum() == 0:
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
            plt.close()
            print(f"✅ ROC curve saved to: {roc_path}")
    except Exception as e:
        print(f"❌ ROC plot failed: {e}")

    # === Fairness Plot ===
    try:
        fairness_df = compute_fairness_by_group(y_true, y_probs, class_names, skin_groups=skin_groups)
        fairness_plot_path = os.path.join(output_dir, f"{model_name}_fairness.png")
        plot_fairness(fairness_df, save_path=fairness_plot_path)
    except Exception as e:
        print(f"❌ Fairness plot failed: {e}")

    # === Class-SkinGroup Heatmap 🆕 ===
    try:
        heatmap_path = os.path.join(output_dir, f"{model_name}_class_skin_group_heatmap.png")
        plot_class_skin_group_counts(y_true, skin_vecs, save_path=heatmap_path)
    except Exception as e:
        print(f"❌ Class-SkinGroup heatmap failed: {e}")

    # === MST Distribution 🆕 ===
    try:
        mst_dist_path = os.path.join(output_dir, f"{model_name}_mst_distribution.png")
        plot_mst_distribution_by_class(y_true, mst_bins, class_names, save_path=mst_dist_path)
    except Exception as e:
        print(f"❌ MST distribution plot failed: {e}")

    # === t-SNE Plots 🆕 ===
    if features is not None:
        try:
            tsne_combined_path = os.path.join(output_dir, f"{model_name}_tsne_combined.png")
            plot_tsne_combined(features, y_true, y_pred, class_names,
                               target_confused_class=target_confused_class, save_path=tsne_combined_path)

            tsne_focus_path = os.path.join(output_dir, f"{model_name}_tsne_confusion_class_{target_confused_class}.png")
            plot_tsne_class_confusion(features, y_true, y_pred, class_names,
                                      target_class=target_confused_class, save_path=tsne_focus_path)
        except Exception as e:
            print(f"❌ t-SNE plotting failed: {e}")

    # === Training Curves (optional) 🆕 ===
    if save_training_curves and training_curves_data:
        try:
            plot_training_curves(
                train_loss=training_curves_data.get("train_loss"),
                val_loss=training_curves_data.get("val_loss"),
                train_acc=training_curves_data.get("train_acc"),
                val_acc=training_curves_data.get("val_acc"),
                lrs=training_curves_data.get("lrs"),
                save_path=os.path.join(output_dir, f"{model_name}_training_curves.png")
            )
        except Exception as e:
            print(f"❌ Training curves plot failed: {e}")

def get_gradcam_layer(model, model_name):
    if hasattr(model, "get_gradcam_target_layer"):
        return model.get_gradcam_target_layer()

    if model_name.startswith("efficientnet") or model_name.startswith("densenet"):
        return model.base.blocks[-1]
    elif model_name.startswith("resnet"):
        return model.base.layer4 if hasattr(model.base, 'layer4') else list(model.base.children())[-1]
    elif model_name == "mobilenet_v2":
        return model.base.features[-1]
    elif model_name.startswith("vgg"):
        return model.base.features[-1]
    else:
        raise ValueError(f"No Grad-CAM layer logic defined for: {model_name}")


'''
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

def plot_mst_distribution_by_class(y_true, mst_bins, class_names, save_path=None, show_percent=False):

    # --- Debugging Prints ---
    print(f"\n--- Debugging plot_mst_distribution_by_class ---")
    print(f"y_true (first 20 elements, if available): {y_true[:20]}") # Print a slice to avoid overwhelming output
    print(f"Full y_true length: {len(y_true)}")
    print(f"class_names: {class_names}")
    print(f"len(class_names): {len(class_names)}")
    if len(y_true) > 0:
        max_y_true = np.max(y_true)
        print(f"Max value in y_true: {max_y_true}")
        print(f"Expected max index for class_names (len(class_names) - 1): {len(class_names) - 1}")
        if max_y_true >= len(class_names):
            print(f"🚨 Potential Mismatch: max(y_true) ({max_y_true}) is >= len(class_names) ({len(class_names)})")
    else:
        print("y_true is empty.")
    print(f"---------------------------------------------")
    # --- End Debugging Prints ---

    df = pd.DataFrame({
        'Class': [class_names[y] for y in y_true], # This is the line that will error
        'MST': mst_bins
    })

    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df, x='Class', hue='MST', palette='Spectral')
    plt.title("MST Distribution by Class")
    plt.xticks(rotation=45)
    plt.legend(title="MST Bin", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # === Annotate counts or percentages on bars ===
    totals_by_class = df['Class'].value_counts().to_dict()

    for p in ax.patches:
        height = p.get_height()
        if height == 0:
            continue
        x = p.get_x() + p.get_width() / 2
        y = height

        # Get class label from x position
        class_label = p.get_x() + p.get_width() / 2
        class_idx = int(p.get_x() // p.get_width())
        class_name = ax.get_xticklabels()[class_idx].get_text()
        total = totals_by_class.get(class_name, 1)

        label = f"{height}"
        if show_percent:
            pct = 100 * height / total
            label = f"{height}\n({pct:.1f}%)"

        ax.annotate(label, (x, y), ha='center', va='bottom', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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

def plot_training_curves(
    train_loss, val_loss=None,
    train_acc=None, val_acc=None,
    lrs=None, save_path="training_plot.png"
):
    epochs = list(range(1, len(train_loss) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # === Plot Loss ===
    ax1.plot(epochs, train_loss, label="Train Loss", color="blue", linewidth=2)
    if val_loss is not None:
        ax1.plot(epochs, val_loss, label="Val Loss", color="orange", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # === Plot Accuracy ===
    if train_acc is not None or val_acc is not None:
        ax2 = ax1.twinx()
        if train_acc is not None:
            ax2.plot(epochs, train_acc, label="Train Acc", color="green", linestyle="--", linewidth=2)
        if val_acc is not None:
            ax2.plot(epochs, val_acc, label="Val Acc", color="red", linestyle="--", linewidth=2)
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="upper right")

    # === Plot Learning Rate ===
    if lrs is not None:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))  # Shift right for LR
        ax3.plot(epochs, lrs, label="LR", color="purple", linestyle=":", linewidth=2)
        ax3.set_ylabel("Learning Rate")
        ax3.legend(loc="lower right")

    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training plot saved to: {save_path}")
    plt.close()

def plot_evaluation_results(
    model_name,
    y_true,
    y_pred,
    y_probs,
    confusion,
    class_names,
    skin_vecs,
    mst_bins,
    skin_groups,
    output_dir,
    features=None,  # 🆕 Optional t-SNE input
    save_training_curves=False,
    training_curves_data=None,  # Dict with 'train_loss', 'val_loss', etc.
    target_confused_class=2      # 🆕 Highlight t-SNE confusion for this class
):
    print(f"\n📍 Entered plot_evaluation_results() for {model_name}")
    os.makedirs(output_dir, exist_ok=True)

    # === Confusion Matrix ===
    cm_path = os.path.join(output_dir, f'{model_name}_confusion.png')
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to: {cm_path}")

    # === Probability Distribution ===
    prob_plot_path = os.path.join(output_dir, f'{model_name}_prob_dist.png')
    plot_probability_distributions(y_true, y_probs, len(class_names), class_names, save_path=prob_plot_path)

    # === ROC Curve ===
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(y_probs.shape[1]):
            if y_true_bin[:, i].sum() == 0:
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
            plt.close()
            print(f"✅ ROC curve saved to: {roc_path}")
    except Exception as e:
        print(f"❌ ROC plot failed: {e}")

    # === Fairness Plot ===
    try:
        fairness_df = compute_fairness_by_group(y_true, y_probs, class_names, skin_groups=skin_groups)
        fairness_plot_path = os.path.join(output_dir, f"{model_name}_fairness.png")
        plot_fairness(fairness_df, save_path=fairness_plot_path)
    except Exception as e:
        print(f"❌ Fairness plot failed: {e}")

    # === Class-SkinGroup Heatmap 🆕 ===
    try:
        heatmap_path = os.path.join(output_dir, f"{model_name}_class_skin_group_heatmap.png")
        plot_class_skin_group_counts(y_true, skin_vecs, save_path=heatmap_path)
    except Exception as e:
        print(f"❌ Class-SkinGroup heatmap failed: {e}")

    # === MST Distribution 🆕 ===
    try:
        mst_dist_path = os.path.join(output_dir, f"{model_name}_mst_distribution.png")
        plot_mst_distribution_by_class(y_true, mst_bins, class_names, save_path=mst_dist_path)
    except Exception as e:
        print(f"❌ MST distribution plot failed: {e}")

    # === t-SNE Plots 🆕 ===
    if features is not None:
        try:
            tsne_combined_path = os.path.join(output_dir, f"{model_name}_tsne_combined.png")
            plot_tsne_combined(features, y_true, y_pred, class_names,
                               target_class=target_confused_class, save_path=tsne_combined_path)

            tsne_focus_path = os.path.join(output_dir, f"{model_name}_tsne_confusion_class_{target_confused_class}.png")
            plot_tsne_class_confusion(features, y_true, y_pred, class_names,
                                      target_class=target_confused_class, save_path=tsne_focus_path)
        except Exception as e:
            print(f"❌ t-SNE plotting failed: {e}")

    # === Training Curves (optional) 🆕 ===
    if save_training_curves and training_curves_data:
        try:
            plot_training_curves(
                train_loss=training_curves_data.get("train_loss"),
                val_loss=training_curves_data.get("val_loss"),
                train_acc=training_curves_data.get("train_acc"),
                val_acc=training_curves_data.get("val_acc"),
                lrs=training_curves_data.get("lrs"),
                save_path=os.path.join(output_dir, f"{model_name}_training_curves.png")
            )
        except Exception as e:
            print(f"❌ Training curves plot failed: {e}")

def get_gradcam_layer(model, model_name):
    if hasattr(model, "get_gradcam_target_layer"):
        return model.get_gradcam_target_layer()

    if model_name.startswith("efficientnet") or model_name.startswith("densenet"):
        return model.base.blocks[-1]
    elif model_name.startswith("resnet"):
        return model.base.layer4 if hasattr(model.base, 'layer4') else list(model.base.children())[-1]
    elif model_name == "mobilenet_v2":
        return model.base.features[-1]
    elif model_name.startswith("vgg"):
        return model.base.features[-1]
    else:
        raise ValueError(f"No Grad-CAM layer logic defined for: {model_name}")
'''