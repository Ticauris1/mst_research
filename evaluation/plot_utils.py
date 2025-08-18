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

'''
def plot_mst_distribution_by_class(y_true, mst_bins, class_names, save_path=None, show_percent=True):

    # Debugging check
    print(f"Classes detected: {np.unique(y_true)}")
    print(f"Total samples: {len(y_true)}")

    # Build DataFrame
    df = pd.DataFrame({
        'Class': pd.Categorical([class_names[int(y)] for y in y_true], categories=class_names),
        'MST': mst_bins
    })

    # Plot
    plt.figure(figsize=(16, 8))
    ax = sns.countplot(data=df, x='Class', hue='MST', palette='Spectral', order=class_names)

    plt.title("MST Distribution by Class")
    plt.xticks(rotation=45, ha='right')

    # Legend: compact, multiple columns
    plt.legend(title="MST Bin", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3, fontsize=8)

    # Add labels on bars
    totals = df.groupby('Class').size()
    for container in ax.containers:
        labels = []
        for bar in container:
            count = int(bar.get_height())
            if count == 0:
                labels.append("")
                continue
            class_name = bar.get_x() + bar.get_width()/2
            if show_percent:
                pct = 100 * count / totals[bar.get_xdata()[0]]
                labels.append(f"{count}\n({pct:.1f}%)")
            else:
                labels.append(f"{count}")
        ax.bar_label(container, labels=labels, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot to {save_path}")
    #plt.show()
'''
def plot_mst_distribution_by_class(y_true, mst_bins, class_names, save_path=None, show_percent=True):
    """
    Plots the distribution of MST bins for each class with corrected bar labeling.
    """
    # Build a DataFrame for easy plotting with Seaborn
    df = pd.DataFrame({
        'Class': pd.Categorical([class_names[int(y)] for y in y_true], categories=class_names, ordered=True),
        'MST': mst_bins
    })

    # Create the plot
    plt.figure(figsize=(16, 9))
    ax = sns.countplot(data=df, x='Class', hue='MST', palette='Spectral', order=class_names)

    plt.title("MST Distribution by Class", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Count", fontsize=12)
    plt.xlabel("Class", fontsize=12)
    plt.legend(title="MST Bin", bbox_to_anchor=(1.02, 1), loc='upper left')

    # --- CORRECTED ANNOTATION LOGIC ---
    # Calculate totals for each class to compute percentages
    totals = df.groupby('Class', observed=True).size()

    # Iterate through each container of bars (one container per MST bin)
    for i, container in enumerate(ax.containers):
        # The bars within a container are ordered by class
        # We need the class names to look up the total for each bar
        labels = []
        for j, bar in enumerate(container):
            count = int(bar.get_height())
            if count > 0:
                # Get the class name corresponding to the bar's position
                class_name_for_bar = class_names[j]
                total_for_class = totals[class_name_for_bar]
                
                if show_percent:
                    percent = 100 * count / total_for_class
                    labels.append(f"{percent:.1f}%")
                else:
                    labels.append(f"{count}")
            else:
                labels.append("") # Don't label zero-height bars
        
        # Apply the labels to the container
        ax.bar_label(container, labels=labels, fontsize=7, padding=3)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make room for legend

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved MST distribution plot to {save_path}")
    
    plt.close() # Close the plot to free memory


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

def plot_training_curves(history, save_dir="plots", model_name=""):
    """
    Plots training history (loss, accuracy, LR) and saves each
    metric to a separate file.

    Args:
        history (dict): A dictionary containing lists for 'train_loss', 'val_loss',
                        'train_acc', 'val_acc', and 'lrs'.
        save_dir (str): The directory where the plot images will be saved.
        model_name (str): An optional name to prefix the saved files.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Add a prefix to filenames if a model name is provided
    prefix = f"{model_name}_" if model_name else ""
    epochs = range(1, len(history['train_loss']) + 1)

    # --- 1. Loss Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], 'o-', label="Train Loss", color="blue")
    if 'val_loss' in history and history['val_loss']:
        plt.plot(epochs, history['val_loss'], 'o-', label="Val Loss", color="orange")
    plt.title(f"{model_name} - Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, f"{prefix}loss_curve.png")
    plt.savefig(loss_path, dpi=300)
    print(f"✅ Loss curve saved to {loss_path}")
    plt.close()

    # --- 2. Accuracy Plot ---
    if 'train_acc' in history and history['train_acc']:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history['train_acc'], 'o--', label="Train Accuracy", color="green")
        if 'val_acc' in history and history['val_acc']:
            plt.plot(epochs, history['val_acc'], 'o--', label="Val Accuracy", color="red")
        plt.title(f"{model_name} - Training & Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        acc_path = os.path.join(save_dir, f"{prefix}accuracy_curve.png")
        plt.savefig(acc_path, dpi=300)
        print(f"✅ Accuracy curve saved to {acc_path}")
        plt.close()

    # --- 3. Learning Rate Plot ---
    if 'lrs' in history and history['lrs']:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history['lrs'], 'o:', label="Learning Rate", color="purple")
        plt.title(f"{model_name} - Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        lr_path = os.path.join(save_dir, f"{prefix}lr_curve.png")
        plt.savefig(lr_path, dpi=300)
        print(f"✅ LR curve saved to {lr_path}")
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
            # New, clean way
            plot_training_curves(
                history=training_curves_data,
                save_dir=output_dir,
                model_name=model_name
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

