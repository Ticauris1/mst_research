import os
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn.functional as F # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore
import pandas as pd # type: ignore
from datetime import datetime
import cv2 # type: ignore
from evaluation.plot_utils import plot_evaluation_results, plot_tsne, plot_tsne_class_confusion, plot_mst_distribution_by_class
from evaluation.grad_cam import GradCAM
from utils.utils import extract_features
import traceback

def evaluate_model(
    model, test_loader, device, label_encoder=None,
    save_dir=None, model_name="model", mst_bins=None, skin_groups=None,
    gradcam_layer=None, visualize_gradcam=False,
    graph_dir=None,
    save_training_curves=False,
    training_curves_data=None,
    fold_classes=None # <--- ADDED: Original class labels present in this fold
):
    """
    Evaluates the model on the test set, generates reports, and plots various results.

    Args:
        model (nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test/validation set.
        device (torch.device): The device (CPU, CUDA, MPS) to run evaluation on.
        label_encoder (sklearn.preprocessing.LabelEncoder, optional): Encoder to map numeric labels back to original names. Defaults to None.
        save_dir (str, optional): Directory to save evaluation reports and predictions. Defaults to None.
        model_name (str, optional): Name of the model for file naming. Defaults to "model".
        mst_bins (list, optional): List of MST bin values for each sample. Defaults to None.
        skin_groups (list, optional): List of skin group strings for each sample. Defaults to None.
        gradcam_layer (torch.nn.Module, optional): The target layer for Grad-CAM visualization. Defaults to None.
        visualize_gradcam (bool, optional): Whether to generate Grad-CAM visualizations. Defaults to False.
        graph_dir (str, optional): Directory to save generated plots. Defaults to None.
        save_training_curves (bool, optional): Whether to save training curve plots. Defaults to False.
        training_curves_data (dict, optional): Dictionary containing lists of train/val loss, accuracy, and LRs. Defaults to None.
        fold_classes (list, optional): List of original (global) class integers present in the current fold. Defaults to None.
    """
    model.eval()
    y_true, y_pred, y_probs, all_skin_vecs = [], [], [], []
    all_mst_bins, all_skin_groups = [], []
    all_inputs, all_labels, raw_skin_vecs = [], [], []
    all_probs = []
    all_raw_metadata = [] # To store the raw metadata dictionaries for plotting

    with torch.no_grad():
        for batch in test_loader:
            # Dynamically unpack batch based on its length.
            # CustomDataset.__getitem__ now returns 7 items:
            # (img_tensor, label, skin_vec_tensor, mst_bin, skin_group_string, embedding_tensor, original_metadata_dict_for_plot)
            if len(batch) == 7: # Condition for the 7-item batch
                inputs, labels, skin_vecs, mst_batch, group_batch, triplet_vecs, raw_meta_batch = batch
            elif len(batch) == 6: # Fallback if CustomDataset doesn't return raw metadata (older CustomDataset)
                inputs, labels, skin_vecs, mst_batch, group_batch, triplet_vecs = batch
                # Create dummy metadata list if not provided by CustomDataset
                raw_meta_batch = [{"MST": m} for m in mst_batch] if mst_batch else [{"MST": -1}] * len(labels)
            elif len(batch) == 5: # With only 5 items, triplet_vecs and raw_meta_batch are missing
                inputs, labels, skin_vecs, mst_batch, group_batch = batch
                triplet_vecs = None
                raw_meta_batch = [{"MST": m} for m in mst_batch] if mst_batch else [{"MST": -1}] * len(labels)
            elif len(batch) == 3: # Fallback for datasets without MST/triplet/raw_meta (very old CustomDataset)
                inputs, labels, skin_vecs = batch
                mst_batch = [-1] * len(labels) # Use -1 for "unknown" MST bin
                group_batch = ["unknown"] * len(labels)
                triplet_vecs = None
                raw_meta_batch = [{"MST": -1}] * len(labels) # Create dummy metadata dicts
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            inputs = inputs.to(device)
            labels = labels.to(device)
            skin_vecs = skin_vecs.to(device)
            triplet_vecs = triplet_vecs.to(device) if triplet_vecs is not None else None

            outputs = model(inputs, skin_vecs, triplet_embedding=triplet_vecs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

            # Collect results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            all_skin_vecs.extend(skin_vecs.cpu().numpy())
            all_mst_bins.extend(mst_batch)
            all_skin_groups.extend(group_batch)
            all_probs.append(probs.cpu())
            all_raw_metadata.extend(raw_meta_batch) # Collect raw metadata for plotting functions

            if visualize_gradcam:
                all_inputs.append(inputs.cpu())
                all_labels.append(labels.cpu())
                raw_skin_vecs.append(skin_vecs.cpu())

    print("get to Grad-CAM init")
    # === Grad-CAM Misclassified Samples ===
    if visualize_gradcam and gradcam_layer:
        all_probs = torch.cat(all_probs)
        gradcam = GradCAM(model, gradcam_layer)
        gradcam_dir = os.path.join(graph_dir, "gradcam_misclassified")
        os.makedirs(gradcam_dir, exist_ok=True)

        all_inputs = torch.cat(all_inputs)
        all_labels = torch.cat(all_labels)
        all_skin_vecs_tensor = torch.cat(raw_skin_vecs)

        # === Step 1: Collect all misclassified samples with prediction confidence ===
        print("get to Grad-CAM step 1")
        misclassified = []
        for i in range(len(all_labels)):
            true_label = all_labels[i].item()
            pred_label = y_pred[i]
            if true_label != pred_label:
                confidence = all_probs[i][pred_label].item()
                misclassified.append((i, true_label, pred_label, confidence))

        # === Step 2: Sort by lowest confidence ===
        print("get to Grad-CAM step 2")
        top_k = 10
        misclassified = sorted(misclassified, key=lambda x: x[3])[:top_k]

        print(f"🖼️ Visualizing top {top_k} hard misclassified samples by Grad-CAM")

        # === Step 3: Visualize only top-k hard misclassified ===
        print("get to Grad-CAM step 3")
        for i, (idx, true_label, pred_label, confidence) in enumerate(misclassified):
            with torch.enable_grad():
                input_img = all_inputs[idx:idx+1].to(device).requires_grad_()
                skin_vec = all_skin_vecs_tensor[idx:idx+1].to(device)
                heatmap = gradcam.generate(input_img, skin_vec, target_class=pred_label)

                img_np = input_img.detach().squeeze().permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

                heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

                save_path = os.path.join(gradcam_dir, f"hard{i}_true{true_label}_pred{pred_label}_conf{confidence:.2f}.png")
                cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


    # === Metrics and Reports ===
    print("Make metric ")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Determine class names using label_encoder if available, and fold_classes for alignment
    # This is critical to fix the "list index out of range" error
    if label_encoder and fold_classes is not None:
        # Use label_encoder to get original string names for the classes present in this fold
        class_names = [label_encoder.inverse_transform([cls])[0] for cls in sorted(fold_classes)]
    else:
        # Fallback if label_encoder or fold_classes is not provided/valid
        class_names = [str(c) for c in sorted(np.unique(y_true))]


    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nAccuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", report)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = os.path.join(save_dir, f"{model_name}_report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n" + report + "\n")
        f.write("Confusion Matrix:\n" + np.array2string(cm))
    print(f"✅ Evaluation report saved to: {report_path}")

    pred_df = pd.DataFrame({
        "True_Label": [class_names[t] for t in y_true],
        "Predicted_Label": [class_names[p] for p in y_pred],
        "MST": all_mst_bins,
        "Skin_Group": all_skin_groups
    })
    for i, prob_row in enumerate(y_probs):
        for j, prob in enumerate(prob_row):
            pred_df.loc[i, f"Prob_{class_names[j]}"] = prob

    pred_csv_path = os.path.join(save_dir, f"{model_name}_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"✅ Predictions CSV saved to: {pred_csv_path}")

    print(f"📊 Calling plot_evaluation_results for {model_name} — saving to {graph_dir}")
    print("Plot results init")
    plot_evaluation_results(
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        confusion=cm,
        class_names=class_names,
        skin_vecs=all_raw_metadata, # Pass raw metadata for correct heatmap plotting
        mst_bins=all_mst_bins,
        skin_groups=all_skin_groups,
        output_dir=graph_dir,
        save_training_curves=save_training_curves,
        training_curves_data=training_curves_data
    )
    print(f"✅ Evaluation graphs saved to: {graph_dir}")

    try:
        features, labels_for_tsne = extract_features(model, test_loader, device)
        if features.ndim > 2:
            print(f"🔁 Flattening features from shape {features.shape} for t-SNE...")
            features = features.reshape(features.shape[0], -1)

        print(f"🔍 Feature shape for t-SNE: {features.shape}")
        n_samples = features.shape[0]
        safe_perplexity = min(30, max(5, n_samples // 3))

        if n_samples <= 10:
            print("⚠️ t-SNE skipped: too few samples for reliable visualization.")
        else:
            tsne_path = os.path.join(graph_dir, f"{model_name}_tsne.png")
            confusion_tsne_path = os.path.join(graph_dir, f"{model_name}_tsne_confusion_class2.png")

            plot_tsne(features, labels_for_tsne, class_names, perplexity=safe_perplexity, save_path=tsne_path)
            print(f"✅ t-SNE plot saved to: {tsne_path}")

            plot_tsne_class_confusion(
                features, y_true, y_pred,
                class_names=class_names,
                target_class=1,
                perplexity=safe_perplexity,
                save_path=confusion_tsne_path
            )
            print(f"✅ Confusion-focused t-SNE saved to: {confusion_tsne_path}")
    except Exception as e:
        print(f"⚠️ t-SNE failed: {e}")
        traceback.print_exc()


    if visualize_gradcam and gradcam_layer:
        print(f"✅ Grad-CAM misclassified overlays saved to: {os.path.join(graph_dir, 'gradcam_misclassified')}")

    return acc, report, cm
