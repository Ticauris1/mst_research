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

def evaluate_model(
    model, test_loader, device, label_encoder=None,
    save_dir=None, model_name="model", mst_bins=None, skin_groups=None,
    gradcam_layer=None, visualize_gradcam=False,
    graph_dir=None
):
    model.eval()
    y_true, y_pred, y_probs, all_skin_vecs = [], [], [], []
    all_mst_bins, all_skin_groups = [], []
    all_inputs, all_labels, raw_skin_vecs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 6:
                inputs, labels, skin_vecs, mst_batch, group_batch, triplet_vecs = batch
            elif len(batch) == 5:
                inputs, labels, skin_vecs, mst_batch, group_batch = batch
                triplet_vecs = None
            elif len(batch) == 3:
                inputs, labels, skin_vecs = batch
                mst_batch = ["unknown"] * len(labels)
                group_batch = ["unknown"] * len(labels)
                triplet_vecs = None
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            inputs = inputs.to(device)
            labels = labels.to(device)
            skin_vecs = skin_vecs.to(device)
            triplet_vecs = triplet_vecs.to(device) if triplet_vecs is not None else None

            outputs = model(inputs, skin_vecs, triplet_embedding=triplet_vecs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            all_skin_vecs.extend(skin_vecs.cpu().numpy())
            all_mst_bins.extend(mst_batch)
            all_skin_groups.extend(group_batch)

            if visualize_gradcam:
                all_inputs.append(inputs.cpu())
                all_labels.append(labels.cpu())
                raw_skin_vecs.append(skin_vecs.cpu())

    # === Grad-CAM Misclassified Samples ===
    if visualize_gradcam and gradcam_layer:
        gradcam = GradCAM(model, gradcam_layer)
        gradcam_dir = os.path.join(graph_dir, "gradcam_misclassified")
        os.makedirs(gradcam_dir, exist_ok=True)

        all_inputs = torch.cat(all_inputs)
        all_labels = torch.cat(all_labels)
        all_skin_vecs_tensor = torch.cat(raw_skin_vecs)

        for i in range(len(all_labels)):
            true_label = all_labels[i].item()
            pred_label = y_pred[i]
            if true_label != pred_label:
                with torch.enable_grad():
                    input_img = all_inputs[i:i+1].to(device).requires_grad_()
                    skin_vec = all_skin_vecs_tensor[i:i+1].to(device)
                    heatmap = gradcam.generate(input_img, skin_vec, target_class=pred_label)

                    img_np = input_img.detach().squeeze().permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

                    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

                    save_path = os.path.join(gradcam_dir, f"sample{i}_true{true_label}_pred{pred_label}.png")
                    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # === Metrics and Reports ===
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    class_names = [str(cls) for cls in label_encoder.classes_] if label_encoder else [str(c) for c in sorted(set(y_true))]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
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

    plot_evaluation_results(
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        confusion=cm,
        class_names=class_names,
        skin_vecs=all_skin_vecs,
        mst_bins=all_mst_bins,
        skin_groups=all_skin_groups,
        output_dir=graph_dir
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

    if visualize_gradcam and gradcam_layer:
        print(f"✅ Grad-CAM misclassified overlays saved to: {os.path.join(graph_dir, 'gradcam_misclassified')}")

    mst_dist_path = os.path.join(graph_dir, f"{model_name}_mst_distribution_by_class.png")
    plot_mst_distribution_by_class(y_true, all_mst_bins, class_names, save_path=mst_dist_path)

    return acc, report, cm