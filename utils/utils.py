import os
import cv2 # type: ignore
import pandas as pd # type: ignore 
import numpy as np # type: ignore
from skimage import color # type: ignore
from collections import defaultdict, Counter
import torch # type: ignore
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score  # type: ignore

def bin_mst_to_skin_group(mst_value: int) -> str:
    return f"MST_{mst_value}" if 1 <= mst_value <= 10 else "unknown"

def normalize_color_features(L, h):
    L_scaled = L / 100.0         # L* ∈ [0, 100]
    h_scaled = h / 360.0         # h* ∈ [0, 360]
    return L_scaled, h_scaled

def extract_color_metrics(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None, None  # Unreadable image

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb / 255.0  # Normalize for skimage

    lab = color.rgb2lab(image_rgb)
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    h = np.degrees(np.arctan2(b, a)) % 360

    skin_pixels = l > 0
    avg_L = np.mean(l[skin_pixels])
    avg_h = np.mean(h[skin_pixels])

    return avg_L, avg_h

def estimate_mst_from_ita(ita_value):
    # Approximate Monk Skin Tone from ITA ranges
    if ita_value > 55: return 1
    elif ita_value > 41: return 2
    elif ita_value > 28: return 3
    elif ita_value > 19: return 4
    elif ita_value > 10: return 5
    elif ita_value > 0: return 6
    elif ita_value > -10: return 7
    elif ita_value > -20: return 8
    elif ita_value > -30: return 9
    else: return 10

def extract_color_metrics_and_estimate_mst(image_path):
    avg_L, avg_h = extract_color_metrics(image_path)
    if avg_L is None or avg_h is None:
        return None

    ita = np.degrees(np.arctan((avg_L - 50) / avg_h))
    mst_bin = estimate_mst_from_ita(ita)

    return {
        "L": avg_L,
        "h": avg_h,
        "MST": mst_bin
    }

def normalize_ita_hue(ita, hue):
    ita_scaled = (ita + 60) / 120  # ITA in ~[-60, 60]
    hue_scaled = hue / 360.0
    return ita_scaled, hue_scaled

def one_hot_encode_mst(mst_bin, num_classes=10):
    one_hot = np.zeros(num_classes)
    if 1 <= mst_bin <= num_classes:
        one_hot[mst_bin - 1] = 1.0
    return one_hot

def build_skin_vector(color_metrics):
    if color_metrics is None:
        return None

    L = color_metrics["L"]
    h = color_metrics["h"]
    ita = np.degrees(np.arctan((L - 50) / h))
    ita_scaled, hue_scaled = normalize_ita_hue(ita, h)

    mst_onehot = one_hot_encode_mst(color_metrics["MST"])
    return np.array([ita_scaled, hue_scaled], dtype=np.float32).tolist() + mst_onehot.tolist()

def stratified_sample_no_oversampling(
    X, y, z,
    group_fn,
    max_per_combo=250,
    min_per_combo=10
):

    combo_to_indices = defaultdict(list)
    for i, (label, meta) in enumerate(zip(y, z)):
        skin_group = group_fn(meta.get("MST", -1))
        if skin_group != "unknown":
            combo_to_indices[(label, skin_group)].append(i)

    # Keep only combos with enough samples
    valid_combos = {k: v for k, v in combo_to_indices.items() if len(v) >= min_per_combo}

    sampled_indices = []
    sampled_combo_counts = Counter()

    for (cls, group), indices in valid_combos.items():
        selected = indices[:max_per_combo]  # no upsampling
        sampled_indices.extend(selected)
        sampled_combo_counts[(cls, group)] = len(selected)

    skipped_combos = sorted(set(combo_to_indices.keys()) - set(sampled_combo_counts.keys()))

    X_filtered = [X[i] for i in sampled_indices]
    y_filtered = [y[i] for i in sampled_indices]
    z_filtered = [z[i] for i in sampled_indices]

    print(f"✅ Sampled {len(sampled_indices)} total from {len(sampled_combo_counts)} combos")
    if skipped_combos:
        print(f"⚠️ Skipped {len(skipped_combos)} combos due to min_per_combo={min_per_combo}: {skipped_combos}")

    return X_filtered, y_filtered, z_filtered, sampled_combo_counts, skipped_combos

def extract_color_metrics_and_estimate_mst(image_path):
    avg_L, avg_h = extract_color_metrics(image_path)
    if avg_L is None or avg_h is None:
        return None

    ita = np.degrees(np.arctan((avg_L - 50) / avg_h))
    mst_bin = estimate_mst_from_ita(ita)

    return {
        "L": avg_L,
        "h": avg_h,
        "MST": mst_bin
    }

def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 6:
                inputs, labels, skin_vecs, _, _, triplet_vecs = batch
            elif len(batch) == 5:
                inputs, labels, skin_vecs, _, _ = batch
                triplet_vecs = None
            elif len(batch) == 3:
                inputs, labels, skin_vecs = batch
                triplet_vecs = None
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            inputs = inputs.to(device)
            labels = labels.to(device)
            skin_vecs = skin_vecs.to(device)
            triplet_vecs = triplet_vecs.to(device) if triplet_vecs is not None else None

            # ✅ Expect model to support `return_features=True`
            outputs = model(inputs, skin_vecs, triplet_embedding=triplet_vecs, return_features=True)

            # Output is either (logits, features) or features only
            if isinstance(outputs, tuple):
                _, features = outputs
            else:
                features = outputs

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()

def compute_fairness_by_group(y_true, y_probs, class_names, skin_groups=None):
    y_preds = np.argmax(y_probs, axis=1)
    results = []
    if skin_groups is None:
        skin_groups = ['unknown'] * len(y_true)
    unique_groups = sorted(set(skin_groups))
    for group in unique_groups:
        indices = [i for i, g in enumerate(skin_groups) if g == group]
        if not indices:
            continue
        group_y_true = [y_true[i] for i in indices]
        group_y_pred = [y_preds[i] for i in indices]
        results.append({
            "Skin Group": group,
            "Accuracy": accuracy_score(group_y_true, group_y_pred),
            "Precision": precision_score(group_y_true, group_y_pred, average='macro', zero_division=0),
            "Recall": recall_score(group_y_true, group_y_pred, average='macro', zero_division=0),
            "F1": f1_score(group_y_true, group_y_pred, average='macro', zero_division=0),
        })
    return pd.DataFrame(results)