import os
import torch # type: ignore
from config import DEVICE, BASE_DATASET_DIR, RESULTS_DIR, EMBED_NPY, MIN_PER_COMBO, RESULT_DIR
from load import load_img_from_dir
from  utils.utils import extract_color_metrics_and_estimate_mst
from utils.utils import bin_mst_to_skin_group, stratified_sample_no_oversampling
from training.kfold import run_multiple_kfold_trials, summarize_multiple_trials
from sklearn.preprocessing import LabelEncoder # type: ignore
from torchvision import transforms # type: ignore

if __name__ == "__main__":
    # === 🔧 Configuration ===
    dataset_dir = BASE_DATASET_DIR
    save_model_root = RESULT_DIR
    triplet_path = EMBED_NPY

    num_epochs = 100
    batch_size = 64
    n_splits = 5
    num_trials = 1
    model_names = ["efficientnet_b3"]
    attention_types = ["cbam"]
    MIN_PER_COMBO = 1


    # === 📂 Load image paths and labels ===
    X_paths, y_labels = load_img_from_dir(dataset_dir, max_images_per_class=10)
    if not X_paths or not y_labels:
        raise RuntimeError(f"❌ No images or labels found in: {dataset_dir}")

    # === 📥 Load triplet embeddings ===
    raw = torch.load(triplet_path)
    if isinstance(raw, dict) and "labels" in raw and "embeddings" in raw:
        triplet_embedding_dict = {
            f"{label}.jpg": torch.tensor(emb, dtype=torch.float32)
            for label, emb in zip(raw["labels"], raw["embeddings"])
        }
    else:
        triplet_embedding_dict = {
            os.path.basename(k).lower(): (
                v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
            )
            for k, v in raw.items()
        }

    # === 🔍 Filter image paths based on triplet keys ===
    X_paths = [p for p in X_paths if os.path.basename(p).lower() in triplet_embedding_dict]

    # === 🧠 Extract MST via ITA color estimation ===
    X, y, z = [], [], []
    skipped = []

    for path, label in zip(X_paths, y_labels):
        color_metrics = extract_color_metrics_and_estimate_mst(path)
        if color_metrics is not None and 1 <= color_metrics["MST"] <= 10:
            X.append(path)
            y.append(label)
            z.append(color_metrics)  # includes "L", "h", and "MST"
        else:
            skipped.append(path)

    from collections import Counter
    mst_groups = [entry["MST"] for entry in z]
    combo_counts = Counter([(label, mst) for label, mst in zip(y, mst_groups)])

    #print("\n🔍 Available (class, MST) combos:")
    #for combo, count in sorted(combo_counts.items()):
        #print(f"{combo}: {count}")

    #print(f"🔢 Total filtered images before sampling: {len(X)}")
    #print(f"❌ Skipped {len(skipped)} images due to failed MST estimation.")
    #if skipped:
        #print("🔍 Example skipped paths:", skipped[:3])

    #print("🧪 Sample (class, MST):")
    #for i in range(min(5, len(X))):
        #print(f"{y[i]} — MST {z[i]['MST']} → Group {bin_mst_to_skin_group(z[i]['MST'])}")

    # === 🧪 Sample with fallback if 0 samples ===
    X_sampled, y_sampled, z_sampled, combo_counts, _ = stratified_sample_no_oversampling(
        X, y, z,
        group_fn=bin_mst_to_skin_group,
        max_per_combo=250,
        min_per_combo=MIN_PER_COMBO
    )

    if len(X_sampled) == 0:
        print("⚠️ No samples after stratified sampling — falling back to all filtered data.")
        X_sampled, y_sampled, z_sampled = X, y, z
        combo_counts = Counter([(label, bin_mst_to_skin_group(entry["MST"]))
                                for label, entry in zip(y_sampled, z_sampled)])

    # ✅ Final dataset
    X, y, z = X_sampled, y_sampled, z_sampled
    mst_groups = [entry["MST"] for entry in z]

    print(f"\n📦 Total Samples: {len(y)}")
    print("\n📊 Final Class Counts:")
    for cls, count in sorted(Counter(y).items()):
        print(f"Class {cls}: {count}")

    #print("\n✅ Final Sampled Class-MST Distribution:")
    #sampled_combo_labels = [f"{c}-{bin_mst_to_skin_group(m['MST'])}" for c, m in zip(y, z)]
    #for combo, count in sorted(Counter(sampled_combo_labels).items()):
        #print(f"{combo}: {count}")

    # === 🏷️ Encode labels ===
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    # === 🔄 Transform pipeline ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # === 🔁 Run trials ===
    trial_folders = run_multiple_kfold_trials(
        X=X,
        y=y_encoded,
        label_encoder=label_encoder,
        model_names=model_names,
        attention_types=attention_types,
        num_classes=num_classes,
        transform=transform,
        num_trials=num_trials,
        num_folds=n_splits,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_root=save_model_root,
        triplet_embedding_dict=triplet_embedding_dict
    )

    # === 📊 Summarize results ===
    summarize_multiple_trials(
        trial_folders=trial_folders,
        model_names=model_names,
        attention_types=attention_types,
        label_encoder=label_encoder,
        output_path=os.path.join(save_model_root, "trial_summary")
    )
