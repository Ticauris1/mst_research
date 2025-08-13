import os
import torch # type: ignore
from config import  BASE_DATASET_DIR, BASE_DATASET_DIR_4, EMBED_NPY, MIN_PER_COMBO, RESULTS_DIR
from load import load_img_from_dir
from  utils.utils import extract_color_metrics_and_estimate_mst
from utils.utils import bin_mst_to_skin_group, stratified_sample_enforced_mst_class
from training.kfold import kfold_cross_validation
from sklearn.preprocessing import LabelEncoder # type: ignore
from torchvision import transforms # type: ignore
from collections import Counter


if __name__ == "__main__":
    # === 🔧 Configuration ===
    dataset_dir = BASE_DATASET_DIR_4
    save_model_root = RESULTS_DIR
    triplet_path = EMBED_NPY

    num_epochs = 50
    batch_size = 32
    n_splits = 2
    num_trials = 1
    model_names = [ "resnet152d"]
    attention_types = ["cbam"]


    # === 📂 Load image paths and labels ===
    X_paths, y_labels = load_img_from_dir(dataset_dir, max_images_per_class=4150)
    if not X_paths or not y_labels:
        raise RuntimeError(f"❌ No images or labels found in: {dataset_dir}")

    # === 📥 Load triplet embeddings from .pt ===
    triplet_embedding_dict = torch.load(triplet_path, map_location='cpu')
    #for k, v in list(triplet_embedding_dict.items())[:5]:
        #print(f"{k}: shape={v.shape}, dtype={v.dtype}")


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

    #print(f"🔢 Total filtered images before sampling: {len(X)}")
    #print(f"❌ Skipped {len(skipped)} images due to failed MST estimation.")

    mst_groups = [entry["MST"] for entry in z]
    combo_counts = Counter([(label, mst) for label, mst in zip(y, mst_groups)])

    #print("\n🔍 Available (class, MST) combos:")
    #for combo, count in sorted(combo_counts.items()):
        #print(f"{combo}: {count}")


    # === 🧪 Sample with fallback if 0 samples ===
    per_class_targets = {
        "Black": 3150,
        "East Asian": 3150,
        "Indian": 4150,
        "Latino_Hispanic": 4150,  # prioritize weak class
        "Middle Eastern": 4150,
        "Southeast Asian": 3150,
        "White": 3150
    }

    per_class_caps = {
       "Black": 3150,
        "East Asian": 3150,
        "Indian": 4150,
        "Latino_Hispanic": 4150,  # prioritize weak class
        "Middle Eastern": 4150,
        "Southeast Asian": 3150,
        "White": 3150
    }

    X_sampled, y_sampled, z_sampled, counts, skipped = stratified_sample_enforced_mst_class(
        X, y, z,
        group_fn=bin_mst_to_skin_group,
        per_class_targets=per_class_targets,
        per_class_caps=per_class_caps,
        min_per_combo=5,
        allow_undersample_below_min_combo=True
    )

    if len(X_sampled) == 0:
        print("⚠️ No samples after stratified sampling — falling back to all filtered data.")
        X_sampled, y_sampled, z_sampled = X, y, z
        combo_counts = Counter([(label, bin_mst_to_skin_group(entry["MST"]))
                                for label, entry in zip(y_sampled, z_sampled)])

    # ✅ Final dataset assignment
    X, y, z = X_sampled, y_sampled, z_sampled
    mst_groups = [entry["MST"] for entry in z]


    print(f"\n📦 Total Samples: {len(y)}")
    print("\n📊 Final Class Counts:")
    for cls, count in sorted(Counter(y).items()):
        print(f"Class {cls}: {count}")

    #print("\n✅ Final Sampled Class-MST Distribution:")
    #sampled_combo_labels = [f"{c}-{bin_mst_to_skin_group(m['MST'])}" for c, m in zip(y, z)]
    #for combo, count in sorted(Counter(sampled_combo_labels).items()):
    #    print(f"{combo}: {count}")

    # === 🏷️ Encode labels ===
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    # === 🔄 Transform pipeline ===
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    '''print("📊 Checking input shapes before k-fold")
    print(f"🔢 X shape: {X.shape if hasattr(X, 'shape') else type(X)}")
    print(f"🧠 y length: {len(y)}")
    print(f"🧬 Unique labels in y: {set(y)}")
    print(f"🚫 Triplet dict keys (sample): {list(triplet_embedding_dict.keys())[:5]}")'''


    # === 🔁 kfold trials ===
    kfold_cross_validation(
        X=X,
        y=y_encoded,
        z=z,  # Pass the metadata (color metrics)
        label_encoder=label_encoder,
        model_names=model_names,
        attention_types=attention_types,
        num_classes=num_classes,
        transform=transform,
        num_folds=n_splits,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_root=save_model_root,
        triplet_embedding_dict=triplet_embedding_dict
    )
