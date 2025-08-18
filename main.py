import os
import torch # type: ignore
from config import DEVICE, BASE_DATASET_DIR_4, EMBED_NPY, RESULTS_DIR, CFD_DIR
from load import load_img_from_dir
from utils.utils import extract_color_metrics_and_estimate_mst
from training.kfold import kfold_cross_validation
from sklearn.preprocessing import LabelEncoder  # type: ignore
from final_test import final_test

if __name__ == "__main__":
    # === 🔧 Configuration ===
    train_dataset_dir = BASE_DATASET_DIR_4
    test_dataset_dir = CFD_DIR  # Use the config variable for the separate test set
    save_model_root = RESULTS_DIR
    triplet_path = EMBED_NPY

    num_epochs = 50
    batch_size = 64
    n_splits = 2
    model_names = ["resnet152d"]
    attention_types = ["cbam"]

    # === 📂 1. Load and Process Training Data ===
    print("--- Loading and Processing Training Data ---")
    X_train_paths, y_train_labels = load_img_from_dir(train_dataset_dir, max_images_per_class=4150)
    if not X_train_paths:
        raise RuntimeError(f"❌ No training images found in: {train_dataset_dir}")

    triplet_embedding_dict = torch.load(triplet_path, map_location='cpu')

    X_train, y_train, z_train = [], [], []
    for path, label in zip(X_train_paths, y_train_labels):
        if os.path.basename(path).lower() in triplet_embedding_dict:
            color_metrics = extract_color_metrics_and_estimate_mst(path)
            if color_metrics and 1 <= color_metrics.get("MST", 0) <= 10:
                X_train.append(path)
                y_train.append(label)
                z_train.append(color_metrics) # z_train is a list of dictionaries

    print(f"✅ Total usable training images: {len(X_train)}")

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    num_classes = len(label_encoder.classes_)

    # === 🔁 2. Run K-fold Cross-Validation on the Training Set ===
    kfold_cross_validation(
        X=X_train,
        y=y_train_encoded,
        z=z_train, # Pass the correct list of dictionaries
        label_encoder=label_encoder,
        model_names=model_names,
        attention_types=attention_types,
        num_classes=num_classes,
        transform=None,
        num_folds=n_splits,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_root=save_model_root,
        triplet_embedding_dict=triplet_embedding_dict
    )
'''
    # === 🧪 3. Prepare and Run Final Test on the Separate Test Set ===
    print("\n--- Loading and Processing Final Test Data ---")
    X_test_paths, y_test_labels = load_img_from_dir(test_dataset_dir)

    if not X_test_paths:
        print(f"⚠️ No test images found in {test_dataset_dir}. Skipping final test.")
    else:
        # Process the test data just like the training data to get metadata dictionaries
        X_test, y_test, z_test = [], [], []
        for path, label in zip(X_test_paths, y_test_labels):
            if os.path.basename(path).lower() in triplet_embedding_dict:
                color_metrics = extract_color_metrics_and_estimate_mst(path)
                if color_metrics and 1 <= color_metrics.get("MST", 0) <= 10:
                    X_test.append(path)
                    y_test.append(label)
                    z_test.append(color_metrics)
        
        # Encode test labels using the *same* encoder from training
        y_test_encoded = []
        valid_indices = []
        for i, label in enumerate(y_test):
            if label in label_encoder.classes_:
                y_test_encoded.append(label_encoder.transform([label])[0])
                valid_indices.append(i)
            else:
                print(f"Ignoring unknown label '{label}' in test set.")

        # Filter the data to only include samples with known labels
        X_test = [X_test[i] for i in valid_indices]
        z_test = [z_test[i] for i in valid_indices] # z_test is now a list of dictionaries

        print(f"✅ Total usable test images: {len(X_test)}")
        
        if X_test:
            # Call the final test function
            final_test(
                test_data=(X_test, y_test_encoded, z_test),
                label_encoder=label_encoder,
                save_root=save_model_root,
                batch_size=batch_size,
                triplet_embedding_dict=triplet_embedding_dict
            )
'''