from collections import Counter
import os
import gc
import time
import numpy as np # type: ignore
from config import  RESULTS_DIR, DEVICE
import torch # type: ignore
import pandas as pd # type: ignore
#import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import StratifiedShuffleSplit # type: ignore
from dataset.custom_dataset import CustomDataset  # Assuming you have a CustomDataset class defined
from dataset.transforms import standard_transform  # Assuming you have a standard_transform defined
from training.utils import setup_directories, get_model_with_attention
from evaluation.plot_utils import get_gradcam_layer
from training.train_loop import local_train
from training.eval import evaluate_model
import traceback


def kfold_cross_validation(
    X, y, z, label_encoder, model_names, attention_types, num_classes,
    transform, num_folds=5, num_epochs=10, batch_size=64,
    save_root=RESULTS_DIR, triplet_embedding_dict=None, val_size=0.3
):
    device = DEVICE
    seed = np.random.randint(0, 99999)
    print(f"🎲 Using random_state = {seed} for this k-fold trial")

    splitter = StratifiedShuffleSplit(n_splits=num_folds, test_size=val_size, random_state=seed)

    # 🔁 70/30 stratified splits, repeated num_folds times
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        current_fold_num = fold_idx + 1  # Adjust for 1-based indexing for display
        print(f"\n🔁 Fold {current_fold_num}/{num_folds}")

        X_tr = [X[i] for i in train_idx]
        y_tr_orig = [y[i] for i in train_idx]
        z_tr = [z[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val_orig = [y[i] for i in val_idx]
        z_val = [z[i] for i in val_idx]
        print(f"Train set size: {len(X_tr)} | Val set size: {len(X_val)}")

        # 🔁 Re-map class labels to a contiguous range for each fold
        fold_classes = sorted(set(y_tr_orig + y_val_orig))
        fold_num_classes = len(fold_classes)
        class_mapping = {label: idx for idx, label in enumerate(fold_classes)}
        y_tr = [class_mapping[lbl] for lbl in y_tr_orig]
        y_val = [class_mapping[lbl] for lbl in y_val_orig]

        print(f"Fold {current_fold_num} Classes: {fold_classes} → Remapped to: {list(class_mapping.values())}")

        # Display class distribution for current fold's train and validation sets
        train_class_counts = Counter(y_tr)
        val_class_counts = Counter(y_val)

        print(f"📊 Fold {current_fold_num} Class Distribution (Train): {dict(train_class_counts)}")
        print(f"📊 Fold {current_fold_num} Class Distribution (Val): {dict(val_class_counts)}")

        y_tr = [class_mapping[lbl] for lbl in y_tr_orig]

        '''CLASS_POLICY_MAP = {
            0: "standard_transform",      # Black
            1: "standard_transform",           # East Asian
            2: "standard_transform",         # SE Asian
            3: "aggressive_transform",      # Indian
            4: "color_transform",         # Latino
            5: "aggressive_transform",    # Middle Eastern
            6: "geo_transform",    # White
        }'''

        CLASS_POLICY_MAP = {
            0: "standard_transform",     # Black
            1: "standard_transform",     # East Asian
            2: "aggressive_transform",   # Latino        
            3: "standard_transform",     # White
        }


        train_dataset = CustomDataset(
            image_paths=X_tr,
            labels=y_tr,
            metadata=z_tr, # <--- Pass z_tr here
            transform=None,
            include_skin_vec=True,
            skip_failed=False,
            triplet_embedding_dict=triplet_embedding_dict,
            class_policy_map=CLASS_POLICY_MAP,
        )

        train_dataset.set_epoch(0)

        val_dataset = CustomDataset(
            X_val,
            y_val,
            metadata=z_val, # <--- Pass z_val here
            transform=standard_transform,
            include_skin_vec=True,
            triplet_embedding_dict=triplet_embedding_dict,  # ✅ Add this line
        )

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

           # Iterate through each attention type and model name
        for attn_type in attention_types:
            for model_name in model_names:
                run_name = f"{model_name}_{attn_type}"
                print(f"\n🧪 Training: {run_name.upper()} — Fold {current_fold_num}")

                # Setup directories for saving checkpoints, best weights, graphs, and predictions
                checkpoint_path, best_weights_path, graph_dir, predictions_dir = setup_directories(
                    base_path=save_root,
                    model_name=model_name,
                    fold=current_fold_num,
                    attention_type=attn_type
                )

                try:
                    start_time = time.time()  # ⏱️ Start timer for training and evaluation

                    # Get the model with the specified attention type and other configurations
                    model = get_model_with_attention(
                        model_name=model_name,
                        num_classes=fold_num_classes,
                        attention_type=attn_type,
                        pretrained=True,
                        fold=current_fold_num,
                        weights_root=save_root,
                        resume=True, # Attempt to resume training from a checkpoint
                        use_film_before=True,
                        use_film_in_cbam=True,
                        use_triplet_embedding=True,
                        triplet_embedding_dim=512,
                        fusion_mode="mlp"  # Options: "gated", "concat", "mlp"
                    ).to(device) # Move model to the specified device (CPU/GPU/MPS)

                    # Call the local training loop function
                    model, training_history_data  = local_train(
                        train_loader=train_loader,
                        model=model,
                        device=device,
                        num_epochs=num_epochs,
                        lr=0.001,
                        val_loader=val_loader,
                        save_model_path=save_root,
                        model_name=model_name,
                        fold=current_fold_num,
                        resume_path=checkpoint_path,
                        alpha=0.3,
                        mixup_enabled=True,
                        warmup_epochs=5,
                        num_classes=fold_num_classes,
                        attention_type=attn_type,
                        y_train=y_tr
                    )

                    print('getting ready for graphs')

                    # Get the appropriate Grad-CAM layer for visualization
                    gradcam_layer = get_gradcam_layer(model, model_name)

                    # Evaluate the trained model on the validation set
                    evaluate_model(
                        model=model,
                        test_loader=val_loader,
                        device=device,
                        label_encoder=label_encoder,
                        save_dir=predictions_dir,
                        model_name=f"{model_name}_{attn_type}_fold{current_fold_num}",
                        mst_bins=[sample[3] for sample in val_dataset], # Extract MST bins for evaluation
                        skin_groups=[sample[4] for sample in val_dataset], # Extract skin groups for evaluation
                        visualize_gradcam=True,
                        gradcam_layer=gradcam_layer,
                        graph_dir=graph_dir, 
                        save_training_curves=True,      # <--- Set this to True
                        training_curves_data=training_history_data # <--- Pass the captured history
                    )

                    elapsed = time.time() - start_time
                    print(f"Training and evaluation time for {run_name.upper()} — Fold {current_fold_num}: {elapsed:.2f} seconds")

                # === Separate handling for CUDA Out of Memory (OOM) errors ===
                except torch.cuda.OutOfMemoryError as e:
                    print(f"CUDA OOM — Skipping {run_name.upper()} (Fold {current_fold_num}): {e}")
                    torch.cuda.empty_cache() # Clear CUDA cache to free memory

                # === Handle generic RuntimeError and specifically detect OOM keywords ===
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg:
                        print(f"CUDA OOM — Skipping {run_name.upper()} (Fold {current_fold_num}): {e}")
                        torch.cuda.empty_cache()
                    else:
                        print(f"RuntimeError — Skipping {run_name.upper()} (Fold {current_fold_num}): {e}")
                        traceback.print_exc() # Print full traceback for other RuntimeErrors

                # === Catch-all for any other unexpected errors ===
                except Exception as e:
                    print(f"Error — Skipping {run_name.upper()} (Fold {current_fold_num}): {e}")
                    traceback.print_exc() # Print full traceback for general exceptions

                finally:
                    # Clean up model and clear cache to free resources after each run
                    if 'model' in locals():
                        del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect() # Force garbage collection

def run_multiple_kfold_trials(
    X, y, label_encoder, model_names, attention_types, num_classes,
    transform, num_trials=3, num_folds=5, num_epochs=10, batch_size=32,
    save_root="results_dir", triplet_embedding_dict=None
):
    """
    Runs multiple k-fold cross-validation trials with different random seeds.

    Args:
        X (list): List of image paths (features).
        y (list): List of original class labels.
        label_encoder: Encoder to map original labels to display names.
        model_names (list): List of model architectures to train.
        attention_types (list): List of attention mechanisms to use.
        num_classes (int): Total number of unique classes.
        transform (callable): Transformation function.
        num_trials (int, optional): Number of k-fold trials to run. Defaults to 3.
        num_folds (int, optional): Number of folds per trial. Defaults to 5.
        num_epochs (int, optional): Number of training epochs per fold/model. Defaults to 10.
        batch_size (int, optional): Batch size. Defaults to 32.
        save_root (str, optional): Base directory for saving trial results. Defaults to "results_dir".
        triplet_embedding_dict (dict, optional): Dictionary for triplet embeddings. Defaults to None.

    Returns:
        list: A list of trial root directories where results are saved.
    """
    # Generate unique random seeds for each trial
    seeds = np.random.randint(0, 99999, size=num_trials)
    trial_tags = [] # To store the root directories for each trial

    for trial_num, seed in enumerate(seeds, 1):
        print(f"\n🚀 Starting Trial {trial_num}/{num_trials} with random seed {seed}")

        # Adjust save_root to nest results by trial (e.g., results_dir/trial_1_seed_XXXXX)
        trial_root = os.path.join(save_root, f"trial_{trial_num}_seed_{seed}")
        trial_tags.append(trial_root)

        # Call the k-fold cross-validation function for the current trial
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
            save_root=trial_root, # Pass the trial-specific save root
            triplet_embedding_dict=triplet_embedding_dict
        )

    return trial_tags