import os
import numpy as np
import torch
from collections import defaultdict
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from dataset.transforms import (
    standard_transform,
    aggressive_transform,
    ClassBasedAugmentationSchedule,
)
from utils.utils import (
    extract_color_metrics_and_estimate_mst,
    bin_mst_to_skin_group,
    build_skin_vector
)

class CustomDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        metadata=None,
        transform=None,
        include_skin_vec=False,
        skip_failed=True,
        epoch=0,
        triplet_embedding_dict=None,
        match_triplet_by_filename=True,
        class_policy_map=None,
        num_classes=None,
    ):
        self.transform = transform
        self.include_skin_vec = include_skin_vec
        self.skip_failed = skip_failed
        self.epoch = epoch
        self.triplet_embedding_dict = triplet_embedding_dict or {}
        self.match_triplet_by_filename = match_triplet_by_filename
        self.metadata = metadata if metadata is not None else []

        self.aug_schedule = ClassBasedAugmentationSchedule(
            class_policy_map=class_policy_map,
            num_classes=num_classes
        )

        self.data = []
        self.transform_usage = defaultdict(int)

        self.transform_map = {
            "standard_transform": standard_transform,
            "aggressive_transform": aggressive_transform,
            "specific_transform": standard_transform,
            "color_transform": standard_transform,
            "geo_transform": standard_transform,
        }

        if labels and isinstance(labels[0], str):
            class_names = sorted(set(labels))
            self.class_to_label = {name: idx for idx, name in enumerate(class_names)}
            labels = [self.class_to_label[l] for l in labels]
        else:
            self.class_to_label = None

        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            # --- Triplet Embedding Handling ---
            triplet_key = os.path.basename(img_path).lower() if match_triplet_by_filename else img_path.lower()
            embedding = self.triplet_embedding_dict.get(triplet_key)
            if embedding is None:
                if skip_failed:
                    print(f"Skipping {img_path}: Triplet embedding not found.")
                    continue
                embedding = np.zeros(512, dtype=np.float32)

            # --- Robust Skin Vector (MST) Handling ---
            skin_vec = np.zeros(12, dtype=np.float32)
            mst_bin, skin_group = -1, "unknown"
            raw_metadata_for_plot = {"MST": -1, "error": "No metadata"}

            if include_skin_vec and self.metadata and i < len(self.metadata):
                current_metadata = self.metadata[i]
                
                # FIX: This logic ensures raw_metadata_for_plot is always a dictionary
                if isinstance(current_metadata, dict):
                    raw_metadata_for_plot = current_metadata
                    try:
                        skin_vec = build_skin_vector(current_metadata)
                        mst_bin = current_metadata.get("MST", -1)
                        skin_group = bin_mst_to_skin_group(mst_bin)
                    except Exception as e:
                        print(f"Warning: Failed to build skin vector from metadata dict for {img_path}: {e}")
                else:
                    # If metadata is not a dict, extract it from the image as a fallback
                    extracted_metrics = extract_color_metrics_and_estimate_mst(img_path)
                    if extracted_metrics:
                        raw_metadata_for_plot = extracted_metrics
                        try:
                            skin_vec = build_skin_vector(extracted_metrics)
                            mst_bin = extracted_metrics.get("MST", -1)
                            skin_group = bin_mst_to_skin_group(mst_bin)
                        except Exception as e:
                            print(f"Warning: Failed to build skin vector from extracted metrics for {img_path}: {e}")

            self.data.append((img_path, label, skin_vec, mst_bin, skin_group, embedding, raw_metadata_for_plot))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, skin_vec, mst_bin, skin_group, embedding, raw_metadata_for_plot = self.data[idx]

        try:
            img = Image.open(img_path).convert("RGB")
            current_transform_name = self.aug_schedule.get_transform(self.epoch, label)
            current_transform = self.transform_map.get(current_transform_name, standard_transform)
            img_tensor = current_transform(img)
            self.transform_usage[current_transform_name] += 1
        except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
            print(f"⚠️ Failed to load or process image {img_path}: {e}")
            return (torch.zeros(3, 224, 224), label, torch.zeros(12), -1, "unknown",
                    torch.zeros(512), {"MST": -1, "error": "Image load failed"})

        skin_vec_tensor = torch.tensor(skin_vec, dtype=torch.float32)
        
        if isinstance(embedding, torch.Tensor):
            embedding_tensor = embedding.clone().detach()
        else:
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        
        return img_tensor, label, skin_vec_tensor, mst_bin, skin_group, embedding_tensor, raw_metadata_for_plot

    def set_epoch(self, epoch):
        """Sets the current epoch for the dataset, used by the augmentation schedule."""
        self.epoch = epoch
        self.transform_usage.clear()
'''
# === Custom Dataset Class ===
def get_transform(self, epoch, class_label):
    if epoch < 5:
        return "standard_transform"

    if class_label not in self.class_policy_map:
        print(f"⚠️ Unknown class label: {class_label}, defaulting to standard_transform")
    return self.class_policy_map.get(class_label, "standard_transform")
'''        
    