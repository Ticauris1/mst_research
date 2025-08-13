import os
import numpy as np # type: ignore
import torch # type: ignore
from collections import defaultdict
from PIL import Image, UnidentifiedImageError # type: ignore
from torch.utils.data import Dataset # type: ignore
from dataset.transforms import (
    standard_transform,
    aggressive_transform,
    specific_transform,
    color_transform,
    geo_transform,  
    ClassBasedAugmentationSchedule,
)
from utils.utils import ( extract_color_metrics_and_estimate_mst, bin_mst_to_skin_group, build_skin_vector)

# === Custom Dataset Class ===
def get_transform(self, epoch, class_label):
    if epoch < 5:
        return "standard_transform"

    if class_label not in self.class_policy_map:
        print(f"⚠️ Unknown class label: {class_label}, defaulting to standard_transform")
    return self.class_policy_map.get(class_label, "standard_transform")

class CustomDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        metadata=None, # Add metadata parameter (list of dicts containing MST)
        transform=None,
        include_skin_vec=False,
        skip_failed=True,
        epoch=0,
        triplet_embedding_dict=None,
        match_triplet_by_filename=True,
        class_policy_map=None,
    ):
        self.transform = transform
        self.include_skin_vec = include_skin_vec
        self.skip_failed = skip_failed
        self.epoch = epoch
        self.triplet_embedding_dict = triplet_embedding_dict or {}
        self.match_triplet_by_filename = match_triplet_by_filename
        self.metadata = metadata if metadata is not None else [] # Store metadata list

        self.aug_schedule = ClassBasedAugmentationSchedule(
            class_policy_map=class_policy_map
        )

        self.data = [] # This will store (img_path, label, skin_vec_np, mst_bin, skin_group_str, embedding_np, raw_metadata_dict_for_plot)
        self.transform_usage = defaultdict(int)

        self.transform_map = {
            "standard_transform": standard_transform,
            "aggressive_transform": aggressive_transform,
            "specific_transform": standard_transform, # Placeholder, replace with actual
            "color_transform": standard_transform,     # Placeholder, replace with actual
            "geo_transform": standard_transform,       # Placeholder, replace with actual
        }
        # Ensure all transforms in CLASS_POLICY_MAP are defined/imported.
        # If specific_transform, color_transform, geo_transform are not defined,
        # you might use standard_transform as a fallback or raise an error.

        # If labels are strings, convert them to numerical format
        if labels and isinstance(labels[0], str): # Check if labels list is not empty and first element is string
            class_names = sorted(set(labels))
            self.class_to_label = {name: idx for idx, name in enumerate(class_names)}
            labels = [self.class_to_label[l] for l in labels]
        else:
            self.class_to_label = None # Labels are already numerical

        # Populate self.data list
        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            # --- Triplet Embedding Handling ---
            triplet_key = os.path.basename(img_path).lower() if match_triplet_by_filename else img_path.lower()
            embedding = self.triplet_embedding_dict.get(triplet_key)
            if embedding is None:
                if skip_failed:
                    # Skip samples if embedding is missing and skip_failed is true
                    print(f"Skipping {img_path}: Triplet embedding not found.")
                    continue
                # Provide a zero tensor as fallback if embedding is missing and not skipping
                embedding = np.zeros(512, dtype=np.float32)

            # --- Skin Vector (MST) Handling ---
            # Initialize with robust default values
            skin_vec = np.zeros(12, dtype=np.float32)
            mst_bin, skin_group = -1, "unknown"
            raw_metadata_for_plot = {"MST": -1, "ITA": 0, "L": 0, "h": 0, "error": "No metadata or invalid"} # Robust default dict

            if include_skin_vec:
                # Prioritize pre-computed metadata if provided and valid for current index
                if self.metadata and i < len(self.metadata):
                    current_metadata_dict = self.metadata[i]
                    if isinstance(current_metadata_dict, dict) and "MST" in current_metadata_dict:
                        # Use valid pre-computed metadata
                        try:
                            skin_vec = build_skin_vector(current_metadata_dict)
                            mst_bin = current_metadata_dict["MST"]
                            skin_group = bin_mst_to_skin_group(mst_bin)
                            raw_metadata_for_plot = current_metadata_dict # Store the actual dict for plotting
                        except Exception as e:
                            print(f"Warning: Failed to build skin vector or MST from pre-computed metadata for {img_path}: {e}")
                            # Fallback to default if there's an issue with the pre-computed dict's content
                    else:
                        print(f"Warning: Invalid metadata format at index {i} for {img_path}. Expected dict with 'MST'.")
                else:
                    # Fallback: If no metadata or index out of bounds, try re-extracting from image (less reliable)
                    # This path might lead to 'unknown' if image processing fails again.
                    print(f"Info: No pre-computed metadata for {img_path} (idx {i}). Attempting on-the-fly extraction.")
                    color_metrics_extracted = extract_color_metrics_and_estimate_mst(img_path)
                    if color_metrics_extracted is not None and color_metrics_extracted.get("MST") is not None:
                        try:
                            skin_vec = build_skin_vector(color_metrics_extracted)
                            mst_bin = color_metrics_extracted["MST"]
                            skin_group = bin_mst_to_skin_group(mst_bin)
                            raw_metadata_for_plot = color_metrics_extracted # Store the extracted dict
                        except Exception as e:
                            print(f"Warning: Failed to build skin vector from extracted metrics for {img_path}: {e}")
                    else:
                        print(f"Warning: On-the-fly MST extraction failed for {img_path}.")

            # Append all 7 items to self.data
            self.data.append((img_path, label, skin_vec, mst_bin, skin_group, embedding, raw_metadata_for_plot))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Unpack all 7 items from the stored tuple
        img_path, label, skin_vec, mst_bin, skin_group, embedding, raw_metadata_for_plot = self.data[idx]

        try:
            # Open image using PIL
            img = Image.open(img_path).convert("RGB")

            # Apply class-specific or standard transformations
            current_transform_name = self.aug_schedule.get_transform(self.epoch, label)
            current_transform = self.transform_map.get(current_transform_name, standard_transform)
            img_tensor = current_transform(img)
            self.transform_usage[current_transform_name] += 1 # Track usage
        except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
            print(f"⚠️ Failed to load or process image {img_path}: {e}")
            # Return dummy data (7 items) in case of image loading/processing failure
            if self.skip_failed:
                # If skip_failed, this specific item shouldn't have been added if initial loading failed.
                # However, for robustness during getitem, return dummy values.
                 return (torch.zeros(3, 224, 224), -1, torch.zeros(12), -1, "unknown",
                         torch.zeros(512), {"MST": -1, "error": "Image load failed"})
            else:
                # If not skipping failed, return dummy but preserve label, etc.
                 return (torch.zeros(3, 224, 224), label,
                         skin_vec if skin_vec is not None else torch.zeros(12),
                         mst_bin, skin_group,
                         embedding if embedding is not None else torch.zeros(512),
                         raw_metadata_for_plot) # Pass the original raw metadata

        # Convert numpy arrays to PyTorch tensors
        skin_vec_tensor = torch.tensor(skin_vec, dtype=torch.float32) if skin_vec is not None else torch.zeros(12, dtype=torch.float32)

        # Ensure embedding is a tensor
        if embedding is not None:
          if isinstance(embedding, torch.Tensor):
            embedding_tensor = embedding.clone().detach().float()
          else:
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        else:
            embedding_tensor = torch.zeros(512, dtype=torch.float32)

        assert isinstance(label, int), f"❌ Label is not int: {label} (type={type(label)})"

        # Return all 7 items as expected by the DataLoader and evaluate_model
        return img_tensor, label, skin_vec_tensor, mst_bin, skin_group, embedding_tensor, raw_metadata_for_plot

    def set_epoch(self, epoch):
        """Sets the current epoch for the dataset (used by augmentation schedule)."""
        self.epoch = epoch
        # self.aug_schedule.set_epoch(epoch) # Update the augmentation schedule's epoch - COMMENTED OUT
        self.transform_usage.clear() # Reset usage counts per epoch
