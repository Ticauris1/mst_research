import os
import numpy as np # type: ignore
import torch # type: ignore
from collections import defaultdict
from PIL import Image # type: ignore
from torch.utils.data import Dataset # type: ignore
from dataset.transforms import (
    standard_transform,
    aggressive_transform,
    specific_transform,
    ClassBasedAugmentationSchedule,
)
from utils.utils import ( extract_color_metrics_and_estimate_mst, bin_mst_to_skin_group, build_skin_vector)
# === Main Custom Dataset Class ===
class CustomDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        transform=None,
        include_skin_vec=False,
        skip_failed=True,
        epoch=0,
        triplet_embedding_dict=None,
        match_triplet_by_filename=True,
        confused_classes=None,
        underrepresented_classes=None,
        dominant_classes=None,
    ):
        self.transform = transform
        self.include_skin_vec = include_skin_vec
        self.skip_failed = skip_failed
        self.epoch = epoch
        self.triplet_embedding_dict = triplet_embedding_dict or {}
        self.match_triplet_by_filename = match_triplet_by_filename
        self.aug_schedule = ClassBasedAugmentationSchedule(
            confused=confused_classes,
            underrepresented=underrepresented_classes,
            dominant=dominant_classes,
        )
        self.data = []
        self.transform_usage = defaultdict(int)

        self.transform_map = {
            "standard_transform": standard_transform,
            "aggressive_transform": aggressive_transform,
            "specific_transform": specific_transform,
        }

        if isinstance(labels[0], str):
            class_names = sorted(set(labels))
            self.class_to_label = {name: idx for idx, name in enumerate(class_names)}
            labels = [self.class_to_label[l] for l in labels]
        else:
            self.class_to_label = None

        for img_path, label in zip(image_paths, labels):
            triplet_key = os.path.basename(img_path).lower() if match_triplet_by_filename else img_path.lower()
            embedding = self.triplet_embedding_dict.get(triplet_key)

            if embedding is None:
                if skip_failed:
                    continue
                embedding = np.zeros(128, dtype=np.float32)

            if include_skin_vec:
                color_metrics = extract_color_metrics_and_estimate_mst(img_path)
                if color_metrics is None or color_metrics.get("MST") is None:
                    if skip_failed:
                        continue
                    skin_vec = np.zeros(12, dtype=np.float32)
                    mst_bin, skin_group = -1, "unknown"
                else:
                    skin_vec = build_skin_vector(color_metrics)
                    mst_bin = color_metrics["MST"]
                    skin_group = bin_mst_to_skin_group(mst_bin)
            else:
                skin_vec, mst_bin, skin_group = None, -1, "unknown"

            self.data.append((img_path, label, skin_vec, mst_bin, skin_group, embedding))

    def __len__(self):
        return len(self.data)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, idx):
        img_path, label, skin_vec, mst_bin, skin_group, embedding = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            transform_key = self.aug_schedule.get_transform(self.epoch, label, skin_group)  # ✅ FIXED
            transform_fn = self.transform_map.get(transform_key)
            self.transform_usage[transform_key] += 1
            if transform_fn is None:
                raise ValueError(f"Unknown transform key: {transform_key}")
            image = transform_fn(image)


        label_tensor = torch.tensor(label, dtype=torch.long)

        if isinstance(embedding, np.ndarray):
            embedding_tensor = torch.from_numpy(embedding).float()
        elif isinstance(embedding, torch.Tensor):
            embedding_tensor = embedding.float()
        else:
            raise TypeError(f"Unexpected embedding type: {type(embedding)}")
        embedding_tensor = embedding_tensor.view(-1)

        if self.include_skin_vec:
            return (
                image,
                label_tensor,
                torch.tensor(skin_vec, dtype=torch.float32),
                mst_bin,
                skin_group,
                embedding_tensor,
            )
        else:
            return (
                image,
                label_tensor,
                embedding_tensor,
            )
