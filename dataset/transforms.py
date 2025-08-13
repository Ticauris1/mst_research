import torch # type: ignore
import os
import numpy as np # type: ignore
from PIL import Image, UnidentifiedImageError # type: ignore
from torchvision import transforms # type: ignore
from collections import defaultdict
from torch.utils.data import Dataset # type: ignore
from utils.utils import build_skin_vector, bin_mst_to_skin_group
# === Transforms ===
# === Transforms ===
standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

aggressive_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomAffine(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])

specific_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
    ], p=0.9),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3)
])

color_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7),
    ], p=0.9),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

geo_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
    ], p=0.9),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3),
])

class ClassBasedAugmentationSchedule:
    def __init__(self, class_policy_map=None):
        self.class_policy_map = class_policy_map or {}

    def get_transform(self, epoch, class_label):
        class_label = int(class_label)
        if epoch < 5:
            return "standard_transform"
        if class_label not in self.class_policy_map:
            print(f"⚠️ Unknown class label: {class_label}, defaulting to standard_transform")
        return self.class_policy_map.get(class_label, "standard_transform")





