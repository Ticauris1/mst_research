import torch # type: ignore
import os
import numpy as np # type: ignore
from PIL import Image, UnidentifiedImageError # type: ignore
from torchvision import transforms # type: ignore
from collections import defaultdict
from torch.utils.data import Dataset # type: ignore
from utils.utils import build_skin_vector, bin_mst_to_skin_group
from sklearn.metrics import recall_score  # type: ignore

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
    def __init__(self, class_policy_map=None, num_classes=0):
        self.class_policy_map = class_policy_map or {}
        # Initialize performance with a perfect score to use base policies initially
        self.class_performance = {i: 1.0 for i in range(num_classes)}

    def update_performance(self, y_true, y_pred):
        """Calculates and updates the recall for each class."""
        # Get the unique labels that are present in the true values to avoid errors
        present_labels = sorted(list(set(y_true)))
        
        # Calculate recall for the labels that are actually present
        recalls = recall_score(y_true, y_pred, average=None, labels=present_labels, zero_division=0)
        
        # Create a dictionary mapping the present labels to their recall scores
        recall_map = dict(zip(present_labels, recalls))
        
        # Update the performance metric for each class based on the new recall scores
        for class_idx in self.class_performance.keys():
            # If a class was present in this validation batch, update its score
            # Otherwise, its score remains unchanged
            if class_idx in recall_map:
                self.class_performance[class_idx] = recall_map[class_idx]

        print(f"📊 Updated Augmentation Performance Metrics: {self.class_performance}")


    def get_transform(self, epoch, class_label):
        class_label = int(class_label)
        # Use a simple transform during warmup epochs
        if epoch < 5:
            return "standard_transform"

        # Get the base policy for the given class
        base_policy = self.class_policy_map.get(class_label, "standard_transform")
        
        # If a class's recall is below a threshold (e.g., 75%),
        # switch to a more aggressive augmentation policy.
        if self.class_performance.get(class_label, 1.0) < 0.75:
            # You could even have multiple tiers of aggression
            # print(f"Applying aggressive transform for class {class_label} due to low recall.")
            return "aggressive_transform"
        
        return base_policy
