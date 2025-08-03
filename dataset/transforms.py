from torchvision import transforms # type: ignore
# === Augmentation Policies ===
standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # ✅ must come before Normalize
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

aggressive_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomAffine(degrees=15),
    transforms.ToTensor(),  # ✅ convert BEFORE normalization/erasing
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
    transforms.ToTensor(),  # ✅ before normalize
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3)
])

class ClassBasedAugmentationSchedule:
    def __init__(self, confused=None, underrepresented=None, dominant=None, mst_policy_map=None):
        self.confused = confused or {2, 4}
        self.underrepresented = underrepresented or {3, 6, 7}
        self.dominant = dominant or {0, 1}
        self.mst_policy_map = mst_policy_map or {}

    def get_transform(self, epoch, class_label, mst_group=None):
        if epoch < 5:
            return "standard_transform"

        if mst_group and mst_group in self.mst_policy_map:
            return self.mst_policy_map[mst_group]

        if class_label in self.underrepresented:
            return "specific_transform"
        elif class_label in self.confused:
            return "aggressive_transform"
        elif class_label in self.dominant:
            return "standard_transform"
        else:
            return "standard_transform"