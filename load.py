import os

def load_img_from_dir(dir_path, max_images_per_class=None):
    image_paths, labels = [], []

    for class_name in sorted(os.listdir(dir_path)):
        class_path = os.path.join(dir_path, class_name)
        if not os.path.isdir(class_path):
            continue

        count = 0
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_path, img_name)
            if not os.path.isfile(img_path):
                continue

            image_paths.append(img_path)
            labels.append(class_name)
            count += 1

            if max_images_per_class and count >= max_images_per_class:
                break

    print(f"✅ Loaded {len(image_paths)} usable images.")
    return image_paths, labels