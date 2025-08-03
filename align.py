import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import zipfile


# === Paths ===
predictor_path = "/content/drive/My Drive/research_project/shape_predictor_68_face_landmarks.dat"
img_root = "/content/FairFace"
train_csv = os.path.join(img_root, "train_labels.csv")
val_csv = os.path.join(img_root, "val_labels.csv")

main_output_dir = "/content/fairface_aligned_all"
zip_output_path = "/content/drive/My Drive/fairface_aligned_all.zip"
debug_dir = os.path.join(main_output_dir, "debug_faces")

# === Create output dirs
os.makedirs(main_output_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

# === Load labels
df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)
df_train["split"] = "train"
df_val["split"] = "val"
df = pd.concat([df_train, df_val], ignore_index=True)

# === Initialize Face Detector and Embedder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# === Store results
embeddings = []
labels = []
failed_files = []

# === Debug control
debug_limit = 10
debug_count = 0

# === Main loop
for _, row in tqdm(df.iterrows(), total=len(df)):
    img_name = row['file']
    race = row['race']
    split = row['split']

    # Fix image path
    img_path = os.path.join(img_root, img_name)

    if not os.path.exists(img_path):
        failed_files.append(img_name)
        continue

    try:
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)

        if face is None:
            failed_files.append(img_name)
            continue

        face = face.unsqueeze(0).to(device)
        embedding = resnet(face).detach().cpu().squeeze().numpy()

        embeddings.append(embedding)
        labels.append({'file': img_name, 'race': race, 'split': split})

        # Save debug image (first few only)
        if debug_count < debug_limit:
            aligned_pil = transforms.ToPILImage()(face.squeeze().cpu())
            debug_path = os.path.join(debug_dir, os.path.basename(img_name))
            aligned_pil.save(debug_path)
            debug_count += 1

    except Exception as e:
        print(f"❌ Failed {img_name}: {e}")
        failed_files.append(img_name)

print(f"\n✅ Finished embedding extraction.")
print(f"📉 Failed on {len(failed_files)} images.")
print(f"📊 Successful embeddings: {len(embeddings)}")

# === Save outputs
np.save(os.path.join(main_output_dir, "embeddings.npy"), np.array(embeddings))
pd.DataFrame(labels).to_csv(os.path.join(main_output_dir, "embedding_labels.csv"), index=False)

# Save failed image names
with open(os.path.join(main_output_dir, "failed_images.txt"), "w") as f:
    for name in failed_files:
        f.write(f"{name}\n")

print(f"💾 Saved embeddings, labels, and debug images to: {main_output_dir}")
