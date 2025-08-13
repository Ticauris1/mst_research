import torch # type: ignore

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🚀 Using CUDA GPU")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("⚡ Using MPS backend")

BASE_DATASET_DIR = "/Users/ticaurisstokes/Desktop/research utilities/filtered_fairface_by_race/"
BASE_DATASET_DIR_4 = "/Users/ticaurisstokes/Desktop/research utilities/filtered_fairface_4_by_race/"
RESULTS_DIR = '/Users/ticaurisstokes/Desktop/research utilities/save_dir'
EMBED_NPY = "/Users/ticaurisstokes/Desktop/research utilities/fairface_embeddings_v3.pt"
BATCH_SIZE = 64
MIN_PER_COMBO = 10