import torch # type: ignore

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🚀 Using CUDA GPU")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("⚡ Using MPS backend")
else:
    DEVICE = torch.device("cpu")
    print("🐌 Using CPU")

# Directory for training and validation data
BASE_DATASET_DIR = "/Users/ticaurisstokes/Desktop/research utilities/filtered_fairface_by_race/"
BASE_DATASET_DIR_4 = "/Users/ticaurisstokes/Desktop/research utilities/filtered_fairface_4_by_race/"

# Directory for the final hold-out test set
CFD_DIR = "/Users/ticaurisstokes/Desktop/CFD/"

# Directory to save model weights, results, and graphs
RESULTS_DIR = '/Users/ticaurisstokes/Desktop/research utilities/save_dir'

# Path to the pre-computed embeddings file
EMBED_NPY = "/Users/ticaurisstokes/Desktop/research utilities/fairface_embeddings_v3.pt"