import torch # type: ignore

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🚀 Using CUDA GPU")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("⚡ Using MPS backend")

BASE_DATASET_DIR = "/Users/ticaurisstokes/Desktop/research utilities/filtered_fairface_by_race/"
RESULTS_DIR = '/Users/ticaurisstokes/Desktop/grad_research/save_dir'
EMBED_NPY = "/Users/ticaurisstokes/Desktop/research utilities/race_embeddings.npt"
MIN_PER_COMBO = 150
