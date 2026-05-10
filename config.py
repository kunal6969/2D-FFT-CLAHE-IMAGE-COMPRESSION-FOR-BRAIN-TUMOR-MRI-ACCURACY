import os

# Dataset and model paths
DATASET_ROOT = os.path.abspath("./brain_tumor_dataset")
MODEL_PATH = os.path.abspath("./output/models/best_baseline.pth")
OUTPUT_ROOT = os.path.abspath("./output")

# Directories
PROCESSED_DATASETS_DIR = os.path.join(OUTPUT_ROOT, "processed_datasets")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
VISUALS_DIR = os.path.join(OUTPUT_ROOT, "visuals")

# Ensure base directories exist
for d in [PROCESSED_DATASETS_DIR, RESULTS_DIR, VISUALS_DIR]:
    os.makedirs(d, exist_ok=True)

# FFT + CLAHE Settings
COMPRESSION_RATIOS = {
    "light": 0.4,
    "intermediate": 0.2,
    "aggressive": 0.1
}
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Model Settings
IMG_SIZE = 224
DEVICE = "cuda"
