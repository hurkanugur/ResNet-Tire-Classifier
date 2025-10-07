# -------------------------
# Paths
# -------------------------
MODEL_PATH = "../model/tire_classifier.pth"
DATASET_PATH = "../data"

# -------------------------
# Training hyperparameters
# -------------------------
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 10
VAL_INTERVAL = 5

# -------------------------
# Data split ratios
# -------------------------
SPLIT_DATASET = True
TRAIN_SPLIT_RATIO = 0.8
VAL_SPLIT_RATIO = 0.1
TEST_SPLIT_RATIO = 0.1
SPLIT_RANDOMIZATION_SEED = None       # Int -> Reproducible splits | None -> Fully random splits
