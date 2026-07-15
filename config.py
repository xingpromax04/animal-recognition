"""Project configuration."""

from pathlib import Path
import os

import torch


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = BASE_DIR / "output"
MODEL_SAVE_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ANIMAL_CLASSES = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
    "American_Bulldog",
    "American_Pit_Bull_Terrier",
    "Basset_Hound",
    "Beagle",
    "Boxer",
    "Chihuahua",
    "English_Cocker_Spaniel",
    "English_Setter",
    "German_Shorthaired",
    "Great_Pyrenees",
    "Havanese",
    "Japanese_Chin",
    "Keeshond",
    "Leonberger",
    "Miniature_Pinscher",
    "Newfoundland",
    "Pomeranian",
    "Pug",
    "Saint_Bernard",
    "Samoyed",
    "Scottish_Terrier",
    "Shiba_Inu",
    "Staffordshire_Bull_Terrier",
    "Wheaten_Terrier",
    "Yorkshire_Terrier",
]

NUM_CLASSES = len(ANIMAL_CLASSES)

IMG_SIZE = 224
VAL_RESIZE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = max(2, min(8, (os.cpu_count() or 4) // 2))
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = NUM_WORKERS > 0

EPOCHS = 30
LEARNING_RATE = 3e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.1

MODEL_BACKBONE = "efficientnet_b0"
LEGACY_BACKBONE = "resnet50"
PRETRAINED = True
FREEZE_LAYERS = False
DROPOUT = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = torch.cuda.is_available()

REJECTION_THRESHOLD = 0.40
REJECTION_MARGIN = 0.08
REJECTION_MESSAGE = "不在当前类别范围内，无法可靠识别"

LOG_INTERVAL = 20
SAVE_INTERVAL = 5

TRAIN_AUGMENTATION = {
    "random_rotation": 10,
    "random_horizontal_flip": True,
    "random_resized_crop_scale": (0.75, 1.0),
    "random_resized_crop_ratio": (0.85, 1.15),
    "color_jitter": (0.15, 0.15, 0.15, 0.05),
    "random_erasing_p": 0.15,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
}

BEST_MODEL_PATH = MODEL_SAVE_DIR / "best_model.pth"
LAST_MODEL_PATH = MODEL_SAVE_DIR / "last_model.pth"
MODEL_META_PATH = OUTPUT_DIR / "model_meta.json"
