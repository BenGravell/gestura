"""Constants."""

from train_config import TRAIN_CONFIG


DATASET_NAME = "UWaveGestureLibrary"
NUM_CLASSES = 8
SEQUENCE_LENGTH = 315
INPUT_SIZE = 3
OUTPUT_SIZE = NUM_CLASSES

LABELS = [i for i in range(NUM_CLASSES)]
LABEL_TO_NAME_MAP = {
    0: "Angle Down",
    1: "Square CW",
    2: "Straight Right",
    3: "Straight Left",
    4: "Straight Up",
    5: "Straight Down",
    6: "Circle CW",
    7: "Circle CCW",
}

HEAD_NAMES = [f"Head {i}" for i in range(TRAIN_CONFIG.num_heads)]
