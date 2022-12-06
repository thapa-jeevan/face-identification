import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

IMG_HEIGHT, IMG_WIDTH = 218, 178
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH)

# FaceNonFace classification
IDX_2_CATEGORY = ["face", "non-face"]
CATEGORY_2_IDX = {v: k for k, v in enumerate(IDX_2_CATEGORY)}
