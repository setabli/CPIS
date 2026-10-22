from pathlib import Path


class TGRSConfig:
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent
    SEG_DIR = BASE_DIR

    DINO_CONFIG = SEG_DIR / "GroundingDINO_SwinT_OGC.py"
    DINO_CHECKPOINT = SEG_DIR / "groundingdino_swint_ogc.pth"
    BERT_MODEL_PATH = SEG_DIR / "bert-base-uncased"

    SAM_CHECKPOINT = SEG_DIR / "sam_vit_h_4b8939.pth"
    SAM_MODEL_TYPE = "vit_h"

    GPU = "0"
    FORCE_CPU = False
    SEED = 0

    IMAGE_PATH = PROJECT_DIR / "images" / "input.png"
    OUTPUT_DIR = PROJECT_DIR / "tgrs_output"

    TARGET_TEXT = "car"
    REGION_MODE = "preserve"

    IMAGE_SIZE = (512, 512)
    LATENT_SIZE = (64, 64)

    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.5
    MASK_THRESHOLD = 0.5

    SAM_BATCH_SIZE = 4