from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent

    MODEL_PATH = "t5-base"
    TRAIN_FILE = BASE_DIR / "train.json"
    OUTPUT_DIR = BASE_DIR / "t5_training_output"

    GPU = "0"
    FORCE_CPU = False
    SEED = 0

    INPUT_PREFIX = "parse semantics: "
    MAX_INPUT_LENGTH = 256
    MAX_TARGET_LENGTH = 256

    EPOCHS = 500
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.0
    MAX_GRAD_NORM = 1.0

    PRINT_EVERY = 50
    SAVE_EVERY = 50

    START_TENSORBOARD = True
    OPEN_BROWSER = True
    TENSORBOARD_HOST = "127.0.0.1"
    TENSORBOARD_PORT = 6006