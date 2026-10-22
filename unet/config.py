from pathlib import Path


class Config:
    GPU = "0"
    FORCE_CPU = False
    SEED = 0
    CPU_THREADS = 1

    CHANNELS = 4
    SIZE = 64
    WIDTHS = (16, 32, 64, 128)
    GROUPS = 4

    LEARNING_RATE = 1e-5
    ITERATIONS = 2000
    PRINT_EVERY = 50
    SAVE_EVERY = 500

    TRAIN_NEGATIVE_KEYS = 8
    DELTA_STD = 0.25
    NUMERICAL_EPS = 1e-8

    PAIR_ID = " "
    CORRECT_KEY = " "

    START_TENSORBOARD = True
    OPEN_BROWSER = True
    TENSORBOARD_HOST = "127.0.0.1"
    TENSORBOARD_PORT = 6006

    OUTPUT_DIR = Path(__file__).resolve().parent / "training_output"