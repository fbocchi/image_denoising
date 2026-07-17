from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"


def create_directories():

    directories = [
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        FIGURE_DIR,
        METRICS_DIR,
        PREDICTION_DIR,
    ]

    for directory in directories:

        directory.mkdir(parents=True, exist_ok=True)