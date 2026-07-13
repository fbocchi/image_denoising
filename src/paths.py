from pathlib import Path


# Cartella principale del progetto.
#
# Se questo file è:
#
# progetto/src/paths.py
#
# allora:
#
# Path(__file__).resolve().parent
#
# corrisponde a:
#
# progetto/src
#
# e .parent.parent corrisponde a:
#
# progetto

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# File di configurazione

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "config.yaml"
)


# Directory dei dati preprocessati

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# Directory dei modelli

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)


# Directory degli output

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)


# Sottodirectory degli output

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

METRICS_DIR = (
    OUTPUT_DIR
    / "metrics"
)

PREDICTION_DIR = (
    OUTPUT_DIR
    / "predictions"
)


def create_directories():
    """
    Crea tutte le directory necessarie.

    Se esistono già, non genera errori.
    """

    directories = [
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        FIGURE_DIR,
        METRICS_DIR,
        PREDICTION_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )