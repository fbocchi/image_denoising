import json

import numpy as np
from tensorflow import keras

from src.paths import (
    METRICS_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    create_directories,
)


def load_array(filename):
    file_path = PROCESSED_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"File non trovato: {file_path}\n"
            "Eseguire prima:\n"
            "python -m src.preprocessing"
        )

    return np.load(file_path)


def load_model():
    model_path = MODEL_DIR / "best_model.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modello non trovato: {model_path}\n"
            "Eseguire prima:\n"
            "python -m src.training"
        )

    return keras.models.load_model(model_path)


def evaluate_model(model, x_test_noisy, x_test_clean):
    return model.evaluate(
        x=x_test_noisy,
        y=x_test_clean,
        verbose=1,
    )


def save_mse(test_mse):
    output_path = METRICS_DIR / "test_metrics.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump({"test_mse": float(test_mse)}, file, indent=4)

    return output_path


def main():
    create_directories()

    print("\nCaricamento del test set...")
    x_test_clean = load_array("x_test_clean.npy")
    x_test_noisy = load_array("x_test_noisy.npy")

    print("Test pulito:", x_test_clean.shape)
    print("Test rumoroso:", x_test_noisy.shape)

    print("\nCaricamento del modello...")
    model = load_model()

    print("\nValutazione sul test set...")
    test_mse = evaluate_model(
        model=model,
        x_test_noisy=x_test_noisy,
        x_test_clean=x_test_clean,
    )

    print("\nRisultato finale:")
    print(f"Test MSE: {test_mse:.6f}")

    output_path = save_mse(test_mse)

    print("\nMetriche salvate in:")
    print(output_path)


if __name__ == "__main__":
    main()