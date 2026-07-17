import numpy as np
from tensorflow import keras

from src.paths import (
    FIGURE_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    PROCESSED_DATA_DIR,
    create_directories,
)
from src.visualization import (
    plot_reconstructions,
    plot_training_history,
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

    return keras.models.load_model(model_path), model_path


def reconstruct_images(model, x_test_noisy):
    return model.predict(
        x=x_test_noisy,
        batch_size=128,
        verbose=1,
    )


def save_reconstructions(reconstructed_images):
    output_path = PREDICTION_DIR / "reconstructed_test.npy"

    np.save(output_path, reconstructed_images)

    return output_path


def save_reconstruction_figure(
    x_test_clean,
    x_test_noisy,
    reconstructed_images,
):
    output_path = FIGURE_DIR / "reconstructions.png"

    plot_reconstructions(
        clean_images=x_test_clean,
        noisy_images=x_test_noisy,
        reconstructed_images=reconstructed_images,
        output_path=output_path,
        number_of_images=10,
    )

    return output_path


def main():
    create_directories()

    print("\nCaricamento delle immagini...")
    x_test_clean = load_array("x_test_clean.npy")
    x_test_noisy = load_array("x_test_noisy.npy")

    print("\nCaricamento del modello...")
    model, model_path = load_model()
    print(model_path)

    print("\nGenerazione delle ricostruzioni...")
    reconstructed_images = reconstruct_images(
        model=model,
        x_test_noisy=x_test_noisy,
    )

    predictions_path = save_reconstructions(reconstructed_images)

    print("\nRicostruzioni salvate in:")
    print(predictions_path)

    reconstruction_figure_path = save_reconstruction_figure(
        x_test_clean=x_test_clean,
        x_test_noisy=x_test_noisy,
        reconstructed_images=reconstructed_images,
    )

    print("\nFigura delle ricostruzioni salvata in:")
    print(reconstruction_figure_path)

    print("\nGenerazione completata.")


if __name__ == "__main__":
    main()