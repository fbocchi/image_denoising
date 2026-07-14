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
    """
    Carica un array dalla directory:

        data/processed/
    """

    file_path = (
        PROCESSED_DATA_DIR
        / filename
    )


    if not file_path.exists():

        raise FileNotFoundError(
            f"File non trovato: {file_path}\n"
            "Eseguire prima:\n"
            "python -m src.preprocessing"
        )


    return np.load(
        file_path
    )


def main():

    # ========================================================
    # CREAZIONE DELLE DIRECTORY
    # ========================================================

    create_directories()


    # ========================================================
    # CARICAMENTO DEI DATI
    # ========================================================

    print(
        "\nCaricamento delle immagini..."
    )


    x_test_clean = load_array(
        "x_test_clean.npy"
    )


    x_test_noisy = load_array(
        "x_test_noisy.npy"
    )


    # ========================================================
    # CARICAMENTO DEL MODELLO
    # ========================================================

    model_path = (
        MODEL_DIR
        / "best_model.keras"
    )


    if not model_path.exists():

        raise FileNotFoundError(
            f"Modello non trovato: {model_path}\n"
            "Eseguire prima:\n"
            "python -m src.training"
        )


    print(
        "\nCaricamento del modello:"
    )


    print(
        model_path
    )


    model = keras.models.load_model(
        model_path
    )


    # ========================================================
    # GENERAZIONE DELLE RICOSTRUZIONI
    # ========================================================

    print(
        "\nGenerazione delle ricostruzioni..."
    )


    reconstructed_images = model.predict(

        x=x_test_noisy,

        batch_size=128,

        verbose=1,

    )


    # ========================================================
    # SALVATAGGIO DELLE RICOSTRUZIONI
    # ========================================================

    predictions_path = (

        PREDICTION_DIR

        / "reconstructed_test.npy"

    )


    np.save(

        predictions_path,

        reconstructed_images,

    )


    print(
        "\nRicostruzioni salvate in:"
    )


    print(
        predictions_path
    )


    # ========================================================
    # CONFRONTO VISIVO
    # ========================================================

    reconstruction_figure_path = (

        FIGURE_DIR

        / "reconstructions.png"

    )


    plot_reconstructions(

        clean_images=(
            x_test_clean
        ),

        noisy_images=(
            x_test_noisy
        ),

        reconstructed_images=(
            reconstructed_images
        ),

        output_path=(
            reconstruction_figure_path
        ),

        number_of_images=10,

    )


    print(
        "\nFigura delle ricostruzioni salvata in:"
    )


    print(
        reconstruction_figure_path
    )


    # ========================================================
    # GRAFICO DEL TRAINING
    # ========================================================

    history_path = (

        METRICS_DIR

        / "training_history.json"

    )


    if history_path.exists():

        training_figure_path = (

            FIGURE_DIR

            / "training_history.png"

        )


        plot_training_history(

            history_path=(
                history_path
            ),

            output_path=(
                training_figure_path
            ),

        )


        print(
            "\nGrafico del training salvato in:"
        )


        print(
            training_figure_path
        )


    else:

        print(
            "\nStorico del training non trovato."
        )


    print(
        "\nGenerazione completata."
    )


if __name__ == "__main__":

    main()