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
    # CARICAMENTO DEL TEST SET
    # ========================================================

    print(
        "\nCaricamento del test set..."
    )


    x_test_clean = load_array(
        "x_test_clean.npy"
    )


    x_test_noisy = load_array(
        "x_test_noisy.npy"
    )


    print(
        "Test pulito:",
        x_test_clean.shape,
    )


    print(
        "Test rumoroso:",
        x_test_noisy.shape,
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
    # VALUTAZIONE
    # ========================================================

    print(
        "\nValutazione sul test set..."
    )


    test_mse = model.evaluate(

        x=x_test_noisy,

        y=x_test_clean,

        verbose=1,

    )


    # Con una sola loss e nessuna metrica,
    # evaluate restituisce un singolo valore.

    test_mse = float(
        test_mse
    )


    # ========================================================
    # VISUALIZZAZIONE DEL RISULTATO
    # ========================================================

    print(
        "\nRisultato finale:"
    )


    print(
        f"Test MSE: {test_mse:.6f}"
    )


    # ========================================================
    # SALVATAGGIO DEL RISULTATO
    # ========================================================

    results = {
        "test_mse": test_mse
    }


    output_path = (
        METRICS_DIR
        / "test_metrics.json"
    )


    with open(

        output_path,

        mode="w",

        encoding="utf-8",

    ) as file:

        json.dump(

            results,

            file,

            indent=4,

        )


    print(
        "\nMetriche salvate in:"
    )


    print(
        output_path
    )


if __name__ == "__main__":

    main()