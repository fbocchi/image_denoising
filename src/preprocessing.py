import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras

from src.config import load_config

from src.paths import (
    FIGURE_DIR,
    PROCESSED_DATA_DIR,
    create_directories,
)


def set_seed(seed):
    """
    Imposta i seed per rendere
    l'esperimento riproducibile.
    """

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)


def normalize_images(images):
    """
    Converte i pixel da [0, 255]
    all'intervallo [0, 1].

    Aggiunge inoltre la dimensione
    corrispondente al canale.

    Input:
        (N, 28, 28)

    Output:
        (N, 28, 28, 1)
    """

    images = images.astype("float32")

    images = images / 255.0

    images = np.expand_dims(
        images,
        axis=-1,
    )

    return images


def add_gaussian_noise(
    images,
    standard_deviation,
    seed,
):
    """
    Aggiunge rumore gaussiano:

        noisy = clean + noise

    con:

        noise ~ N(0, sigma^2)

    I valori finali vengono limitati
    all'intervallo [0, 1].
    """

    random_generator = (
        np.random.default_rng(seed)
    )

    noise = random_generator.normal(
        loc=0.0,
        scale=standard_deviation,
        size=images.shape,
    )

    noisy_images = images + noise

    noisy_images = np.clip(
        noisy_images,
        0.0,
        1.0,
    )

    return noisy_images.astype(
        "float32"
    )


def save_array(
    filename,
    array,
):
    """
    Salva un array NumPy nella directory
    data/processed.
    """

    output_path = (
        PROCESSED_DATA_DIR
        / filename
    )

    np.save(
        output_path,
        array,
    )

    print(
        f"Salvato: {output_path}"
    )


def save_noise_examples(
    clean_images,
    noisy_images,
    number_of_images=10,
):
    """
    Salva una figura contenente:

    riga 1:
        immagini pulite

    riga 2:
        immagini rumorose
    """

    figure = plt.figure(
        figsize=(15, 4)
    )

    for index in range(
        number_of_images
    ):

        # Immagine pulita

        axis = plt.subplot(
            2,
            number_of_images,
            index + 1,
        )

        plt.imshow(
            clean_images[
                index
            ].squeeze(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        plt.axis("off")

        if index == 0:

            axis.set_ylabel(
                "Pulita"
            )

        # Immagine rumorosa

        axis = plt.subplot(
            2,
            number_of_images,
            index
            + 1
            + number_of_images,
        )

        plt.imshow(
            noisy_images[
                index
            ].squeeze(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        plt.axis("off")

        if index == 0:

            axis.set_ylabel(
                "Rumorosa"
            )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "noise_examples.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Figura salvata: {output_path}"
    )


def main():

    # ========================================================
    # CONFIGURAZIONE
    # ========================================================

    config = load_config()

    seed = config["seed"]

    validation_size = (
        config["data"]
        ["validation_size"]
    )

    noise_standard_deviation = (
        config["noise"]
        ["standard_deviation"]
    )

    set_seed(seed)

    create_directories()

    # ========================================================
    # CARICAMENTO DI FASHION-MNIST
    # ========================================================

    (
        x_train_full,
        _,
    ), (
        x_test,
        _,
    ) = (
        keras.datasets
        .fashion_mnist
        .load_data()
    )

    print(
        "\nDimensioni originali:"
    )

    print(
        "Training completo:",
        x_train_full.shape,
    )

    print(
        "Test:",
        x_test.shape,
    )

    # ========================================================
    # NORMALIZZAZIONE
    # ========================================================

    x_train_full = normalize_images(
        x_train_full
    )

    x_test = normalize_images(
        x_test
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    split_index = (
        len(x_train_full)
        - validation_size
    )

    x_train = (
        x_train_full[
            :split_index
        ]
    )

    x_validation = (
        x_train_full[
            split_index:
        ]
    )

    # ========================================================
    # AGGIUNTA DEL RUMORE
    # ========================================================

    x_train_noisy = (
        add_gaussian_noise(
            images=x_train,
            standard_deviation=(
                noise_standard_deviation
            ),
            seed=seed,
        )
    )

    x_validation_noisy = (
        add_gaussian_noise(
            images=x_validation,
            standard_deviation=(
                noise_standard_deviation
            ),
            seed=seed + 1,
        )
    )

    x_test_noisy = (
        add_gaussian_noise(
            images=x_test,
            standard_deviation=(
                noise_standard_deviation
            ),
            seed=seed + 2,
        )
    )

    # ========================================================
    # SALVATAGGIO
    # ========================================================

    save_array(
        "x_train_clean.npy",
        x_train,
    )

    save_array(
        "x_train_noisy.npy",
        x_train_noisy,
    )

    save_array(
        "x_validation_clean.npy",
        x_validation,
    )

    save_array(
        "x_validation_noisy.npy",
        x_validation_noisy,
    )

    save_array(
        "x_test_clean.npy",
        x_test,
    )

    save_array(
        "x_test_noisy.npy",
        x_test_noisy,
    )

    # ========================================================
    # FIGURA DI CONTROLLO
    # ========================================================

    save_noise_examples(
        clean_images=x_train,
        noisy_images=x_train_noisy,
    )

    # ========================================================
    # RIEPILOGO
    # ========================================================

    print(
        "\nPreprocessing completato."
    )

    print(
        "\nDimensioni finali:"
    )

    print(
        "Training:",
        x_train.shape,
    )

    print(
        "Validation:",
        x_validation.shape,
    )

    print(
        "Test:",
        x_test.shape,
    )

    print(
        "\nIntervalli:"
    )

    print(
        "Training pulito:",
        x_train.min(),
        x_train.max(),
    )

    print(
        "Training rumoroso:",
        x_train_noisy.min(),
        x_train_noisy.max(),
    )


if __name__ == "__main__":

    main()