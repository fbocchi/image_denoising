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
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def normalize_images(images):
    images = images.astype("float32")
    images /= 255.0
    return np.expand_dims(images, axis=-1)


def add_gaussian_noise(images, standard_deviation, seed):
    random_generator = np.random.default_rng(seed)

    noise = random_generator.normal(
        loc=0.0,
        scale=standard_deviation,
        size=images.shape,
    )

    noisy_images = images + noise
    noisy_images = np.clip(noisy_images, 0.0, 1.0)

    return noisy_images.astype("float32")


def load_dataset():
    return keras.datasets.fashion_mnist.load_data()


def split_dataset(x_train_full, validation_size):
    split_index = len(x_train_full) - validation_size
    x_train = x_train_full[:split_index]
    x_validation = x_train_full[split_index:]
    return x_train, x_validation


def save_array(filename, array):
    output_path = PROCESSED_DATA_DIR / filename
    np.save(output_path, array)
    return output_path


def save_dataset(
    x_train,
    x_train_noisy,
    x_validation,
    x_validation_noisy,
    x_test,
    x_test_noisy,
):
    return [
        save_array("x_train_clean.npy", x_train),
        save_array("x_train_noisy.npy", x_train_noisy),
        save_array("x_validation_clean.npy", x_validation),
        save_array("x_validation_noisy.npy", x_validation_noisy),
        save_array("x_test_clean.npy", x_test),
        save_array("x_test_noisy.npy", x_test_noisy),
    ]


def save_noise_examples(
    clean_images,
    noisy_images,
    number_of_images=10,
):
    figure = plt.figure(figsize=(15, 4))

    for index in range(number_of_images):
        axis = plt.subplot(2, number_of_images, index + 1)

        plt.imshow(
            clean_images[index].squeeze(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        plt.axis("off")

        if index == 0:
            axis.set_ylabel("Pulita")

        axis = plt.subplot(
            2,
            number_of_images,
            index + 1 + number_of_images,
        )

        plt.imshow(
            noisy_images[index].squeeze(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        plt.axis("off")

        if index == 0:
            axis.set_ylabel("Rumorosa")

    plt.tight_layout()

    output_path = FIGURE_DIR / "noise_examples.png"

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def main():
    config = load_config()

    seed = config["seed"]
    validation_size = config["data"]["validation_size"]
    noise_standard_deviation = config["noise"]["standard_deviation"]

    set_seed(seed)
    create_directories()

    print("\nCaricamento del dataset...")

    (x_train_full, _), (x_test, _) = load_dataset()

    print("\nDimensioni originali:")
    print("Training completo:", x_train_full.shape)
    print("Test:", x_test.shape)

    x_train_full = normalize_images(x_train_full)
    x_test = normalize_images(x_test)

    x_train, x_validation = split_dataset(
        x_train_full,
        validation_size,
    )

    x_train_noisy = add_gaussian_noise(
        images=x_train,
        standard_deviation=noise_standard_deviation,
        seed=seed,
    )

    x_validation_noisy = add_gaussian_noise(
        images=x_validation,
        standard_deviation=noise_standard_deviation,
        seed=seed + 1,
    )

    x_test_noisy = add_gaussian_noise(
        images=x_test,
        standard_deviation=noise_standard_deviation,
        seed=seed + 2,
    )

    print("\nSalvataggio degli array...")

    saved_files = save_dataset(
        x_train,
        x_train_noisy,
        x_validation,
        x_validation_noisy,
        x_test,
        x_test_noisy,
    )

    for path in saved_files:
        print(path)

    print("\nGenerazione della figura...")

    figure_path = save_noise_examples(
        clean_images=x_train,
        noisy_images=x_train_noisy,
    )

    print(figure_path)

    print("\nPreprocessing completato.")

    print("\nDimensioni finali:")
    print("Training:", x_train.shape)
    print("Validation:", x_validation.shape)
    print("Test:", x_test.shape)

    print("\nIntervalli:")
    print("Training pulito:", x_train.min(), x_train.max())
    print(
        "Training rumoroso:",
        x_train_noisy.min(),
        x_train_noisy.max(),
    )


if __name__ == "__main__":
    main()