import json

import matplotlib.pyplot as plt


def plot_training_history(history_path, output_path):

    # ========================================================
    # LETTURA DELLO STORICO
    # ========================================================

    with open(history_path, "r", encoding="utf-8") as file:
        history = json.load(file)

    training_loss = history["loss"]
    validation_loss = history["val_loss"]

    epochs = range(1, len(training_loss) + 1)

    # ========================================================
    # CREAZIONE DEL GRAFICO
    # ========================================================

    figure = plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        training_loss,
        label="Training MSE",
    )

    plt.plot(
        epochs,
        validation_loss,
        label="Validation MSE",
    )

    plt.xlabel("Epoca")
    plt.ylabel("Mean Squared Error")
    plt.title("Andamento della loss")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    # ========================================================
    # SALVATAGGIO
    # ========================================================

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_reconstructions(
    clean_images,
    noisy_images,
    reconstructed_images,
    output_path,
    number_of_images=10,
):
    figure = plt.figure(figsize=(16, 6))

    for index in range(number_of_images):

        # ====================================================
        # IMMAGINE PULITA
        # ====================================================

        axis = plt.subplot(
            3,
            number_of_images,
            index + 1,
        )

        plt.imshow(
            clean_images[index].squeeze(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        plt.axis("off")

        if index == 0:
            axis.set_ylabel("Pulita")

        # ====================================================
        # IMMAGINE RUMOROSA
        # ====================================================

        axis = plt.subplot(
            3,
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

        # ====================================================
        # IMMAGINE RICOSTRUITA
        # ====================================================

        axis = plt.subplot(
            3,
            number_of_images,
            index + 1 + 2 * number_of_images,
        )

        plt.imshow(
            reconstructed_images[index].squeeze(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        plt.axis("off")

        if index == 0:
            axis.set_ylabel("Ricostruita")

    plt.tight_layout()

    # ========================================================
    # SALVATAGGIO
    # ========================================================

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)