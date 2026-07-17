import json

import matplotlib.pyplot as plt


def load_history(history_path):
    with open(history_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_figure(figure, output_path):
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_training_history(history_path, output_path):
    history = load_history(history_path)

    training_loss = history["loss"]
    validation_loss = history["val_loss"]

    epochs = range(1, len(training_loss) + 1)

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

    save_figure(
        figure=figure,
        output_path=output_path,
    )


def plot_reconstructions(
    clean_images,
    noisy_images,
    reconstructed_images,
    output_path,
    number_of_images=10,
):
    figure = plt.figure(figsize=(16, 6))

    for index in range(number_of_images):
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

    save_figure(
        figure=figure,
        output_path=output_path,
    )