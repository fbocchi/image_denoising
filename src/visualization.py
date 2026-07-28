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


def plot_training_history_2(history_path, output_path):

    history = load_history(history_path)

    training_loss = history["loss"]
    validation_loss = history["val_loss"]

    epochs = range(1, len(training_loss) + 1)

    figure, ax = plt.subplots(figsize=(10, 4.8))

    ax.plot(
        epochs,
        training_loss,
        linewidth=2.5,
        label="Training MSE",
    )

    ax.plot(
        epochs,
        validation_loss,
        linewidth=2.5,
        linestyle="--",
        label="Validation MSE",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Training History")

    ax.set_xticks(epochs)
    ax.set_xlim(1, len(training_loss))

    minimum = min(
        min(training_loss),
        min(validation_loss),
    )

    maximum = max(
        max(training_loss),
        max(validation_loss),
    )

    # piccolo margine verticale
    margin = 0.02 * (maximum - minimum)

    ax.set_ylim(
        minimum - margin,
        maximum + margin,
    )

    ax.grid(
        linestyle="--",
        alpha=0.3,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="upper right",
    )

    figure.tight_layout()

    save_figure(
        figure=figure,
        output_path=output_path,
    )


def plot_training_history_1(history_path, output_path):

    history = load_history(history_path)

    training_loss = history["loss"]
    validation_loss = history["val_loss"]

    epochs = range(1, len(training_loss) + 1)

    figure, ax = plt.subplots(figsize=(8.5, 5))

    ax.plot(
        epochs,
        training_loss,
        marker="o",
        markersize=4,
        linewidth=2,
        label="Training MSE",
    )

    ax.plot(
        epochs,
        validation_loss,
        marker="s",
        markersize=4,
        linewidth=2,
        linestyle="--",
        label="Validation MSE",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Training History")

    ax.set_xticks(range(1, len(training_loss) + 1))

    minimum = min(
        min(training_loss),
        min(validation_loss),
    )

    maximum = max(
        max(training_loss),
        max(validation_loss),
    )

    margin = 0.05 * (maximum - minimum)

    ax.set_ylim(
        minimum - margin,
        maximum + margin,
    )

    ax.grid(
        linestyle="--",
        alpha=0.3,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="upper right",
    )

    figure.tight_layout()

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