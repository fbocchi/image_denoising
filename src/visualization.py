import json

import matplotlib.pyplot as plt


def plot_training_history(
    history_path,
    output_path,
):
    """
    Legge lo storico del training da un file JSON
    e salva il grafico della MSE.

    Parameters
    ----------
    history_path : Path
        Percorso del file training_history.json.

    output_path : Path
        Percorso in cui salvare il grafico.
    """

    # ========================================================
    # LETTURA DELLO STORICO
    # ========================================================

    with open(
        history_path,
        mode="r",
        encoding="utf-8",
    ) as file:

        history = json.load(file)


    # La loss è la MSE perché il modello
    # è stato compilato con:
    #
    # loss="mse"

    training_loss = history["loss"]

    validation_loss = history["val_loss"]


    # Le epoche partono da 1, non da 0.

    epochs = range(
        1,
        len(training_loss) + 1,
    )


    # ========================================================
    # CREAZIONE DEL GRAFICO
    # ========================================================

    figure = plt.figure(
        figsize=(9, 5)
    )


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


    plt.xlabel(
        "Epoca"
    )


    plt.ylabel(
        "Mean Squared Error"
    )


    plt.title(
        "Andamento della loss"
    )


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
    """
    Salva un confronto visivo tra:

    riga 1:
        immagini originali pulite

    riga 2:
        immagini rumorose

    riga 3:
        immagini ricostruite dal modello

    Parameters
    ----------
    clean_images : numpy.ndarray
        Immagini originali pulite.

    noisy_images : numpy.ndarray
        Immagini corrotte dal rumore.

    reconstructed_images : numpy.ndarray
        Immagini prodotte dall'autoencoder.

    output_path : Path
        Percorso della figura da salvare.

    number_of_images : int
        Numero di immagini da visualizzare.
    """

    figure = plt.figure(
        figsize=(16, 6)
    )


    for index in range(
        number_of_images
    ):


        # ====================================================
        # IMMAGINE PULITA
        # ====================================================

        axis = plt.subplot(
            3,
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


        # ====================================================
        # IMMAGINE RUMOROSA
        # ====================================================

        axis = plt.subplot(
            3,
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


        # ====================================================
        # IMMAGINE RICOSTRUITA
        # ====================================================

        axis = plt.subplot(
            3,
            number_of_images,
            index
            + 1
            + 2 * number_of_images,
        )


        plt.imshow(
            reconstructed_images[
                index
            ].squeeze(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )


        plt.axis("off")


        if index == 0:

            axis.set_ylabel(
                "Ricostruita"
            )


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