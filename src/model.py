from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(config):
    """
    Costruisce un convolutional denoising autoencoder.

    Encoder:
        28 x 28 x 1
        -> 28 x 28 x 32
        -> 14 x 14 x 32
        -> 14 x 14 x 64
        -> 7 x 7 x 64

    Decoder:
        7 x 7 x 64
        -> 14 x 14 x 64
        -> 28 x 28 x 32
        -> 28 x 28 x 1

    Parameters
    ----------
    config : dict
        Configurazione caricata da config.yaml.

    Returns
    -------
    keras.Model
        Il convolutional denoising autoencoder.
    """

    # ========================================================
    # PARAMETRI DEL MODELLO
    # ========================================================

    first_filters = config["model"]["first_filters"]

    second_filters = config["model"]["second_filters"]

    kernel_size = config["model"]["kernel_size"]


    # ========================================================
    # INPUT
    # ========================================================

    # Immagine Fashion-MNIST rumorosa:
    #
    # 28 x 28 pixel
    # 1 canale, perché è in scala di grigi.

    inputs = keras.Input(
        shape=(28, 28, 1),
        name="noisy_image",
    )


    # ========================================================
    # ENCODER
    # ========================================================

    # Input:
    #
    # 28 x 28 x 1
    #
    # Output:
    #
    # 28 x 28 x 32

    x = layers.Conv2D(
        filters=first_filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        name="encoder_conv_1",
    )(inputs)


    # Riduzione della risoluzione:
    #
    # 28 x 28 x 32
    #
    # ->
    #
    # 14 x 14 x 32

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        padding="same",
        name="encoder_pool_1",
    )(x)


    # Estrazione di feature più complesse:
    #
    # 14 x 14 x 32
    #
    # ->
    #
    # 14 x 14 x 64

    x = layers.Conv2D(
        filters=second_filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        name="encoder_conv_2",
    )(x)


    # Seconda riduzione della risoluzione:
    #
    # 14 x 14 x 64
    #
    # ->
    #
    # 7 x 7 x 64

    latent_representation = layers.MaxPooling2D(
        pool_size=(2, 2),
        padding="same",
        name="latent_representation",
    )(x)


    # ========================================================
    # DECODER
    # ========================================================

    # Aumento della risoluzione:
    #
    # 7 x 7 x 64
    #
    # ->
    #
    # 14 x 14 x 64

    x = layers.Conv2DTranspose(
        filters=second_filters,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding="same",
        activation="relu",
        name="decoder_transposed_conv_1",
    )(latent_representation)


    # Secondo aumento della risoluzione:
    #
    # 14 x 14 x 64
    #
    # ->
    #
    # 28 x 28 x 32

    x = layers.Conv2DTranspose(
        filters=first_filters,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding="same",
        activation="relu",
        name="decoder_transposed_conv_2",
    )(x)


    # ========================================================
    # OUTPUT
    # ========================================================

    # Ricostruzione dell'immagine pulita:
    #
    # 28 x 28 x 32
    #
    # ->
    #
    # 28 x 28 x 1
    #
    # L'attivazione sigmoid produce valori
    # nell'intervallo [0, 1], come le immagini
    # normalizzate.

    outputs = layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        padding="same",
        activation="sigmoid",
        name="reconstructed_image",
    )(x)


    # ========================================================
    # MODELLO COMPLETO
    # ========================================================

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="convolutional_denoising_autoencoder",
    )


    return model

if __name__ == "__main__":

    from src.config import load_config

    config = load_config()

    model = build_autoencoder(config)

    model.summary()