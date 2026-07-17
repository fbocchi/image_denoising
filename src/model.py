from tensorflow import keras
from tensorflow.keras import layers


def build_encoder(config):

    first_filters = config["model"]["first_filters"]
    second_filters = config["model"]["second_filters"]
    kernel_size = config["model"]["kernel_size"]

    inputs = keras.Input(
        shape=(28, 28, 1),
        name="encoder_input",
    )

    # 28 x 28 x 1
    # ->
    # 28 x 28 x first_filters

    x = layers.Conv2D(
        filters=first_filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        name="encoder_conv_1",
    )(inputs)

    # 28 x 28 x first_filters
    # ->
    # 14 x 14 x first_filters

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        padding="same",
        name="encoder_pool_1",
    )(x)

    # 14 x 14 x first_filters
    # ->
    # 14 x 14 x second_filters

    x = layers.Conv2D(
        filters=second_filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        name="encoder_conv_2",
    )(x)

    # 14 x 14 x second_filters
    # ->
    # 7 x 7 x second_filters

    latent = layers.MaxPooling2D(
        pool_size=(2, 2),
        padding="same",
        name="latent_representation",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=latent,
        name="encoder",
    )

def build_decoder(config):

    first_filters = config["model"]["first_filters"]
    second_filters = config["model"]["second_filters"]
    kernel_size = config["model"]["kernel_size"]

    latent_inputs = keras.Input(
        shape=(7, 7, second_filters),
        name="latent_input",
    )

    # 7 x 7 x second_filters
    # ->
    # 14 x 14 x second_filters

    x = layers.Conv2DTranspose(
        filters=second_filters,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding="same",
        activation="relu",
        name="decoder_transposed_conv_1",
    )(latent_inputs)

    # 14 x 14 x second_filters
    # ->
    # 28 x 28 x first_filters

    x = layers.Conv2DTranspose(
        filters=first_filters,
        kernel_size=kernel_size,
        strides=(2, 2),
        padding="same",
        activation="relu",
        name="decoder_transposed_conv_2",
    )(x)

    # 28 x 28 x first_filters
    # ->
    # 28 x 28 x 1

    outputs = layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        padding="same",
        activation="sigmoid",
        name="reconstructed_image",
    )(x)

    return keras.Model(
        inputs=latent_inputs,
        outputs=outputs,
        name="decoder",
    )

def build_autoencoder(config):

    encoder = build_encoder(config)
    decoder = build_decoder(config)

    inputs = keras.Input(
        shape=(28, 28, 1),
        name="noisy_image",
    )

    latent = encoder(inputs)

    outputs = decoder(latent)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="convolutional_denoising_autoencoder",
    )


if __name__ == "__main__":
    from src.config import load_config

    config = load_config()

    model = build_autoencoder(config)

    model.summary()