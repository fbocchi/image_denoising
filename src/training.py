import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import load_config
from src.model import build_autoencoder
from src.paths import (
    FIGURE_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    create_directories,
)
from src.visualization import plot_training_history


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_array(filename):
    file_path = PROCESSED_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"File non trovato: {file_path}\n"
            "Eseguire prima:\n"
            "python -m src.preprocessing"
        )

    return np.load(file_path)


def load_training_data():
    x_train_clean = load_array("x_train_clean.npy")
    x_train_noisy = load_array("x_train_noisy.npy")

    x_validation_clean = load_array("x_validation_clean.npy")
    x_validation_noisy = load_array("x_validation_noisy.npy")

    return (
        x_train_clean,
        x_train_noisy,
        x_validation_clean,
        x_validation_noisy,
    )


def compile_model(model):
    optimizer = keras.optimizers.Adam()

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
    )


def create_callbacks(
    early_stopping_patience,
    reduce_lr_patience,
):
    best_model_path = MODEL_DIR / "best_model.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]

    return callbacks, best_model_path


def train_model(
    model,
    x_train_clean,
    x_train_noisy,
    x_validation_clean,
    x_validation_noisy,
    maximum_epochs,
    batch_size,
    callbacks,
):
    return model.fit(
        x=x_train_noisy,
        y=x_train_clean,
        validation_data=(
            x_validation_noisy,
            x_validation_clean,
        ),
        epochs=maximum_epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )


def save_model(model):
    output_path = MODEL_DIR / "final_model.keras"

    model.save(output_path)

    return output_path


def convert_history_to_json(history):
    converted_history = {}

    for metric_name, values in history.history.items():
        converted_history[metric_name] = [
            float(value)
            for value in values
        ]

    return converted_history


def save_history(history):
    history_dictionary = convert_history_to_json(history)

    output_path = METRICS_DIR / "training_history.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            history_dictionary,
            file,
            indent=4,
        )

    return output_path


def save_training_history_figure(history_path):
    output_path = FIGURE_DIR / "training_history.png"

    plot_training_history(
        history_path=history_path,
        output_path=output_path,
    )

    return output_path


def summarize_training(history):
    best_epoch = (
        np.argmin(history.history["val_loss"]) + 1
    )

    best_validation_loss = np.min(
        history.history["val_loss"]
    )

    return (
        best_epoch,
        best_validation_loss,
    )


def main():
    config = load_config()

    seed = config["seed"]
    batch_size = config["training"]["batch_size"]
    maximum_epochs = config["training"]["maximum_epochs"]
    learning_rate = config["training"]["learning_rate"]
    early_stopping_patience = config["training"]["early_stopping_patience"]
    reduce_lr_patience = config["training"]["reduce_lr_patience"]

    set_seed(seed)
    create_directories()

    print("\nCaricamento dei dati...")

    (
        x_train_clean,
        x_train_noisy,
        x_validation_clean,
        x_validation_noisy,
    ) = load_training_data()

    print("Training pulito:", x_train_clean.shape)
    print("Training rumoroso:", x_train_noisy.shape)

    print("Validation pulito:", x_validation_clean.shape)
    print("Validation rumoroso:", x_validation_noisy.shape)

    print("\nCostruzione del modello...")

    model = build_autoencoder(config)

    model.summary()

    compile_model(model=model)

    callbacks, best_model_path = create_callbacks(
        early_stopping_patience=early_stopping_patience,
        reduce_lr_patience=reduce_lr_patience,
    )

    print("\nInizio del training...")

    history = train_model(
        model=model,
        x_train_clean=x_train_clean,
        x_train_noisy=x_train_noisy,
        x_validation_clean=x_validation_clean,
        x_validation_noisy=x_validation_noisy,
        maximum_epochs=maximum_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    final_model_path = save_model(model)

    history_path = save_history(history)

    history_figure_path = save_training_history_figure(
        history_path=history_path,
    )

    best_epoch, best_validation_loss = summarize_training(history)

    print("\nTraining completato.")
    print(f"Epoca migliore: {best_epoch}")
    print(f"Validation loss migliore: {best_validation_loss:.6f}")

    print("\nModello migliore salvato in:")
    print(best_model_path)

    print("\nModello finale salvato in:")
    print(final_model_path)

    print("\nStorico salvato in:")
    print(history_path)

    print("\nGrafico del training salvato in:")
    print(history_figure_path)


if __name__ == "__main__":
    main()