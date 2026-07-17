import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import load_config
from src.model import build_autoencoder
from src.paths import (
    METRICS_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    create_directories,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_array(filename):

    file_path = PROCESSED_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"File non trovato: {file_path}\n"
            "Eseguire prima il preprocessing:\n"
            "python -m src.preprocessing"
        )

    return np.load(file_path)


def convert_history_to_json(history):

    converted_history = {}

    for metric_name, values in history.history.items():
        converted_history[metric_name] = [
            float(value)
            for value in values
        ]

    return converted_history


def main():

    # ========================================================
    # CONFIGURAZIONE
    # ========================================================

    config = load_config()

    seed = config["seed"]
    batch_size = config["training"]["batch_size"]
    maximum_epochs = config["training"]["maximum_epochs"]
    learning_rate = config["training"]["learning_rate"]
    early_stopping_patience = config["training"]["early_stopping_patience"]
    reduce_lr_patience = config["training"]["reduce_lr_patience"]

    set_seed(seed)
    create_directories()

    # ========================================================
    # CARICAMENTO DEI DATI
    # ========================================================

    print("\nCaricamento dei dati...")

    x_train_clean = load_array("x_train_clean.npy")
    x_train_noisy = load_array("x_train_noisy.npy")

    x_validation_clean = load_array("x_validation_clean.npy")
    x_validation_noisy = load_array("x_validation_noisy.npy")

    print("Training pulito:", x_train_clean.shape)
    print("Training rumoroso:", x_train_noisy.shape)

    print("Validation pulito:", x_validation_clean.shape)
    print("Validation rumoroso:", x_validation_noisy.shape)

    # ========================================================
    # COSTRUZIONE DEL MODELLO
    # ========================================================

    model = build_autoencoder(config)

    model.summary()

    # ========================================================
    # COMPILAZIONE
    # ========================================================

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())

    # ========================================================
    # CALLBACK
    # ========================================================

    best_model_path = MODEL_DIR / "best_model.keras"

    # Salva il modello corrispondente alla
    # validation loss più bassa.

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    # Interrompe il training se la validation
    # loss non migliora per un certo numero
    # di epoche.

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )

    # Riduce il learning rate se la validation
    # loss smette temporaneamente di migliorare.

    reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=1e-6,
        verbose=1,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print("\nInizio del training...")

    history = model.fit(
        x=x_train_noisy,
        y=x_train_clean,

        validation_data=(
            x_validation_noisy,
            x_validation_clean,
        ),

        epochs=maximum_epochs,
        batch_size=batch_size,
        shuffle=True,

        callbacks=[
            model_checkpoint,
            early_stopping,
            reduce_learning_rate,
        ],

        verbose=1,
    )

    # ========================================================
    # SALVATAGGIO DEL MODELLO FINALE
    # ========================================================

    final_model_path = MODEL_DIR / "final_model.keras"

    model.save(final_model_path)

    # ========================================================
    # SALVATAGGIO DELLO STORICO
    # ========================================================

    history_dictionary = convert_history_to_json(history)

    history_path = METRICS_DIR / "training_history.json"

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(
            history_dictionary,
            file,
            indent=4,
        )

    # ========================================================
    # RIEPILOGO
    # ========================================================

    best_epoch = np.argmin(history.history["val_loss"]) + 1

    best_validation_loss = np.min(
        history.history["val_loss"]
    )

    print("\nTraining completato.")
    print(f"Epoca migliore: {best_epoch}")
    print(f"Validation loss migliore: {best_validation_loss:.6f}")

    print("\nModello migliore salvato in:")
    print(best_model_path)

    print("\nModello finale salvato in:")
    print(final_model_path)

    print("\nStorico salvato in:")
    print(history_path)


if __name__ == "__main__":
    main()