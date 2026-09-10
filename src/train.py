"""
Trains a CNN (from scratch) on the CLAHE-preprocessed, grayscale steel
surface defect dataset (NEU-style: crazing, inclusion, patches,
pitted_surface, rolled-in_scale, scratches).
"""

import keras
import tensorflow as tf

DATA_DIR = "Data/train_clahe"
IMAGE_SIZE = (200, 200)
BATCH_SIZE = 16
SEED = 42


def load_data():
    train_data = keras.utils.image_dataset_from_directory(
        directory=DATA_DIR,
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset="training",
    )
    valid_data = keras.utils.image_dataset_from_directory(
        directory=DATA_DIR,
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset="validation",
    )
    return train_data, valid_data


def build_model(num_classes: int = 6):
    return keras.Sequential([
        keras.layers.Input((200, 200, 1)),
        keras.layers.Rescaling(1. / 255),
        keras.layers.Conv2D(32, padding="same", activation="relu", kernel_size=(3, 3)),
        keras.layers.MaxPool2D(strides=2),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(64, padding="same", activation="relu", kernel_size=(3, 3)),
        keras.layers.Dropout(0.2),
        keras.layers.MaxPool2D(strides=2),
        keras.layers.Conv2D(128, padding="same", activation="relu", kernel_size=(3, 3)),
        keras.layers.MaxPool2D(strides=2),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(num_classes, activation=keras.activations.softmax),
    ])


def main():
    train_data, valid_data = load_data()
    model = build_model()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3
    )

    model.fit(
        train_data,
        epochs=20,
        verbose=2,
        validation_data=valid_data,
        callbacks=[early_stopping, reduce_lr],
    )

    model.save("models/model.keras")
    return model


if __name__ == "__main__":
    main()
