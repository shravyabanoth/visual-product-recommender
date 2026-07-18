import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image_dataset_from_directory

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 10
DATASET_PATH = "data/subset_train"
MODEL_PATH = "models/transfer_learning_model.keras"

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets():

    train_dataset = image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    validation_dataset = image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_dataset.class_names

    train_dataset = train_dataset.map(
        lambda x, y: (preprocess_input(x), y)
    )

    validation_dataset = validation_dataset.map(
        lambda x, y: (preprocess_input(x), y)
    )

    train_dataset = train_dataset.prefetch(AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(AUTOTUNE)

    return train_dataset, validation_dataset, class_names


def build_model(num_classes):

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        512,
        activation="relu",
    )(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(
        256,
        activation="relu",
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model
def train_model(model, base_model, train_dataset, validation_dataset):

    os.makedirs("models", exist_ok=True)

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    print("\n" + "=" * 60)
    print("STAGE 1 : TRAINING CLASSIFIER")
    print("=" * 60)

    history_stage1 = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS_STAGE1,
        callbacks=[checkpoint, early_stop],
    )

    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n" + "=" * 60)
    print("STAGE 2 : FINE TUNING")
    print("=" * 60)

    history_stage2 = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS_STAGE2,
        callbacks=[checkpoint, early_stop],
    )

    model.save(MODEL_PATH)

    print("\n" + "=" * 60)
    print("TRANSFER LEARNING COMPLETED")
    print("=" * 60)
    print(f"Model Saved : {MODEL_PATH}")
    print("=" * 60)

    return history_stage1, history_stage2


def main():

    print("\nLoading Dataset...\n")

    train_dataset, validation_dataset, class_names = load_datasets()

    print(f"Classes Found : {class_names}")

    model, base_model = build_model(len(class_names))

    model.summary()

    train_model(
        model,
        base_model,
        train_dataset,
        validation_dataset,
    )


if __name__ == "__main__":
    main()