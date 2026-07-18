"""
generate_transfer_embeddings.py

Generates embeddings using the fine-tuned transfer-learning model
(classification head stripped off) and saves them, with ids, filepaths,
and labels, to embeddings/transfer_embeddings.npz.

Usage:
    python src/generate_transfer_embeddings.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

IMG_SIZE = (224, 224)

MODEL_PATH = "models/transfer_learning_model.keras"
METADATA_PATH = "data/subset/subset_metadata.csv"
OUTPUT_PATH = "embeddings/transfer_embeddings.npz"


def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-3].output
    )

    return feature_model


def load_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img


def main():

    print("=" * 60)
    print("GENERATING TRANSFER LEARNING EMBEDDINGS")
    print("=" * 60)

    df = pd.read_csv(METADATA_PATH)

    ids = df["id"].values
    labels = df["articleType"].values
    filepaths = df["filepath"].values

    model = load_model()

    embeddings = []

    for path in tqdm(filepaths):

        image = load_image(path)

        image = np.expand_dims(image, axis=0)

        embedding = model.predict(image, verbose=0)[0]

        embeddings.append(embedding)

    embeddings = np.array(embeddings)

    os.makedirs("embeddings", exist_ok=True)

    np.savez(
        OUTPUT_PATH,
        ids=ids,
        filepaths=filepaths,
        embeddings=embeddings,
        labels=labels,
    )

    print("\n")
    print("=" * 60)
    print("TRANSFER EMBEDDINGS GENERATED")
    print("=" * 60)
    print(f"Images Processed : {len(ids)}")
    print(f"Embedding Dimension : {embeddings.shape[1]}")
    print(f"Saved File : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()