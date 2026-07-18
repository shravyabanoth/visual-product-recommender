"""
scripts/train_siamese.py

Step 2. Trains the Siamese network with triplet loss on the TRAIN split only
(data/subset/train_metadata.csv). Test-split images are never used here,
so retrieval evaluation later reflects genuine generalization.

After training, embeddings are still extracted for the FULL catalog
(data/subset/subset_metadata.csv) so the app can search across all products
-- the model just never trained on the test portion of that catalog.

Run: python scripts/train_siamese.py
(Slow on CPU with a small subset -- reduce EPOCHS/STEPS_PER_EPOCH if needed.)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from src.data_loader import make_triplets
from src.feature_extraction import load_image, extract_embeddings, save_embeddings
from src.siamese_model import build_embedding_model, build_siamese_triplet_model, triplet_loss

EPOCHS = 10
STEPS_PER_EPOCH = 100
BATCH_SIZE = 16


def triplet_generator(triplets, batch_size=16):
    while True:
        batch = [triplets[i] for i in np.random.choice(len(triplets), batch_size, replace=False)]
        anchors = np.stack([load_image(a, resnet_preprocess) for a, p, n in batch])
        positives = np.stack([load_image(p, resnet_preprocess) for a, p, n in batch])
        negatives = np.stack([load_image(n, resnet_preprocess) for a, p, n in batch])
        dummy_y = np.zeros((batch_size, 1))
        yield (anchors, positives, negatives), dummy_y


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)

    # TRAIN split only -- test images are never used to build triplets.
    train_df = pd.read_csv("data/subset/train_metadata.csv")
    triplets = make_triplets(train_df, n_triplets=3000)
    print(f"Generated {len(triplets)} triplets from TRAIN split ({len(train_df)} images)")

    embedding_model = build_embedding_model()
    siamese_model = build_siamese_triplet_model(embedding_model)
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=triplet_loss(margin=0.3))
    siamese_model.summary()

    gen = triplet_generator(triplets, batch_size=BATCH_SIZE)
    history = siamese_model.fit(gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)

    plt.plot(history.history["loss"])
    plt.title("Triplet Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/triplet_loss.png")
    print("Saved loss curve to results/triplet_loss.png")

    embedding_model.save("models/siamese_embedding_model.keras")
    print("Saved trained embedding model to models/siamese_embedding_model.keras")

    # Extract embeddings for the FULL catalog (train + test) so the app can
    # search across everything. The model itself only ever trained on the
    # train portion, so test images here are genuinely unseen.
    ids, filepaths, labels, embeds = extract_embeddings(
        metadata_csv="data/subset/subset_metadata.csv",
        weights_path="models/siamese_embedding_model.keras",
    )

    save_embeddings(
        output_path="embeddings/siamese_embeddings.npz",
        ids=ids,
        filepaths=filepaths,
        labels=labels,
        embeddings=embeds,
    )