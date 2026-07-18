"""
feature_extraction.py

Extracts image embeddings using a pretrained backbone (ResNet50 / EfficientNetB0)
or a custom trained embedding model (e.g. a Siamese embedding model).

Usage:
    python feature_extraction.py --metadata data/subset/subset_metadata.csv \
        --out embeddings/baseline_embeddings.npz --model resnet50

    # Using a custom trained model's weights:
    python feature_extraction.py --weights models/siamese_embedding_model.keras \
        --out embeddings/transfer_embeddings.npz
"""

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

IMG_SIZE = (224, 224)


def build_feature_extractor(model_name="resnet50", weights_path=None):
    """
    Builds (or loads) the model used to produce embeddings.

    If `weights_path` is given, loads a full trained model (e.g. a Siamese
    embedding model) instead of building one from scratch. Note: this branch
    currently always pairs the loaded model with `resnet_preprocess` — update
    this if your custom model was trained with a different preprocessing fn.
    """
    if weights_path:
        model = tf.keras.models.load_model(
            weights_path, compile=False, safe_mode=False
        )
        return model, resnet_preprocess

    if model_name == "resnet50":
        model = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3),
        )
        preprocess_fn = resnet_preprocess

    elif model_name == "efficientnet":
        model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3),
        )
        preprocess_fn = effnet_preprocess

    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return model, preprocess_fn


def load_image(path, preprocess_fn):
    """Loads a single image from disk and applies the given preprocessing fn."""
    image = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    image = tf.keras.utils.img_to_array(image)
    image = preprocess_fn(image)
    return image


def extract_embeddings(metadata_csv, model_name="resnet50", weights_path=None, batch_size=32):
    """
    Reads image metadata, runs the feature extractor in batches, and
    returns (ids, filepaths, labels, embeddings).
    """
    df = pd.read_csv(metadata_csv)

    model, preprocess_fn = build_feature_extractor(
        model_name=model_name,
        weights_path=weights_path,
    )

    ids = df["id"].values
    filepaths = df["filepath"].values
    labels = df["articleType"].values

    embeddings = []

    for i in tqdm(range(0, len(filepaths), batch_size), desc="Extracting Embeddings"):
        batch_paths = filepaths[i : i + batch_size]

        batch_images = np.stack(
            [load_image(path, preprocess_fn) for path in batch_paths]
        )

        batch_embeddings = model.predict(batch_images, verbose=0)
        embeddings.append(batch_embeddings)

    embeddings = np.concatenate(embeddings, axis=0)

    return ids, filepaths, labels, embeddings


def save_embeddings(output_path, ids, filepaths, labels, embeddings):
    """Saves extracted embeddings + metadata to a .npz file."""
    np.savez(
        output_path,
        ids=ids,
        filepaths=filepaths,
        labels=labels,
        embeddings=embeddings,
    )

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 60)
    print(f"Images Processed     : {len(ids)}")
    print(f"Embedding Dimension  : {embeddings.shape[1]}")
    print(f"Saved File           : {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/subset/subset_metadata.csv")
    parser.add_argument("--out", default="embeddings/baseline_embeddings.npz")
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "efficientnet"])
    parser.add_argument("--weights", default=None)

    args = parser.parse_args()

    ids, filepaths, labels, embeddings = extract_embeddings(
        metadata_csv=args.metadata,
        model_name=args.model,
        weights_path=args.weights,
    )

    save_embeddings(
        output_path=args.out,
        ids=ids,
        filepaths=filepaths,
        labels=labels,
        embeddings=embeddings,
    )


if __name__ == "__main__":
    main()