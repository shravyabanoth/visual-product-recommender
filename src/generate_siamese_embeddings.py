"""
generate_siamese_embeddings.py

Generates embeddings using the trained Siamese embedding model and
saves them (with ids, filepaths, and labels) to embeddings/siamese_embeddings.npz.

Usage:
    python generate_siamese_embeddings.py
"""

from feature_extraction import extract_embeddings, save_embeddings

METADATA_PATH = "data/subset/subset_metadata.csv"
WEIGHTS_PATH = "models/siamese_embedding_model.keras"
OUTPUT_PATH = "embeddings/siamese_embeddings.npz"


def main():
    print("=" * 60)
    print("GENERATING SIAMESE EMBEDDINGS")
    print("=" * 60)

    ids, filepaths, labels, embeddings = extract_embeddings(
        metadata_csv=METADATA_PATH,
        weights_path=WEIGHTS_PATH,
    )

    save_embeddings(
        output_path=OUTPUT_PATH,
        ids=ids,
        filepaths=filepaths,
        labels=labels,
        embeddings=embeddings,
    )


if __name__ == "__main__":
    main()