"""
scripts/run_evaluation.py

Step 11 (per project spec): Compares Baseline vs Transfer Learning vs Siamese
using held-out TEST-SPLIT images as queries against the full catalog. This
avoids the data leakage of evaluating on images the models were trained on.

Run: python scripts/run_evaluation.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.evaluate import precision_recall_at_k_held_out, measure_inference_time, plot_comparison

K = 5
TEST_METADATA_PATH = "data/subset/test_metadata.csv"

# (label, path to .npz)
EMBEDDING_SETS = [
    ("Baseline", "embeddings/baseline_embeddings.npz"),
    ("Transfer Learning", "embeddings/transfer_embeddings.npz"),
    ("Siamese", "embeddings/siamese_embeddings.npz"),
]


def evaluate_all():
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(TEST_METADATA_PATH):
        print(
            f"'{TEST_METADATA_PATH}' not found. Run scripts/prepare_data.py "
            "(updated version with train/test split) first."
        )
        return {}

    test_ids = set(pd.read_csv(TEST_METADATA_PATH)["id"].tolist())
    print(f"Using {len(test_ids)} held-out test images as queries.\n")

    results = {}

    for label, path in EMBEDDING_SETS:
        if not os.path.exists(path):
            print(f"[{label}] embeddings not found at '{path}' -- skipping.")
            continue

        data = np.load(path, allow_pickle=True)
        ids, embeddings, labels = data["ids"], data["embeddings"], data["labels"]

        query_mask = np.isin(ids, list(test_ids))
        if query_mask.sum() == 0:
            print(
                f"[{label}] none of this embeddings file's ids match the "
                "test split -- was it generated before rebuilding the "
                "train/test split? Regenerate it and retry."
            )
            continue

        precision, recall = precision_recall_at_k_held_out(
            catalog_embeddings=embeddings,
            catalog_labels=labels,
            catalog_ids=ids,
            query_embeddings=embeddings[query_mask],
            query_labels=labels[query_mask],
            query_ids=ids[query_mask],
            k=K,
        )
        avg_time = measure_inference_time(embeddings)

        results[label] = (precision, recall)

        print(
            f"[{label:<18}] "
            f"Precision@{K}: {precision:.4f}  "
            f"Recall@{K}: {recall:.4f}  "
            f"Time/query: {avg_time * 1000:.2f} ms  "
            f"(queries: {query_mask.sum()})"
        )

    if len(results) >= 2:
        plot_comparison(results)
    elif len(results) == 1:
        print("\nOnly one embedding set found -- generate at least one more "
              "(transfer / siamese) to get a comparison plot.")
    else:
        print("\nNo usable embeddings found.")

    return results


if __name__ == "__main__":
    evaluate_all()