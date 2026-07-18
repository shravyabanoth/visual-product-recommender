"""
evaluate.py

Computes Precision@K and Recall@K, treating items in the same category
as "relevant" to each other. Also times inference.

precision_recall_at_k(): original self-vs-self evaluation. Kept for
reference, but has data leakage if the embeddings were also used to train
the model (e.g. Siamese triplets) -- every "query" was something the model
trained on.

precision_recall_at_k_held_out(): the correct evaluation. Queries come only
from a held-out test split; they're compared against the full catalog
(train+test), which is how a real retrieval system is evaluated -- the
catalog can contain anything, but the query must be something the model
never trained on.

Usage:
    python src/evaluate.py --embeddings embeddings/baseline_embeddings.npz --k 5
"""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def precision_recall_at_k(embeddings: np.ndarray, labels: np.ndarray, k: int = 5):
    """Original self-vs-self evaluation. See leakage caveat above."""
    sims = cosine_similarity(embeddings)
    np.fill_diagonal(sims, -np.inf)

    precisions, recalls = [], []
    for i in range(len(embeddings)):
        top_k_idx = np.argsort(-sims[i])[:k]
        relevant_total = np.sum(labels == labels[i]) - 1
        relevant_total = max(relevant_total, 1)

        hits = np.sum(labels[top_k_idx] == labels[i])
        precisions.append(hits / k)
        recalls.append(hits / relevant_total)

    return float(np.mean(precisions)), float(np.mean(recalls))


def precision_recall_at_k_held_out(
    catalog_embeddings: np.ndarray,
    catalog_labels: np.ndarray,
    catalog_ids: np.ndarray,
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    query_ids: np.ndarray,
    k: int = 5,
):
    """
    Evaluates using held-out queries (test split) against the full catalog
    (train+test). This is the leakage-free version: queries never appear in
    what the model was trained on.
    """
    sims = cosine_similarity(query_embeddings, catalog_embeddings)

    precisions, recalls = [], []
    for i in range(len(query_embeddings)):
        row = sims[i].copy()

        # If the query image also happens to sit in the catalog (it does,
        # since the catalog is train+test), exclude it from its own results.
        self_idx = np.where(catalog_ids == query_ids[i])[0]
        if len(self_idx):
            row[self_idx] = -np.inf

        top_k_idx = np.argsort(-row)[:k]

        relevant_total = np.sum(catalog_labels == query_labels[i])
        if len(self_idx):
            relevant_total -= 1
        relevant_total = max(relevant_total, 1)

        hits = np.sum(catalog_labels[top_k_idx] == query_labels[i])
        precisions.append(hits / k)
        recalls.append(hits / relevant_total)

    return float(np.mean(precisions)), float(np.mean(recalls))


def measure_inference_time(embeddings: np.ndarray, n_queries: int = 50) -> float:
    idx = np.random.choice(len(embeddings), size=min(n_queries, len(embeddings)), replace=False)
    start = time.time()
    for i in idx:
        _ = cosine_similarity(embeddings[i].reshape(1, -1), embeddings)
    elapsed = time.time() - start
    return elapsed / len(idx)


def plot_comparison(results: dict, out_path: str = "results/precision_recall_comparison.png"):
    labels = list(results.keys())
    precisions = [results[k][0] for k in labels]
    recalls = [results[k][1] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, precisions, width, label="Precision@K")
    ax.bar(x + width / 2, recalls, width, label="Recall@K")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs. Transfer vs. Siamese Retrieval Quality (held-out test queries)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    embeddings, labels = data["embeddings"], data["labels"]

    precision, recall = precision_recall_at_k(embeddings, labels, k=args.k)
    avg_time = measure_inference_time(embeddings)

    print(f"Precision@{args.k}: {precision:.4f}")
    print(f"Recall@{args.k}:    {recall:.4f}")
    print(f"Avg inference time/query: {avg_time * 1000:.2f} ms")