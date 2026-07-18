import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilaritySearch:

    def __init__(self, embeddings_path):

        data = np.load(embeddings_path, allow_pickle=True)

        self.ids = data["ids"]
        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.filepaths = data["filepaths"]

    def query(self, query_embedding, k=5, exclude_id=None):

        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]

        ranked = np.argsort(-similarities)

        recommendations = []

        for idx in ranked:

            if exclude_id is not None and self.ids[idx] == exclude_id:
                continue

            recommendations.append({

                "id": self.ids[idx],

                "filepath": self.filepaths[idx],

                "similarity": float(similarities[idx]),

                "category": self.labels[idx]

            })

            if len(recommendations) == k:
                break

        return recommendations


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings",
        default="embeddings/baseline_embeddings.npz"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5
    )

    args = parser.parse_args()

    searcher = SimilaritySearch(args.embeddings)

    query_embedding = searcher.embeddings[0]

    results = searcher.query(query_embedding, args.k)

    for item in results:
        print(item)