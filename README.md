# 🛍️ Visual Product Recommendation System

An image-based product recommendation engine that retrieves visually similar
fashion products for a query image using deep learning — no text or keywords
required.

Three retrieval pipelines are implemented and compared on a curated subset of
the [Fashion Product Images dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset):

| Pipeline | Approach |
|---|---|
| **Baseline** | Off-the-shelf ResNet50 (ImageNet weights), no training |
| **Transfer Learning** | ResNet50 fine-tuned as an 8-category classifier; embedding taken from the pre-softmax dense layer |
| **Siamese Network** | Dedicated embedding model trained with triplet loss to directly optimize similarity |

---

## Problem Statement

Traditional keyword-based product search cannot capture visual attributes
such as style, texture, or design. This project builds a system that accepts
a product photo and retrieves the top-K most visually similar catalog items
using deep visual embeddings and cosine similarity.

## Results

Evaluated on 400 held-out test images (never seen during training) as queries
against the full 2,000-image catalog, K = 5:

| Model | Precision@5 | Recall@5 | Avg. Time/Query | Embedding Dim |
|---|---|---|---|---|
| Baseline (ResNet50) | 0.9130 | 0.0183 | 21.90 ms | 2048 |
| Transfer Learning | **0.9435** | 0.0189 | 3.56 ms | 512 |
| Siamese Network | 0.9235 | 0.0185 | **0.67 ms** | 128 |

**Takeaway:** Transfer Learning achieves the highest precision. The Siamese
network trades a small precision gap for a ~33× speedup over the baseline
(and ~5× over Transfer Learning), thanks to its much lower-dimensional
embeddings — a genuine precision-vs-latency trade-off rather than a single
"best" model. See `results/precision_recall_comparison.png` and
`sample_testcases/` for the full breakdown and example queries.

> Recall@5 is uniformly low across all models by design: each category has
> ~250 relevant items in the catalog, so retrieving only 5 items can never
> recover more than ~2% of them. Precision@K and inference time are the
> metrics that actually differentiate the three models here.

---

## How It Works

```
Upload Image
     │
     ▼
Feature Extraction (Baseline / Transfer Learning / Siamese)
     │
     ▼
Embedding Vector (2048 / 512 / 128 dims)
     │
     ▼
Cosine Similarity vs. Precomputed Catalog Embeddings
     │
     ▼
Top-K Most Similar Products
```

All three models share the same retrieval mechanism (cosine similarity over
precomputed embeddings) — the only variable across pipelines is how the
embedding itself is produced, keeping the comparison fair.

### Key design decision: leakage-free evaluation

The dataset is split into **train** (~80%) and **test** (~20%, held out) per
category *before* any training happens. Transfer Learning and Siamese
training only ever see the train split; the held-out test split is used
exclusively as evaluation queries against the full catalog. This avoids the
inflated scores that come from testing a model on images it already trained
on.

---

## Project Structure

```
visualproductrecommender/
├── app.py                          # Streamlit UI
├── data/
│   ├── raw/                        # Original Kaggle dataset (images/ + styles.csv)
│   ├── subset/                     # Full catalog subset (train + test)
│   │   ├── subset_metadata.csv
│   │   ├── train_metadata.csv
│   │   └── test_metadata.csv
│   ├── subset_train/                # Train-split images only
│   └── subset_test/                 # Test-split images only (held out)
├── embeddings/
│   ├── baseline_embeddings.npz
│   ├── transfer_embeddings.npz
│   └── siamese_embeddings.npz
├── models/
│   ├── siamese_embedding_model.keras
│   └── transfer_learning_model.keras
├── results/
│   ├── precision_recall_comparison.png
│   └── triplet_loss.png
├── sample_testcases/                # Example query → results screenshots
├── scripts/
│   ├── prepare_data.py              # Step 1: build the subset + train/test split
│   ├── train_siamese.py             # Step 2: train Siamese network
│   └── run_evaluation.py            # Step 3: compare all models (held-out)
└── src/
    ├── data_loader.py               # Subset building, preprocessing, triplet generation
    ├── feature_extraction.py        # Baseline + generic embedding extraction
    ├── transfer_learning.py         # Fine-tuning pipeline
    ├── siamese_model.py             # Embedding model + triplet loss
    ├── generate_siamese_embeddings.py
    ├── generate_transfer_embeddings.py
    ├── similarity_search.py         # Cosine similarity retrieval
    └── evaluate.py                  # Precision@K / Recall@K, leakage-free eval
```

---

## Setup

```bash
git clone <your-repo-url>
cd visualproductrecommender
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Download the [Fashion Product Images dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
and place it at:
```
data/raw/images/       # all product images
data/raw/styles.csv    # metadata
```

## Usage

Run the full pipeline in order:

```bash
# 1. Build the 8-category subset + stratified train/test split
python scripts/prepare_data.py

# 2. Generate baseline embeddings (no training)
python src/feature_extraction.py --metadata data/subset/subset_metadata.csv \
    --out embeddings/baseline_embeddings.npz --model resnet50

# 3. Fine-tune the transfer-learning classifier
python src/transfer_learning.py
python src/generate_transfer_embeddings.py

# 4. Train the Siamese network + generate its embeddings
python scripts/train_siamese.py

# 5. Compare all three models on held-out test queries
python scripts/run_evaluation.py

# 6. Launch the interactive UI
streamlit run app.py
```

The Streamlit app lets you upload a product photo, pick a model (or compare
all three side by side), and view the top-K visually similar results with
similarity scores.

---

## Tech Stack

- **TensorFlow / Keras** — ResNet50 backbone, transfer learning, Siamese network
- **scikit-learn** — cosine similarity
- **Streamlit** — interactive UI
- **PIL / NumPy / pandas** — image processing and data handling

---

## Limitations & Future Work

- Triplets for the Siamese network are sampled randomly rather than via hard-negative
  mining, which limits precision gains on visually similar categories (e.g. T-shirts vs. Shirts).
- Retrieval uses brute-force cosine similarity, which is adequate at this catalog
  size (2,000 images) but would benefit from a FAISS index at larger scale.
- The subset (8 categories, 2,000 images) is intentionally small for training
  efficiency; results may shift with a larger/more diverse dataset.

## Dataset & References

- Fashion Product Images Dataset — Param Aggarwal, Kaggle
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 (ResNet)
- Schroff et al., *FaceNet: A Unified Embedding for Face Recognition and Clustering*, CVPR 2015 (triplet loss)