# Technical Documentation
### Visual Product Recommendation System

This document goes deeper than `README.md` — it covers the internal
architecture, what each module does, key design decisions, and known
behaviors/trade-offs worth understanding before presenting or extending this
project.

**Scope:** offline embedding pipelines (Baseline / Transfer Learning /
Siamese), the Streamlit retrieval app, and the evaluation methodology.
For setup instructions and results, see [`README.md`](./README.md).
For the full narrative write-up, see [`Project_Report.docx`](./Project_Report.docx).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Module Reference](#2-module-reference)
   - [`src/data_loader.py`](#srcdata_loaderpy)
   - [`src/feature_extraction.py`](#srcfeature_extractionpy)
   - [`src/transfer_learning.py`](#srctransfer_learningpy)
   - [`src/siamese_model.py`](#srcsiamese_modelpy)
   - [`src/evaluate.py`](#srcevaluatepy)
   - [`src/similarity_search.py`](#srcsimilarity_searchpy)
   - [`app.py`](#apppy)
3. [Design Decisions Worth Knowing Cold](#3-design-decisions-worth-knowing-cold)
4. [Known Trade-offs (Not Bugs)](#4-known-trade-offs-not-bugs)
5. [Reproducing From Scratch](#5-reproducing-from-scratch)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Offline Pipeline                        │
│                                                                   │
│  data/raw/  ──►  prepare_data.py  ──►  data/subset/ (+ train/    │
│  (Kaggle)         (stratified          test split, per category) │
│                    80/20 split)                                  │
│                                                                   │
│  data/subset/  ──►  feature_extraction.py  ──►  baseline_        │
│  (all images)        (ResNet50, no training)     embeddings.npz  │
│                                                                   │
│  data/subset_train/  ──►  transfer_learning.py  ──►  transfer_   │
│  (train split only)        (fine-tune classifier)    learning_   │
│                                                        model.keras│
│                        ──►  generate_transfer_embeddings.py  ──► │
│                              (full catalog)          transfer_   │
│                                                        embeddings │
│                                                        .npz       │
│                                                                   │
│  data/subset_train/  ──►  train_siamese.py  ──►  siamese_        │
│  (train split only,        (triplet loss)          embedding_    │
│   triplets only)                                    model.keras  │
│                        ──►  (full catalog)  ──►  siamese_         │
│                                                    embeddings.npz │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Online Pipeline                        │
│                                                                   │
│  User uploads image (app.py)                                     │
│         │                                                         │
│         ▼                                                         │
│  embed_uploaded_image()  ──►  same preprocessing as offline       │
│  (in-memory resize +          pipeline (resize → array →          │
│   preprocess_input,            preprocess_input)                  │
│   no disk round-trip)                                             │
│         │                                                         │
│         ▼                                                         │
│  SimilaritySearch.query()  ──►  cosine_similarity() against       │
│  (similarity_search.py)         the selected model's .npz         │
│         │                                                         │
│         ▼                                                         │
│  Top-K results rendered with image, ID, similarity, category      │
└─────────────────────────────────────────────────────────────────┘
```

The offline pipeline runs once (or whenever the dataset/models change) and
produces cached `.npz` embedding files. The online pipeline (the Streamlit
app) never touches the raw dataset or retrains anything — it just embeds one
uploaded image and does a similarity search against precomputed vectors.
This is what keeps per-query latency low (0.67–21.9 ms depending on model).

---

## 2. Module Reference

### `src/data_loader.py`
- `build_subset()` — reads `styles.csv`, filters to 8 target categories,
  samples `per_category` images each, and performs a **stratified 80/20
  train/test split per category** (not a global random split — this matters
  because a global split could leave some categories under-represented in
  test). Writes three metadata CSVs: full catalog, train-only, test-only.
- `load_and_preprocess_image()` — resize to 224×224 + ImageNet mean/std
  normalization. Used only for the manual triplet loading path; the main
  extraction pipeline uses Keras's own `preprocess_input` instead (see
  `feature_extraction.py`), which is functionally equivalent for ResNet50.
- `make_triplets()` — generates (anchor, positive, negative) filepath
  triplets. Positive = random other image from the same category as anchor;
  negative = random image from a different category. **Negatives are
  uniformly random, not hard-mined** — this is the main lever for improving
  Siamese precision in future work (see README limitations).

### `src/feature_extraction.py`
- `build_feature_extractor(model_name, weights_path)` — two modes:
  - No `weights_path`: builds ResNet50 or EfficientNetB0 with ImageNet
    weights, `include_top=False`, `pooling="avg"` (baseline mode).
  - With `weights_path`: loads a full trained model (e.g. the Siamese
    embedding model) and pairs it with `resnet_preprocess` — this is safe
    here because the Siamese model was itself trained on ResNet-preprocessed
    inputs (see `train_siamese.py`), but would need updating if a
    differently-preprocessed backbone were ever swapped in.
- `extract_embeddings()` — batches images through the model, returns
  `(ids, filepaths, labels, embeddings)`.
- `save_embeddings()` — writes the `.npz` format shared by all three
  pipelines: `ids`, `filepaths`, `labels`, `embeddings`.

### `src/transfer_learning.py`
- `build_model()` — ResNet50 backbone (frozen initially) →
  `GlobalAveragePooling2D` → `Dense(512, relu)` → `Dropout(0.3)` →
  `Dense(256, relu)` → `Dense(num_classes, softmax)`.
- `train_model()` — **two-stage training**:
  1. Backbone fully frozen, train only the new head (`Adam(1e-3)`).
  2. Unfreeze the last 30 backbone layers, fine-tune everything at a much
     lower learning rate (`Adam(1e-5)`) to avoid destroying the pretrained
     features.
- Embeddings for retrieval are pulled from `layers[-3]` (see
  `generate_transfer_embeddings.py` / `app.py`), which is the `Dropout`
  layer sitting right after `Dense(512)`. At inference time, Dropout is a
  no-op, so this is equivalent to reading the 512-d `Dense` output directly
  — chosen because it's the richer, higher-dimensional representation
  compared to the later 256-d layer.

### `src/siamese_model.py`
- `build_embedding_model()` — ResNet50 backbone (last 15 layers trainable,
  rest frozen) → `Dense(512, relu)` → `Dense(128)` (linear, no activation —
  this is the final embedding).
- `build_siamese_triplet_model()` — wraps three calls to the *same*
  embedding model (shared weights) on anchor/positive/negative inputs,
  concatenates their outputs for the loss function to unpack.
- `triplet_loss(margin=0.3)` — standard triplet loss:
  `max(d(a,p) - d(a,n) + margin, 0)`, encouraging the anchor-positive
  distance to be at least `margin` smaller than the anchor-negative
  distance.

### `src/evaluate.py`
- `precision_recall_at_k()` — **self-vs-self evaluation, kept for reference
  only.** Has data leakage if used on embeddings the model trained on.
  Not used by `scripts/run_evaluation.py`.
- `precision_recall_at_k_held_out()` — the evaluation actually used.
  Queries come exclusively from the held-out test split; they're compared
  against the *full* catalog (train + test) with the query's own catalog
  entry excluded from its own results (`self_idx` masking) so a query
  can't trivially "find itself."
- `measure_inference_time()` — times `cosine_similarity()` calls for a
  random sample of queries against the full catalog, in seconds/query.

### `src/similarity_search.py`
- `SimilaritySearch` — thin wrapper around a loaded `.npz` file. `query()`
  computes cosine similarity between one embedding and the whole catalog,
  returns the top-K as dicts with `id`, `filepath`, `similarity`, `category`.
  Supports `exclude_id` (used in evaluation to prevent self-matches; not
  used by the live app, since uploaded images have no catalog ID to exclude
  in the first place).

### `app.py`
- `embed_uploaded_image()` — processes the uploaded image **in memory**
  (resize → `img_to_array` → `preprocess_fn`), matching the offline
  extraction pipeline exactly. *(Earlier version of this function saved the
  image to a temp JPEG and reloaded it before embedding — this introduced
  small JPEG-compression differences from the catalog pipeline, which could
  occasionally cause a near-identical catalog image to rank #1 instead of
  the true source image. Fixed by removing the disk round-trip entirely.)*
- `render_recommendations()` — displays each result's image if the file
  exists at `item["filepath"]`, otherwise shows a "file not found" warning
  instead of crashing the whole page. This matters because `.npz` files
  store filepaths as of extraction time; if a referenced image is ever
  moved/deleted, this keeps the rest of the UI functional.
- `get_available_models()` — only lists a model in the sidebar if its
  `.npz` embeddings file actually exists on disk, so the app degrades
  gracefully if, say, only baseline embeddings have been generated so far.

---

## 3. Design Decisions Worth Knowing Cold

| Decision | Why |
|---|---|
| Stratified train/test split, done once, before any training | Prevents any model from training on images later used to evaluate it (data leakage) |
| Held-out queries evaluated against full catalog, not just test set | Mirrors real deployment: a retrieval system's catalog is fixed; only the *query* needs to be unseen |
| Embeddings precomputed and cached as `.npz` | Keeps live query latency independent of catalog size — no re-running the CNN over the whole catalog per search |
| In-memory preprocessing for uploads (no temp-file round-trip) | Keeps uploaded-image embeddings numerically consistent with catalog embeddings (see `app.py` note above) |
| Graceful fallback for missing image files | One missing/moved file doesn't crash the whole results page |
| `data/subset/` and both `.keras` models committed to the repo (via Git LFS for models) | Makes the repo fully self-contained and runnable via `git clone` + `pip install` + `streamlit run`, without requiring the ~multi-GB raw Kaggle dataset |

---

## 4. Known Trade-offs (Not Bugs)

- **Siamese Precision@5 (0.9235) is lower than Transfer Learning (0.9435).**
  Expected: triplet loss with randomly sampled (not hard-mined) negatives
  gives a weaker training signal for fine-grained category separation than
  direct cross-entropy classification. Siamese compensates with much faster
  inference (128-d vs. 512-d embeddings, ~5× faster per query).
- **Recall@5 is uniformly low (~0.018) across all models.** Mathematically
  expected given ~250 relevant items per category and K=5 — not a sign of
  poor retrieval quality. Precision@K and latency are the metrics that
  actually differentiate the three pipelines here.
- **Uploading an image already in the catalog returns itself as the #1
  result, with similarity near 1.0.** Correct behavior for a similarity
  search system — nothing can be more similar to an image than itself.

---

## 5. Reproducing From Scratch

See `README.md` → Usage → "To rebuild the full pipeline from scratch" for
the exact command sequence. Order matters: `prepare_data.py` must run first
(it creates the train/test split that every downstream script depends on),
and `train_siamese.py` / `transfer_learning.py` must only ever be pointed at
`data/subset_train/`, never the full `data/subset/`, to preserve the
leakage-free evaluation guarantee described in Section 3.