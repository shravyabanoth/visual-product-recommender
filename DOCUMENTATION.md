# Documentation

This file explains how the project actually works under the hood — what
each script does and why things were built a certain way. The `README.md`
covers setup and results; this is more about the "how" and "why."

## Architecture

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
app) never touches the raw dataset or retrains anything — it just embeds
one uploaded image and does a similarity search against precomputed
vectors. That's why a search only takes 0.67–21.9 ms depending on the
model, instead of re-running a CNN over the whole catalog every time.

## How everything connects

1. `scripts/prepare_data.py` builds a subset of the dataset (8 categories,
   ~250 images each) and splits it into train/test **before** any model
   sees it. This is done once.
2. Each of the three models (baseline, transfer learning, siamese) turns
   every image into an embedding — just a list of numbers that represents
   what the image looks like.
3. Those embeddings get saved to `.npz` files in `embeddings/` so we don't
   have to re-run the model every time someone searches.
4. When someone uploads an image in the app, it gets converted into an
   embedding the same way, then compared (cosine similarity) against all
   the saved embeddings to find the closest matches.

## What each file does

**`src/data_loader.py`**
Builds the subset and splits it into train/test, stratified per category
(so each category has roughly the same train/test ratio, not just a random
global split). Also has `make_triplets()`, which creates anchor/positive/
negative triplets for training the Siamese network — positive = same
category as anchor, negative = random different category. The negatives
are picked randomly, not specifically "hard" ones, which is part of why
the Siamese model doesn't beat Transfer Learning on precision (more on
that below).

**`src/feature_extraction.py`**
Handles turning images into embeddings using either plain ResNet50
(baseline, no training) or a custom trained model if you pass in a
`weights_path` (used for the Siamese model).

**`src/transfer_learning.py`**
Takes ResNet50, freezes it, adds a small classifier on top, and trains it
to recognize the 8 categories. Trains in two stages — first with the
backbone frozen, then unfreezing the last ~30 layers and fine-tuning at a
much lower learning rate so it doesn't wreck the pretrained weights. The
embedding used for retrieval is actually the layer right before the final
classification output (`layers[-3]`) — that layer happens to be a Dropout
layer, but Dropout doesn't do anything at prediction time, so it's the
same as reading the Dense(512) layer directly.

**`src/siamese_model.py`**
Builds the embedding model (ResNet50 + two Dense layers, ending in 128
numbers) and the triplet loss function. Triplet loss basically says: pull
the anchor and positive closer together, push the anchor and negative
farther apart, by at least some margin.

**`src/evaluate.py`**
Has two evaluation functions. The one actually used
(`precision_recall_at_k_held_out`) only uses the held-out test images as
queries, searched against the full catalog — this avoids testing a model
on images it already trained on. The other one (`precision_recall_at_k`)
is the "wrong" self-vs-self way of doing it, kept in the file just for
reference, not actually called anywhere in the pipeline.

**`src/similarity_search.py`**
Just loads an `.npz` file and does cosine similarity search against it.
Same logic used for all three models — the only thing that changes
between them is which embeddings file gets loaded.

**`app.py`**
The Streamlit app. Worth knowing: the function that processes an uploaded
image (`embed_uploaded_image`) used to save the image to a temp JPEG file
and reload it before running it through the model. That extra save/reload
step introduced small differences from how the catalog images were
originally processed, which occasionally caused a near-duplicate catalog
image to outrank the actual matching image. Fixed by processing the
uploaded image directly in memory instead, so it goes through the exact
same steps as the catalog images did.

Also worth knowing: if a catalog image file is missing on disk (say it
got moved or deleted), the app shows a small warning instead of crashing
the whole page.

## Why the results look the way they do

- **Precision@5 is 91–94% for all three models.** That's high because
  there are only 8 categories and they're pretty visually distinct, so
  even the untrained baseline does reasonably well. There isn't a ton of
  room left for the other two models to improve on.
- **Transfer Learning has the best precision (94.35%), not Siamese.**
  Transfer Learning gets a strong, direct training signal (cross-entropy
  over labeled categories). Siamese only gets a relative signal (triplet
  loss with random negatives, not specifically hard ones), so it has less
  pressure to nail the exact boundary between similar categories.
- **Siamese is way faster (0.67ms vs 21.9ms for baseline).** That's
  because its embeddings are only 128 numbers long instead of 2048, so the
  similarity search has way less math to do per comparison.
- **Recall@5 is tiny (~0.018) for every model.** Each category has around
  250 images in the catalog, so returning only 5 results can never capture
  more than about 2% of what's actually relevant. It's a limitation of
  only looking at K=5, not a sign anything's broken.
- **Uploading a photo that's already in the catalog returns itself as the
  top result.** That's correct — nothing can be more similar to an image
  than itself.

## If you wanted to improve this further

- Pick harder negatives when building triplets for the Siamese network
  (currently just random), which should help it catch up to Transfer
  Learning on precision, especially for similar-looking categories like
  T-shirts vs Shirts.
- Swap cosine similarity for a proper index like FAISS if the catalog got
  much bigger — not needed at 2,000 images, but would matter at scale.
- Bigger subset / more images per category, if there's compute for it.