"""
data_loader.py

Handles:
- Building a subset of the Fashion Product Images dataset
  (5-8 categories, ~200-300 images each)
- Splitting that subset into train/test (stratified per category) so that
  model training and evaluation never touch the same images
- Loading/preprocessing images (resize + ImageNet normalization)
- Generating anchor/positive/negative triplets for Siamese training

Expected raw dataset layout:
    data/raw/
        images/                 # all product images, named <id>.jpg
        styles.csv              # metadata: id, articleType, ... etc.

Output layout after build_subset():
    data/subset/<category>/<id>.jpg        # full catalog (train+test)
    data/subset_train/<category>/<id>.jpg  # train-only images
    data/subset_test/<category>/<id>.jpg   # test-only images (held out)
    data/subset/subset_metadata.csv        # full catalog metadata
    data/subset/train_metadata.csv         # train-split metadata
    data/subset/test_metadata.csv          # test-split metadata
"""

import os
import random
import shutil

import numpy as np
import pandas as pd
from PIL import Image

IMG_SIZE = (224, 224)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

DEFAULT_CATEGORIES = [
    "Tshirts", "Shirts", "Casual Shoes", "Sports Shoes",
    "Dresses", "Handbags", "Watches", "Sunglasses",
]


def build_subset(raw_dir: str, out_dir: str, categories=None,
                  per_category: int = 250, test_size: float = 0.2,
                  seed: int = 42):
    """
    Reads styles.csv, filters to the chosen categories, samples
    `per_category` images per category, and:
      - copies ALL sampled images into out_dir/<category>/ (the full catalog)
      - splits them (stratified, per category) into train/test and copies
        train images into f"{out_dir}_train"/<category>/ and test images
        into f"{out_dir}_test"/<category>/
      - writes subset_metadata.csv, train_metadata.csv, test_metadata.csv

    Returns (subset_df, train_df, test_df).
    """
    random.seed(seed)
    categories = categories or DEFAULT_CATEGORIES

    styles_path = os.path.join(raw_dir, "styles.csv")
    df = pd.read_csv(styles_path, on_bad_lines="skip")
    df = df[df["articleType"].isin(categories)]

    train_out_root = out_dir.rstrip("/\\") + "_train"
    test_out_root = out_dir.rstrip("/\\") + "_test"

    all_records, train_records, test_records = [], [], []

    for cat in categories:
        cat_df = df[df["articleType"] == cat]
        n = min(per_category, len(cat_df))
        sampled = cat_df.sample(n=n, random_state=seed)

        cat_dir = os.path.join(out_dir, cat.replace(" ", "_"))
        cat_train_dir = os.path.join(train_out_root, cat.replace(" ", "_"))
        cat_test_dir = os.path.join(test_out_root, cat.replace(" ", "_"))
        os.makedirs(cat_dir, exist_ok=True)
        os.makedirs(cat_train_dir, exist_ok=True)
        os.makedirs(cat_test_dir, exist_ok=True)

        n_test = max(1, int(round(n * test_size)))
        shuffled = sampled.sample(frac=1.0, random_state=seed)
        test_ids = set(shuffled.iloc[:n_test]["id"].tolist())

        for _, row in shuffled.iterrows():
            img_id = row["id"]
            src = os.path.join(raw_dir, "images", f"{img_id}.jpg")
            if not os.path.exists(src):
                continue

            dst = os.path.join(cat_dir, f"{img_id}.jpg")
            shutil.copyfile(src, dst)
            all_records.append({"id": img_id, "articleType": cat, "filepath": dst})

            is_test = img_id in test_ids
            split_dir = cat_test_dir if is_test else cat_train_dir
            split_dst = os.path.join(split_dir, f"{img_id}.jpg")
            shutil.copyfile(src, split_dst)

            record = {"id": img_id, "articleType": cat, "filepath": split_dst}
            (test_records if is_test else train_records).append(record)

    subset_df = pd.DataFrame(all_records)
    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    subset_df.to_csv(os.path.join(out_dir, "subset_metadata.csv"), index=False)
    train_df.to_csv(os.path.join(out_dir, "train_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test_metadata.csv"), index=False)

    print(f"Built subset: {len(subset_df)} images across {len(categories)} categories")
    print(f"  Train (used for training): {len(train_df)} images")
    print(f"  Test  (held out, eval only): {len(test_df)} images")

    return subset_df, train_df, test_df


def load_and_preprocess_image(path: str) -> np.ndarray:
    """Load an image, resize to IMG_SIZE, normalize with ImageNet stats."""
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


def make_triplets(subset_df: pd.DataFrame, n_triplets: int = 3000, seed: int = 42):
    """
    Generate (anchor, positive, negative) filepath triplets.
    Positive = same articleType as anchor, Negative = different articleType.

    IMPORTANT: pass only a TRAIN split dataframe here (e.g. train_metadata.csv)
    so that test images are never used to shape the embedding space.
    """
    random.seed(seed)
    by_category = subset_df.groupby("articleType")["filepath"].apply(list).to_dict()
    categories = list(by_category.keys())

    triplets = []
    for _ in range(n_triplets):
        pos_cat = random.choice(categories)
        neg_cat = random.choice([c for c in categories if c != pos_cat])

        anchor, positive = random.sample(by_category[pos_cat], 2) \
            if len(by_category[pos_cat]) >= 2 else (by_category[pos_cat][0],) * 2
        negative = random.choice(by_category[neg_cat])

        triplets.append((anchor, positive, negative))

    return triplets