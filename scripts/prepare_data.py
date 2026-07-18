"""
scripts/prepare_data.py

Step 1. Builds the working subset from the raw Kaggle dataset.
Run: python scripts/prepare_data.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import build_subset

if __name__ == "__main__":
    build_subset(
        raw_dir="data/raw",
        out_dir="data/subset",
        per_category=250,
    )