"""Load 20 Newsgroups and print the dataset facts recorded in README.md."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# Download into the repo's /data/ folder (git-ignored) instead of ~/scikit_learn_data,
# so the project is self-contained and the cache is easy to find or delete.
DATA_HOME = Path(__file__).resolve().parents[2] / "data"

# Strip the parts of each post that give the topic away without reading the text:
# headers contain "Newsgroups: sci.space", footers are signatures, quotes repeat
# earlier posts from the same thread. Keeping them inflates accuracy.
REMOVE = ("headers", "footers", "quotes")


def load_20newsgroups(subset, drop_empty=True):
    """Return (texts, labels, label_names) for subset 'train' or 'test'.

    drop_empty removes posts with no tokens left after REMOVE. They become all-zero
    TF-IDF vectors that uncertainty sampling would query first, wasting labels.
    """
    bunch = fetch_20newsgroups(
        subset=subset,
        remove=REMOVE,
        data_home=DATA_HOME,
        shuffle=True,
        random_state=42,  # fixed order, so runs are reproducible
    )
    texts, labels = bunch.data, bunch.target
    if drop_empty:
        # str.split() with no argument splits on any whitespace, so whitespace-only
        # posts also count as empty.
        keep = np.array([len(t.split()) > 0 for t in texts])
        texts = [t for t, k in zip(texts, keep) if k]
        labels = labels[keep]  # boolean mask keeps labels aligned with texts
    return texts, labels, bunch.target_names


def describe(texts, labels, label_names):
    """Collect the dataset facts for one split."""
    # Length in whitespace-separated tokens: a simple, tokenizer-independent measure.
    lengths = np.array([len(t.split()) for t in texts])
    counts = pd.Series(labels).value_counts().sort_index()
    counts.index = [label_names[i] for i in counts.index]
    return {
        "n_docs": len(texts),
        "n_classes": len(label_names),
        "median_tokens": float(np.median(lengths)),
        "empty_docs": int((lengths == 0).sum()),  # posts that were only header/quote
        "class_counts": counts,
        # Imbalance ratio: largest class / smallest class. 1.0 = perfectly balanced.
        "imbalance_ratio": counts.max() / counts.min(),
    }


if __name__ == "__main__":
    for subset in ("train", "test"):
        stats = describe(*load_20newsgroups(subset))
        print(f"\n== {subset} ==")
        for key in ("n_docs", "n_classes", "median_tokens", "empty_docs"):
            print(f"{key}: {stats[key]}")
        print(f"imbalance_ratio: {stats['imbalance_ratio']:.2f}")
        shares = (stats["class_counts"] / stats["n_docs"] * 100).round(1)
        print(pd.DataFrame({"count": stats["class_counts"], "share_%": shares}).to_string())
