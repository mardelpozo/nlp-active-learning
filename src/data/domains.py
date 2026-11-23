"""One loader for the three domains: news, academic papers, tweets.

Every loader returns (texts, labels, label_names) with the same conventions as
load_20newsgroups: labels are a numpy int array, empty texts are dropped.
"""
import os
from pathlib import Path

import numpy as np

from src.data.newsgroups import load_20newsgroups

DATA_HOME = Path(__file__).resolve().parents[2] / "data"
# Hugging Face caches into ~/.cache by default; keep it inside the git-ignored data/ folder.
os.environ.setdefault("HF_HOME", str(DATA_HOME / "hf"))

# Hugging Face id, config, and which named split plays "train" (the unlabelled pool) and "test".
HF_SOURCES = {
    # no_ref: reference lists removed. Like 20NG's header removal, it drops text that
    # isn't the paper's own content (venue names, author lists of cited work).
    "arxiv": ("ccdv/arxiv-classification", "no_ref", {"train": "train", "test": "test"}),
}

# TweetTopic ships a loading script, which datasets>=4 no longer runs, so read the Parquet
# copies the Hub converts automatically. coling2022 is the official temporal split from the
# dataset's paper: test tweets are later than train tweets, like 20NG's by-date split.
TWEETS_REPO = "cardiffnlp/tweet_topic_single"
TWEETS_FILE = "tweet_topic_single/{split}/0000.parquet"
TWEETS_SPLITS = {"train": "train_coling2022", "test": "test_coling2022"}

DOMAINS = ("news", "arxiv", "tweets")


def _drop_empty(texts, labels):
    keep = np.array([len(t.split()) > 0 for t in texts])
    return [t for t, k in zip(texts, keep) if k], labels[keep]


def load_domain(domain, subset):
    """Return (texts, labels, label_names) for domain in DOMAINS, subset 'train' or 'test'."""
    if domain == "news":
        return load_20newsgroups(subset)

    from datasets import load_dataset  # imported here so 20NG-only scripts don't pay for it

    if domain == "tweets":
        from huggingface_hub import hf_hub_download

        # refs/convert/parquet is the branch where the Hub keeps its automatic Parquet copy.
        path = hf_hub_download(TWEETS_REPO, TWEETS_FILE.format(split=TWEETS_SPLITS[subset]),
                               repo_type="dataset", revision="refs/convert/parquet")
        ds = load_dataset("parquet", data_files={subset: path}, split=subset)
        # The Parquet copy stores label ids as plain ints; rebuild the id -> name list
        # from the label_name column that sits next to them.
        id_to_name = dict(zip(ds["label"], ds["label_name"]))
        label_names = [id_to_name[i] for i in range(len(id_to_name))]
    else:
        name, config, splits = HF_SOURCES[domain]
        ds = load_dataset(name, config, split=splits[subset])
        label_names = ds.features["label"].names
    texts = list(ds["text"])
    labels = np.array(ds["label"])
    texts, labels = _drop_empty(texts, labels)
    return texts, labels, label_names


if __name__ == "__main__":
    for domain in DOMAINS:
        for subset in ("train", "test"):
            texts, labels, names = load_domain(domain, subset)
            lengths = np.array([len(t.split()) for t in texts])
            counts = np.bincount(labels, minlength=len(names))
            print(f"{domain:6s} {subset:5s} docs={len(texts):6d} classes={len(names):2d} "
                  f"median_tokens={np.median(lengths):7.0f} "
                  f"imbalance={counts.max() / counts.min():.2f}")
