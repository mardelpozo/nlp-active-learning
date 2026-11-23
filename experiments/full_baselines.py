"""Full-data reference per domain: TF-IDF + LinearSVC trained on every pool document.

This is the ceiling each learning curve is compared against.
Usage: python -m experiments.full_baselines
Writes results/full_baseline_<domain>.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from src.data.domains import DOMAINS, load_domain

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED = 42

for domain in DOMAINS:
    texts_pool, y_pool, names = load_domain(domain, "train")
    texts_test, y_test, _ = load_domain(domain, "test")
    # Same features as experiments/al_curves.py, so the ceiling matches the curves.
    vectorizer = TfidfVectorizer(dtype=np.float32)
    x_pool = vectorizer.fit_transform(texts_pool)
    x_test = vectorizer.transform(texts_test)

    y_pred = LinearSVC(random_state=SEED).fit(x_pool, y_pool).predict(x_test)
    stats = {
        "domain": domain,
        "seed": SEED,
        "n_train": len(y_pool),
        "n_test": len(y_test),
        "n_classes": len(names),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "accuracy": accuracy_score(y_test, y_pred),
    }
    with open(RESULTS / f"full_baseline_{domain}.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"{domain:6s} n_train={stats['n_train']:6d} macro_f1={stats['macro_f1']:.3f} "
          f"accuracy={stats['accuracy']:.3f}")
