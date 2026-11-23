"""Active-learning learning curves: uncertainty sampling vs random sampling, per domain.

For each domain, seed and query strategy:
  1. start from the same balanced cold start (2 labelled documents per class, chosen by seed),
  2. repeatedly query 10 documents, reveal their labels, retrain (small-text's PoolBasedActiveLearner),
  3. score on the full test set each time the labelled set reaches a budget.

Random sampling runs inside the same loop from the same cold start, so for a given seed the
only difference between strategies is which documents get queried (a paired comparison).

Usage: python -m experiments.al_curves news arxiv tweets [--jobs 4]
Writes results/al_curve_<domain>.csv and results/al_meta_<domain>.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from small_text import (
    BreakingTies,
    ConfidenceEnhancedLinearSVC,
    LeastConfidence,
    PoolBasedActiveLearner,
    PredictionEntropy,
    RandomSampling,
    SklearnClassifierFactory,
    SklearnDataset,
)

from src.core.initial_sample import balanced_initial_sample
from src.data.domains import load_domain

RESULTS = Path(__file__).resolve().parents[1] / "results"

SEEDS = [0, 1, 2, 3, 4]
BUDGETS = [50, 100, 200, 400, 800, 1600]
N_PER_CLASS = 2  # cold start: 2 labelled documents per class
BATCH = 10  # documents queried per round
STRATEGIES = {
    "random": RandomSampling,
    "least_confidence": LeastConfidence,
    "entropy": PredictionEntropy,
    "breaking_ties": BreakingTies,
}


def run_one(x_pool, y_pool, x_test, y_test, n_classes, strategy, seed):
    """One active-learning run; returns one row per budget."""
    # RandomSampling draws from numpy's global generator, so seed it per run.
    np.random.seed(seed)
    factory = SklearnClassifierFactory(
        ConfidenceEnhancedLinearSVC(linearsvc_kwargs={"random_state": seed}), n_classes
    )
    # target_labels: tell small-text all classes up front, so it never infers them from a subset.
    pool = SklearnDataset(x_pool, y_pool, target_labels=np.arange(n_classes))
    test = SklearnDataset(x_test, y_test, target_labels=np.arange(n_classes))

    learner = PoolBasedActiveLearner(factory, STRATEGIES[strategy](), pool)
    initial = balanced_initial_sample(y_pool, N_PER_CLASS, seed)
    learner.initialize_data(initial, y_pool[initial])

    rows = []
    for budget in BUDGETS:
        while len(learner.indices_labeled) < budget:
            n = min(BATCH, budget - len(learner.indices_labeled))
            queried = learner.query(num_samples=n)
            learner.update(y_pool[queried])  # the "oracle": reveal the true labels
        labelled = learner.indices_labeled
        y_pred = learner.classifier.predict(test)
        rows.append({
            "seed": seed,
            "strategy": strategy,
            "n_labels": len(labelled),
            "classes_seen": len(np.unique(y_pool[labelled])),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "accuracy": accuracy_score(y_test, y_pred),
        })
    return rows


def run_domain(domain, jobs):
    start = time.time()
    texts_pool, y_pool, names = load_domain(domain, "train")
    texts_test, y_test, _ = load_domain(domain, "test")

    # TF-IDF fitted once on the whole pool: it uses no labels, and the pool exists from day one.
    # float32 halves memory (the arXiv pool is 28k full papers) with no effect on the ranking.
    vectorizer = TfidfVectorizer(dtype=np.float32)
    x_pool = vectorizer.fit_transform(texts_pool)
    x_test = vectorizer.transform(texts_test)
    print(f"[{domain}] pool {x_pool.shape}, test {x_test.shape}", flush=True)

    budgets = [b for b in BUDGETS if b <= len(y_pool)]
    assert budgets == BUDGETS, f"{domain}: pool smaller than the largest budget"

    tasks = [(s, seed) for s in STRATEGIES for seed in SEEDS]
    results = Parallel(n_jobs=jobs, verbose=5)(
        delayed(run_one)(x_pool, y_pool, x_test, y_test, len(names), s, seed) for s, seed in tasks
    )
    df = pd.DataFrame([row for rows in results for row in rows])
    df.to_csv(RESULTS / f"al_curve_{domain}.csv", index=False)

    meta = {
        "domain": domain,
        "n_pool": int(x_pool.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_classes": len(names),
        "vocabulary": int(x_pool.shape[1]),
        "seeds": SEEDS,
        "budgets": BUDGETS,
        "n_per_class_initial": N_PER_CLASS,
        "batch_size": BATCH,
        "strategies": list(STRATEGIES),
        "classifier": "ConfidenceEnhancedLinearSVC (LinearSVC defaults)",
        "features": "TfidfVectorizer defaults, fitted on the unlabelled pool",
        "runtime_s": round(time.time() - start, 1),
    }
    with open(RESULTS / f"al_meta_{domain}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{domain}] done in {meta['runtime_s']} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="+", choices=["news", "arxiv", "tweets"])
    parser.add_argument("--jobs", type=int, default=4, help="parallel runs (each holds a copy of the data)")
    args = parser.parse_args()
    for domain in args.domains:
        run_domain(domain, args.jobs)
