"""Balanced, seeded cold start: the first labelled documents for active learning."""
import numpy as np


def balanced_initial_sample(y, n_per_class, seed):
    """Return indices into the training pool, exactly n_per_class per class.

    Raises ValueError if a class has fewer than n_per_class documents.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    picks = []
    for c in np.unique(y):
        idx_c = np.flatnonzero(y == c)
        if len(idx_c) < n_per_class:
            raise ValueError(
                f"class {c} has only {len(idx_c)} documents, need {n_per_class}"
            )
        picks.append(rng.choice(idx_c, size=n_per_class, replace=False))
    return np.concatenate(picks)


if __name__ == "__main__":
    from src.data.newsgroups import load_20newsgroups

    _, y_train, _ = load_20newsgroups("train")
    idx = balanced_initial_sample(y_train, n_per_class=2, seed=0)
    print(len(idx), np.bincount(y_train[idx], minlength=20))

    assert len(set(idx)) == len(idx), "an index was picked twice"
    assert np.array_equal(idx, balanced_initial_sample(y_train, 2, seed=0)), "same seed, different result"
    assert not np.array_equal(idx, balanced_initial_sample(y_train, 2, seed=1)), "seed has no effect"
    print("all checks passed")
