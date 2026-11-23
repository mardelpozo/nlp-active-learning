"""Summarise the active-learning curves into the numbers the research question asks for.

Per domain and strategy:
  - mean macro-F1 at each budget,
  - paired wins: in how many seeds the strategy beat random at that budget (same seed = same cold start),
  - AUC: mean macro-F1 averaged over the budgets (equal weight per doubling, matching the log x-axis),
  - labels_to_match_random_1600: how many labels the strategy needs to reach random sampling's
    mean macro-F1 at 1,600 labels, interpolated on a log2 scale between budgets (None if never reached).

Usage: python -m experiments.summarise_al
Writes results/al_summary.csv and prints it.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
DOMAINS = ["news", "arxiv", "tweets"]


def labels_to_reach(curve, target):
    """Smallest budget (log2-interpolated) where the mean curve first reaches target."""
    n, f1 = curve.index.to_numpy(), curve.to_numpy()
    if f1[0] >= target:
        return float(n[0])
    for i in range(1, len(n)):
        if f1[i] >= target:
            # Linear in log2(n): the budgets double, so this is "how far between two doublings".
            frac = (target - f1[i - 1]) / (f1[i] - f1[i - 1])
            return float(2 ** (np.log2(n[i - 1]) + frac * (np.log2(n[i]) - np.log2(n[i - 1]))))
    return None


rows = []
for domain in DOMAINS:
    runs = pd.read_csv(RESULTS / f"al_curve_{domain}.csv")
    wide = runs.pivot_table(index=["seed", "n_labels"], columns="strategy", values="macro_f1")
    means = runs.groupby(["strategy", "n_labels"])["macro_f1"].mean()
    target = means["random"].iloc[-1]  # random's mean at the largest budget
    for strategy in runs["strategy"].unique():
        curve = means[strategy]
        row = {"domain": domain, "strategy": strategy, "auc": round(curve.mean(), 4)}
        for n in curve.index:
            row[f"f1@{n}"] = round(curve[n], 3)
            if strategy != "random":
                at_n = wide.xs(n, level="n_labels")
                row[f"wins@{n}"] = int((at_n[strategy] > at_n["random"]).sum())
        reached = labels_to_reach(curve, target)
        row["labels_to_match_random_1600"] = None if reached is None else round(reached)
        rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(RESULTS / "al_summary.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 30):
    print(summary.to_string(index=False))
