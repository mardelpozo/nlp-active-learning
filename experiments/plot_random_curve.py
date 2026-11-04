"""Plot the random-sampling learning curve on 20 Newsgroups.

Reads results/random_curve_20ng.csv and results/baseline_20ng.json,
writes figures/random_curve_20ng.html and .png.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]  # repo root, so the script runs from any folder
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

LINE = "#1f4e79"  # dark blue: the random-sampling series
FONT = dict(family="Arial", size=12, color="black")

runs = pd.read_csv(RESULTS / "random_curve_20ng.csv")
with open(RESULTS / "baseline_20ng.json") as f:
    full_f1 = json.load(f)["macro_f1"]

# One row per budget: mean, min and max over the 5 seeds.
curve = runs.groupby("n_labels")["macro_f1"].agg(["mean", "min", "max"]).reset_index()

fig = go.Figure()

# Min-max band, drawn as one closed shape: along the max values left to right,
# then back along the min values right to left, and fill the inside.
fig.add_trace(go.Scatter(
    x=pd.concat([curve["n_labels"], curve["n_labels"][::-1]]),
    y=pd.concat([curve["max"], curve["min"][::-1]]),
    mode="lines",  # default for Scatter adds markers at every min/max point
    fill="toself",
    fillcolor="rgba(31, 78, 121, 0.15)",  # same blue as LINE, 15% opaque
    line=dict(width=0),
    hoverinfo="skip",
    name="Range over 5 seeds",
))

fig.add_trace(go.Scatter(
    x=curve["n_labels"],
    y=curve["mean"],
    mode="lines+markers",
    line=dict(color=LINE, width=1.5),
    marker=dict(size=5),
    name="Random sampling (mean)",
))

# Reference line: what all 11,014 labels achieve.
fig.add_hline(
    y=full_f1,
    line=dict(color="grey", width=1, dash="dash"),
    annotation_text=f"Full training set ({full_f1:.3f})",
    annotation_position="bottom right",
    annotation_font=FONT,
)

axis = dict(
    showgrid=False,
    showline=True, linecolor="black", linewidth=0.75,
    ticks="outside", tickwidth=0.75, tickcolor="black",
    mirror=False,  # axis lines on the left and bottom only
)

fig.update_layout(
    template="simple_white",
    font=FONT,
    width=600, height=420,
    margin=dict(l=60, r=20, t=20, b=55),
    legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(
        axis,
        title="Labelled training documents",
        type="log",
        # Ticks at the budgets themselves; on a log axis each doubling is the same width.
        tickvals=curve["n_labels"],
        ticktext=[f"{n:,}" for n in curve["n_labels"]],
    ),
    yaxis=dict(axis, title="Test macro-F1", range=[0, 0.75]),
)

FIGURES.mkdir(exist_ok=True)
fig.write_html(FIGURES / "random_curve_20ng.html", include_plotlyjs="cdn")
fig.write_image(FIGURES / "random_curve_20ng.png", scale=3)  # scale=3 gives print resolution
print("Saved figures/random_curve_20ng.html and .png")
