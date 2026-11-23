"""Plot uncertainty vs random sampling learning curves for news, papers and tweets.

Reads results/al_curve_<domain>.csv and results/full_baseline_<domain>.json,
writes figures/al_curves.html and .png (one panel per domain).
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

PANELS = [("news", "News (20 Newsgroups)"), ("arxiv", "Papers (arXiv)"), ("tweets", "Tweets (TweetTopic)")]
# Strategy -> (legend name, colour, rgba band, dash). Dash is a second cue besides colour.
SERIES = {
    "random": ("Random sampling", "#2a78d6", "rgba(42, 120, 214, 0.15)", "dash"),
    "least_confidence": ("Uncertainty sampling (least confidence)", "#eb6834", "rgba(235, 104, 52, 0.15)", "solid"),
}
FONT = dict(family="Arial", size=12, color="black")

fig = make_subplots(rows=1, cols=3, subplot_titles=[title for _, title in PANELS],
                    shared_yaxes=True, horizontal_spacing=0.04)

for col, (domain, _) in enumerate(PANELS, start=1):
    runs = pd.read_csv(RESULTS / f"al_curve_{domain}.csv")
    with open(RESULTS / f"full_baseline_{domain}.json") as f:
        full_f1 = json.load(f)["macro_f1"]

    for strategy, (name, color, band, dash) in SERIES.items():
        curve = (runs[runs["strategy"] == strategy]
                 .groupby("n_labels")["macro_f1"].agg(["mean", "min", "max"]).reset_index())
        # Min-max band over seeds: outline along max left-to-right, back along min, fill inside.
        fig.add_trace(go.Scatter(
            x=pd.concat([curve["n_labels"], curve["n_labels"][::-1]]),
            y=pd.concat([curve["max"], curve["min"][::-1]]),
            mode="lines", fill="toself", fillcolor=band, line=dict(width=0),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=curve["n_labels"], y=curve["mean"], mode="lines+markers", name=name,
            line=dict(color=color, width=2, dash=dash), marker=dict(size=8, color=color),
            showlegend=(col == 1),  # one legend entry per strategy, not per panel
            hovertemplate=f"{name}<br>%{{x:,}} labels: macro-F1 %{{y:.3f}}<extra></extra>",
        ), row=1, col=col)

    fig.add_hline(y=full_f1, line=dict(color="grey", width=1, dash="dot"), row=1, col=col,
                  annotation_text=f"All labels ({full_f1:.2f})", annotation_position="top left",
                  annotation_font=dict(FONT, size=10, color="#555"))

axis = dict(showgrid=False, showline=True, linecolor="black", linewidth=0.75,
            ticks="outside", tickwidth=0.75, tickcolor="black", mirror=False)
budgets = sorted(pd.read_csv(RESULTS / "al_curve_news.csv")["n_labels"].unique())
fig.update_xaxes(axis, type="log", tickvals=budgets, ticktext=[f"{n:,}" for n in budgets])
fig.update_xaxes(title_text="Labelled training documents", row=1, col=2)
fig.update_yaxes(axis, range=[0, 1])
fig.update_yaxes(title_text="Test macro-F1", row=1, col=1)
fig.update_annotations(font=FONT)  # subplot titles use the same font
fig.update_layout(
    template="simple_white", font=FONT, width=1100, height=420,
    margin=dict(l=60, r=20, t=40, b=90),
    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.25, bgcolor="rgba(0,0,0,0)"),
    hovermode="closest",
)

FIGURES.mkdir(exist_ok=True)
fig.write_html(FIGURES / "al_curves.html", include_plotlyjs="cdn")
fig.write_image(FIGURES / "al_curves.png", scale=3)
print("Saved figures/al_curves.html and .png")
