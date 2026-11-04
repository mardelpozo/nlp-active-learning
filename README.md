Research question: Does uncertainty sampling reach the same accuracy with fewer labels than random sampling, and does that saving hold across news, papers and tweets?

Baseline: random sampling must be performed in order to cold start the active learning loop so that the model will request the oracle for the first set of labels randomly, and posterior to that can the uncertainty sampling query strategy be applied given the predictions to verify whether the label budget proves smaller in comparison to fully random sampling without a degrade in accuracy.

The learning-curve output should look relatively convergent at very small and very large label budgets. My hypothesis is that the AUC will be higher for uncertainty sampling, while accuracy will be eventually convergent for both.

│                    Fact                    │   Train    │    Test    │
  ├────────────────────────────────────────────┼────────────┼────────────┤
  │ Documents                                  │ 11,314     │ 7,532      │
  ├────────────────────────────────────────────┼────────────┼────────────┤
  │ Classes                                    │ 20         │ 20         │
  ├────────────────────────────────────────────┼────────────┼────────────┤
  │ Median length (whitespace tokens)          │ 83         │ 82         │
  ├────────────────────────────────────────────┼────────────┼────────────┤
  │ Empty after removal                        │ 300 (2.7%) │ 215 (2.9%) │
  ├────────────────────────────────────────────┼────────────┼────────────┤
  │ Imbalance ratio (largest ÷ smallest class) │ 1.59       │ 1.59       │
  └────────────────────────────────────────────┴────────────┴────────────┘

  