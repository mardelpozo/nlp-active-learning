import numpy as np
import pandas as pd
from src.data.newsgroups import load_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

SEEDS = [0, 1, 2, 3, 4]
BUDGETS = [50, 100, 200, 400, 800, 1600]

vectorizer = TfidfVectorizer()

X_train, y_train, _ = load_20newsgroups(subset='train', drop_empty=True)
X_test, y_test, _ = load_20newsgroups(subset='test', drop_empty=True)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Train TF-IDF shape: {X_train_tfidf.shape}")
print(f"Test TF-IDF shape: {X_test_tfidf.shape}")

runs = []

for seed in SEEDS:
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(X_train_tfidf.shape[0])
    for budget in BUDGETS:
        idx = shuffled[:budget]
        X_lab = X_train_tfidf[idx]
        y_lab = y_train[idx]

        classifier = LinearSVC(random_state=seed)
        classifier.fit(X_lab, y_lab)
        y_pred = classifier.predict(X_test_tfidf)

        run_row = {
            "seed": seed,
            "n_labels": budget,
            "classes_seen": len(set(y_lab)),
            "macro_f1": f1_score(y_test, y_pred, average='macro'),
            "accuracy": accuracy_score(y_test, y_pred)
            }

        runs.append(run_row)


pd.DataFrame(runs).to_csv("results/random_curve_20ng.csv", index=False)
