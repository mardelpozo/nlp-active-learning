import json
import pandas as pd
from src.data.newsgroups import load_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

SEED = 42

vectorizer = TfidfVectorizer()
classifier = LinearSVC(random_state=SEED)

pipe = make_pipeline(vectorizer, classifier)

X_train, y_train, names = load_20newsgroups(subset='train', drop_empty=True)
X_test, y_test, _ = load_20newsgroups(subset='test', drop_empty=True)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(classification_report(y_test, y_pred, target_names=names))
report = classification_report(y_test, y_pred, target_names=names, output_dict=True)

stats = {
    "seed": SEED,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "accuracy": accuracy_score(y_test, y_pred),
    "macro_f1": f1_score(y_test, y_pred, average='macro'),
    "report": report,
}

with open("results/baseline_20ng.json", "w") as f:
    json.dump(stats, f, indent=2)

# Rows = true class, columns = predicted class (sklearn convention).
cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index=names, columns=names)
# Transposed so each predicted class gets its own line; columns sum to class support.
print(cm.loc[["talk.religion.misc", "alt.atheism"]].T)
