# Learning log: from NLP foundations to active learning for topic classification

How to read: §0 is the foundation. Every section then opens with a **Foundations** block (general knowledge) followed by **Advanced (this project)** (what was done here, with run IDs). Numbers come from files in `results/`. A glossary is at the end.

Started 2026-10-28. Section numbers follow the pipeline order.

Research question: does uncertainty sampling reach the same accuracy with fewer labels than random sampling, and does that saving hold across domains (news, academic papers, social media)?

---

## 0. Foundations: supervised text classification

### 0.1 What a text classifier is
_To fill._

### 0.2 Train / validation / test, and why the test set is a lockbox
_To fill._

---

## 1. Data: annotated corpora across domains

#### Foundations
_To fill._

#### Advanced (this project)
_To fill._

---

## 2. Preprocessing: stop words, stemming vs lemmatising

#### Foundations
_To fill._

#### Advanced (this project)
_To fill._

---

## 3. Encoding text: bag-of-words, TF-IDF

#### Foundations
_To fill._

#### Advanced (this project)
_To fill._

---

## 4. Models: Naive Bayes, SVM, logistic regression

#### Foundations
_To fill._

#### Advanced (this project)
_To fill._

---

## 5. Active learning: the pool-based loop

#### Foundations
_To fill._

#### Advanced (this project)
_To fill._

---

## 6. Query strategies: random vs uncertainty sampling

#### Foundations
_To fill._

#### Advanced (this project)
_To fill._

---

## 7. Evaluation: learning curves, accuracy vs macro-F1, seeds

#### Foundations
_To fill._

#### Advanced (this project)
_To fill._

---

## 8. Cross-domain comparison

#### Foundations
_To fill._

#### Advanced (this project)
_To fill._

---

## Glossary
_To fill as terms appear._

---

## Further notes (things I'm writing by hand)
Pinning versions and fixed seeds makes results reproducible because version changes might "move" the underlying mechanisms, weights, or details of the program; and different seeds change the data sampling and therefore produce different results depending on the seed, making it harder to isolate variability due to the sampling or due to the system/model/pipeline itself.

TF-IDF weights refer to the importance of the given terms, as weight(word, doc). TF calculates how relevant a term is to a document, while IDF negatively calculates how common the term is across the entire corpus. The weight of the term will therefore be calculated as TF x IDF. So, words that are very common will be pushed to 0 IDF (e.g. stop words), whereas those that are rare and therefore carry more weight (e.g. specific terms) will have a higher IDF. 

Macro-F1 essentially refers to the average of all the per-class F1 scores. It is a metric that, unlike accuracy, is more robust in the face of class imbalance. Accuracy can be skewed by predicting the largest class most often since there is inherently a higher chance for correctness. Macro-F1, on the other hand, takes into account also the "wrong" predictions absorbed by the other classes and it is lower whenever precision or recall drop. For uncertainty sampling, Macro-F1 is a metric that can help the training process be more efficient by detecting correctly where uncertain boundaries lie. TLDR: accuracy drops only for the wrongly predicted class, macro-F1 drops for both the class AND the precision on the class that absorbed the wrong prediction.

Linear axis: in plotting, widths between steps are proportional to their distance (each label is given same space).
    - Would display early rise as a sudden peak and flattening of curve.
Log axis: in plotting, widths between steps are equal regardless of their relative distance (each doubling is given same space).
    - Displays as gradual increment to show that in this case, doubling the label budgets still added a 0.07-0.11 increment.