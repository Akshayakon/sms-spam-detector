"""Train a spam detector: NLTK text cleaning -> TF-IDF -> Naive Bayes."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from train_utils import clean


# 1. Load data
df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "text"])
df["clean"] = df["text"].apply(clean)
y = (df["label"] == "spam").astype(int)
print(f"{len(df)} messages, {y.sum()} spam ({y.mean():.1%})")

# 2. Split (stratified keeps the spam ratio equal in train/test)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], y, test_size=0.2, stratify=y, random_state=42)

# 3. Pipeline + hyper-parameter search with 5-fold cross validation
pipe = Pipeline([("tfidf", TfidfVectorizer()), ("nb", MultinomialNB())])
grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "tfidf__sublinear_tf": [True, False],
    "nb__alpha": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
}
search = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
print(f"CV F1: {search.best_score_:.4f}")

# 4. Evaluate on held-out test set
model = search.best_estimator_
pred = model.predict(X_test)
print(f"\nTest accuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=4))
print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))

# 5. Save
joblib.dump(model, "spam_model.joblib")
print("\nSaved spam_model.joblib")
