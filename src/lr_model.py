import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

df = pd.read_json("dataset.json")

text_cols = ['Summary', 'Experience', 'Skills', 'Education', 'Certifications']
df[text_cols] = df[text_cols].fillna("")
df["combined_text"] = df[text_cols].agg(" ".join, axis=1)

X = df["combined_text"]
y = df["Decision"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words='english',
        max_features=35000,
        ngram_range=(1, 2),
        min_df=1
    )),
    ("kbest", SelectKBest(chi2, k=2000)),
    ("model", LogisticRegression())
])

param_grid = {
    'model__C': [15, 20, 25],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear'],
    'model__class_weight': ['balanced', None],
    'model__max_iter': [50, 100, 200]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X, y)

print("Best Parameters:", grid.best_params_)

y_pred = grid.predict(X)
print("\nAccuracy:", accuracy_score(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))

with open('lr_model.pkl', 'wb') as file:
    pickle.dump(grid.best_estimator_, file)

print("Model saved as lr_model.pkl")