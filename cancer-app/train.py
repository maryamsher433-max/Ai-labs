"""Train a tumour classifier and save it as model.pkl (+ metadata.json).

Run once:  python train.py
Uses scikit-learn's built-in Breast Cancer Wisconsin dataset (no download needed).
"""
import json
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Six easy-to-understand features from the dataset
FEATURES = [
    "mean radius",
    "mean texture",
    "mean area",
    "mean smoothness",
    "mean concavity",
    "mean symmetry",
]

data = load_breast_cancer(as_frame=True)
X = data.frame[FEATURES]
y = data.target  # 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline(
    [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred).tolist()
print(f"Test accuracy: {acc:.3f}")
print("Confusion matrix [[TN, FP], [FN, TP]] (0=malignant):", cm)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Slider ranges + averages for the frontend
meta = {
    "accuracy": round(acc, 3),
    "confusion_matrix": cm,
    "train_size": len(X_train),
    "test_size": len(X_test),
    "features": [
        {
            "name": c,
            "min": round(float(X[c].min()), 4),
            "max": round(float(X[c].max()), 4),
            "mean": round(float(X[c].mean()), 4),
        }
        for c in FEATURES
    ],
}
with open("metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Saved model.pkl and metadata.json")
