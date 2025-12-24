# ============================================
# TASK 2: RANDOM FOREST (KAGGLE) - COMPLETE
# - Uses sklearn built-in Iris dataset (no CSV needed)
# - Train/Test split
# - Metrics: Accuracy, Precision, Recall, F1
# - Confusion matrix + plots
# - Saves model as rf_iris_model.pkl for Streamlit
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

print("=" * 80)
print("TASK 2: RANDOM FOREST CLASSIFIER (IRIS) - KAGGLE")
print("=" * 80)

# --------------------------
# 1) Load Iris dataset
# --------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
target_names = iris.target_names

df = X.copy()
df["species"] = y

print("\nDataset preview:")
print(df.head())
print("\nShape:", df.shape)
print("Features:", list(X.columns))
print("Classes:", list(target_names))

# --------------------------
# 2) Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print(f"\nTrain samples: {len(X_train)} | Test samples: {len(X_test)}")

# --------------------------
# 3) Train Random Forest
# --------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# --------------------------
# 4) Predictions + metrics
# --------------------------
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 80)
print("METRICS")
print("=" * 80)
print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"Recall   : {rec:.4f} ({rec*100:.2f}%)")
print(f"F1-score : {f1:.4f} ({f1*100:.2f}%)")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# --------------------------
# 5) Plots (confusion matrix, feature importance, metrics)
# --------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion Matrix
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
    xticklabels=target_names, yticklabels=target_names
)
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Feature Importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
sns.barplot(x=importances.values, y=importances.index, ax=axes[1], palette="viridis")
axes[1].set_title("Feature Importance")
axes[1].set_xlabel("Importance")
axes[1].set_ylabel("Feature")

# Metrics bar chart
metric_names = ["Accuracy", "Precision", "Recall", "F1"]
metric_vals = [acc, prec, rec, f1]
sns.barplot(x=metric_names, y=metric_vals, ax=axes[2], palette="Set2")
axes[2].set_ylim(0, 1.05)
axes[2].set_title("Evaluation Metrics")
axes[2].set_ylabel("Score")

for i, v in enumerate(metric_vals):
    axes[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("task2_rf_iris_results.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved plot: task2_rf_iris_results.png")

# --------------------------
# 6) Save model for Streamlit
# --------------------------
model_bundle = {
    "model": rf,
    "feature_names": list(X.columns),
    "target_names": list(target_names),
    "metrics": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }
}

with open("rf_iris_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("\nSaved model: rf_iris_model.pkl")
print("\nDownload 'rf_iris_model.pkl' from Kaggle Output/Files to use in Streamlit.")
print("=" * 80)
print("TASK 2 KAGGLE PART COMPLETED")
print("=" * 80)
``_
