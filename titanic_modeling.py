import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc, classification_report)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 110

def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

charts = {}

# ============ LOAD ============
df = pd.read_csv("/mnt/user-data/outputs/titanic_cleaned.csv")

# ============ FEATURE SELECTION ============
# Drop identifiers/free text that aren't predictive as raw values
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
            "HasCabin", "FamilySize", "IsAlone"]
target = "Survived"

model_df = df[features + [target]].copy()

# Encode categoricals
le_sex = LabelEncoder()
model_df["Sex"] = le_sex.fit_transform(model_df["Sex"])  # male=1, female=0 (alphabetical)

le_emb = LabelEncoder()
model_df["Embarked"] = le_emb.fit_transform(model_df["Embarked"])

X = model_df[features]
y = model_df[target]

# ============ TRAIN/TEST SPLIT ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (helps Logistic Regression converge & perform well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============ TRAIN MODELS ============
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
}

results = {}
roc_data = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "confusion_matrix": cm, "auc": roc_auc
    }
    roc_data[name] = (fpr, tpr, roc_auc)

    # Confusion matrix chart
    fig, ax = plt.subplots(figsize=(4,3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Died","Survived"], yticklabels=["Died","Survived"])
    ax.set_title(f"{name}\nConfusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    charts[f"cm_{name}"] = fig_to_b64(fig)

# ============ COMBINED ROC CURVE ============
fig, ax = plt.subplots(figsize=(6,5))
colors = {"Logistic Regression": "#3d5a80", "Decision Tree": "#e07a5f", "Random Forest": "#81b29a"}
for name, (fpr, tpr, roc_auc) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=colors[name], linewidth=2)
ax.plot([0,1],[0,1], linestyle="--", color="gray", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Model Comparison")
ax.legend(loc="lower right")
charts["roc_combined"] = fig_to_b64(fig)

# ============ ACCURACY COMPARISON ============
fig, ax = plt.subplots(figsize=(6,4))
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
bars = ax.bar(names, accs, color=["#3d5a80","#e07a5f","#81b29a"])
ax.set_ylim(0,1)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, acc+0.02, f"{acc:.1%}", ha="center", fontweight="bold")
charts["accuracy_comparison"] = fig_to_b64(fig)

# ============ FEATURE IMPORTANCE (Random Forest) ============
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(6,4))
importances.plot(kind="barh", ax=ax, color="#81b29a")
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlabel("Importance")
charts["feature_importance"] = fig_to_b64(fig)

# Save results summary + charts
with open("model_results.pkl", "wb") as f:
    pickle.dump({"results": results, "charts": charts, "features": features}, f)

print("=== MODEL RESULTS ===")
for name, r in results.items():
    print(f"\n{name}:")
    print(f"  Accuracy:  {r['accuracy']:.3f}")
    print(f"  Precision: {r['precision']:.3f}")
    print(f"  Recall:    {r['recall']:.3f}")
    print(f"  F1 Score:  {r['f1']:.3f}")
    print(f"  AUC:       {r['auc']:.3f}")

best_model = max(results, key=lambda x: results[x]["accuracy"])
print(f"\nBest model by accuracy: {best_model}")
