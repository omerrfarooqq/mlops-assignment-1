import os
import joblib
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


#os.makedirs("models", exist_ok=True)
#os.makedirs("results", exist_ok=True)


data = load_wine(as_frame=True)
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


models = {
    "LogisticRegression": LogisticRegression(max_iter=500, solver="lbfgs", multi_class="auto"),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}


results = []

for name, model in models.items():

    model.fit(X_train, y_train)
    preds = model.predict(X_test)


    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })


    joblib.dump(model, f"C:/Users/omerf/mlops-assignment-1/models/{name}.joblib")


df_results = pd.DataFrame(results)
df_results.to_csv("C:/Users/omerf/mlops-assignment-1/results/model_comparison.csv", index=False)
print("\nModel Comparison Results:\n", df_results)
