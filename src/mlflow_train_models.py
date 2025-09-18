import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# os.makedirs("models" , exists_ok=True)
# os.makedirs("results" , exists_ok=True)

data = load_wine(as_frame=True)
X = data.data
y= data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=400, solver='lbfgs'),
    "RandomForest": RandomForestClassifier(n_estimators=80, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results=[]

#mlflow experiment ->

mlflow.set_experiment("Wine_Classification")

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc= accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")

        results.append({
            "Model":name,
            "Accuracy":acc,
            "Precision":prec,
            "Recall": rec,
            "F1-Score":f1
        })

        if name == "RandomForest":
            mlflow.log_param("n_estimators",model.n_estimators)
        elif name=="Logisticregression":
            mlflow.log_param("max_iters", model.max_iter)
        elif name=="SVM":
            mlflow.log_param("Kernel", model.kernel)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm,annot=True,fmt='d', cmap="Blues", xticklabels=data.target_names,
                    yticklabels=data.target_names)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_path = f"C:/users/omerf/mlops-assignment-1/results/confusion_matrix_{name}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        model_path = f"C:/users/omerf/mlops-assignment-1/models/mlflow_{name}.joblib"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model,name)

df_results = pd.DataFrame(results)
results_csv = "C:/users/omerf/mlops-assignment-1/reults/mlflow_model_comparison.csv"
df_results.to_csv(results_csv, index=False)
mlflow.log_artifact(results_csv)

print("\nModel Comparison reults:\n", df_results)
