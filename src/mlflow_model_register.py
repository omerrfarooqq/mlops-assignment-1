import mlflow


run_id = "d0cb7c675a8948a0ad35964c9d686b27"
model_uri = f"runs:/{run_id}/RandomForest"

result = mlflow.register_model(
    model_uri=model_uri,
    name="WineClassifierModel"
)

print(f"Model registered as {result.name}, version {result.version}")
