import mlflow

mlflow.artifacts.download_artifacts(
    artifact_uri="models:/Iris@champion",
    dst_path="model_export",
)