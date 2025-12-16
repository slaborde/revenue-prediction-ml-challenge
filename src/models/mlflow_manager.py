"""
MLFlow integration for model tracking and management.
"""
import os
import mlflow
import mlflow.sklearn
from typing import Any, Dict
import joblib


class MLFlowManager:
    """
    Manages MLFlow operations for model tracking and versioning.
    """

    def __init__(self, tracking_uri: str = None, experiment_name: str = "revenue_prediction"):
        """
        Initialize MLFlow manager.

        Args:
            tracking_uri: MLFlow tracking server URI
            experiment_name: Name of the MLFlow experiment
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def log_model_training(self,
                          model: Any,
                          model_name: str,
                          params: Dict[str, Any],
                          metrics: Dict[str, float],
                          artifacts_dir: str = None,
                          tags: Dict[str, str] = None):
        """
        Log a model training run to MLFlow.

        Args:
            model: Trained model object
            model_name: Name of the model
            params: Model hyperparameters
            metrics: Model performance metrics
            artifacts_dir: Directory with additional artifacts
            tags: Additional tags for the run
        """
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )

            # Log artifacts
            if artifacts_dir and os.path.exists(artifacts_dir):
                mlflow.log_artifacts(artifacts_dir, artifact_path="artifacts")

            # Log tags
            if tags:
                mlflow.set_tags(tags)

            run_id = mlflow.active_run().info.run_id
            print(f"Model logged with run_id: {run_id}")

            return run_id

    def load_model(self, model_name: str, version: str = "latest"):
        """
        Load a model from MLFlow registry.

        Args:
            model_name: Name of the registered model
            version: Version to load (default: "latest")

        Returns:
            Loaded model
        """
        if version == "latest":
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{version}"

        model = mlflow.sklearn.load_model(model_uri)
        return model

    def log_prediction_metrics(self, metrics: Dict[str, float]):
        """
        Log metrics from production predictions.

        Args:
            metrics: Dictionary of metrics to log
        """
        with mlflow.start_run(run_name="production_metrics"):
            mlflow.log_metrics(metrics)

    def register_model(self, run_id: str, model_name: str):
        """
        Register a model from a specific run.

        Args:
            run_id: ID of the run containing the model
            model_name: Name for the registered model
        """
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, model_name)

    def transition_model_stage(self, model_name: str, version: str, stage: str):
        """
        Transition a model version to a different stage.

        Args:
            model_name: Name of the registered model
            version: Version number
            stage: Target stage (Staging, Production, Archived)
        """
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"Model {model_name} version {version} transitioned to {stage}")


def log_training_to_mlflow(model, model_name: str, metrics: Dict[str, float],
                           params: Dict[str, Any] = None,
                           artifacts_dir: str = None):
    """
    Convenience function to log a training run to MLFlow.

    Args:
        model: Trained model
        model_name: Name of the model
        metrics: Performance metrics
        params: Model hyperparameters
        artifacts_dir: Directory with artifacts

    Returns:
        MLFlow run ID
    """
    mlflow_manager = MLFlowManager()

    # Extract model parameters if not provided
    if params is None:
        params = {}
        if hasattr(model, 'get_params'):
            params = model.get_params()

    # Log to MLFlow
    run_id = mlflow_manager.log_model_training(
        model=model,
        model_name=model_name,
        params=params,
        metrics=metrics,
        artifacts_dir=artifacts_dir,
        tags={
            'framework': 'sklearn',
            'task': 'regression',
            'target': 'revenue'
        }
    )

    return run_id
