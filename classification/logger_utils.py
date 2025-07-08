import os
import torch
import mlflow
import wandb
from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """
    Abstract base class for logging utilities.
    """

    def __init__(self, run_name, logger_experiment_name, **kwargs):
        self.run_name = run_name
        self.experiment_name = logger_experiment_name

    @abstractmethod
    def log_params(self, parameters):
        """
        Log parameters to the logger.

        :param parameters: Dictionary of parameters to log.
        """
        pass
    @abstractmethod
    def log_metrics(self, metrics, step=None):
        """
        Log metrics to the logger.

        :param metrics: Dictionary of metrics to log.
        :param step: Optional step number for the metric.
        """
        pass
    @abstractmethod
    def log_model(self, model, model_name, model_path=None, epoch=None):
        """
        Log the model to the logger.

        :param model: The model to log.
        :param model_name: Name of the model.
        :param model_path: Optional path to save the model.
        :param epoch: Optional epoch number for the model.
        """
        pass
    @abstractmethod
    def __enter__(self):
        return self
    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Cleanup method to close the logger.
        """
        pass

class MLflowLogger(BaseLogger):
    """MLflow implentation."""

    def __enter__(self):
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        return self

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, model_name, model_path=None, epoch=None):
        # Log the model to MLflow
        mlflow.pytorch.log_model(model, artifact_path=model_name)
        print(f"Model '{model_name}' saved in MLflow.")

        # Can also save the model locally if a path is provided
        if model_path and epoch is not None:
            epoch_path = os.path.join(model_path, f"{model_name}_epoch_{epoch}.pth")
            dir_path = os.path.dirname(epoch_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            torch.save(model.state_dict(), epoch_path)
            print(f"Model saved in: {epoch_path}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()


class WandbLogger(BaseLogger):
    """Weights & Biases implementation."""

    def __init__(self, run_name, logger_experiment_name, **kwargs):
        super().__init__(run_name, logger_experiment_name)
        self.project = logger_experiment_name
        self.config = kwargs.get("config", {})

    def __enter__(self):
        self.run = wandb.init(
            project=self.project,
            name=self.run_name,
            config=self.config
        )
        return self

    def log_params(self, params):
        wandb.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics, step=None):
        wandb.log(metrics, step=step)

    def log_model(self, model, model_name, model_path=None, epoch=None):
        # Logging the model as an artifact in W&B
        artifact = wandb.Artifact(name=model_name, type="model")

        if model_path and epoch is not None:
            state_dict_path = os.path.join(model_path, f"{model_name}_epoch_{epoch}_state_dict.pth")
            dir_path = os.path.dirname(state_dict_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            torch.save(model.state_dict(), state_dict_path)
            artifact.add_file(state_dict_path)

            full_model_path = os.path.join(dir_path, f"{model_name}_epoch_{epoch}_full_model.pth")
            torch.save(model, full_model_path)
            artifact.add_file(full_model_path)
            print(f"Model locally saved in: {state_dict_path}")

            os.remove(full_model_path)
        else:  # If no path is provided, save to a temporary location
            tmp_path = f"/tmp/{model_name}.pth"
            torch.save(model.state_dict(), tmp_path)
            artifact.add_file(tmp_path)

        self.run.log_artifact(artifact)
        print(f"Model '{model_name}' saved as an artifact in W&B.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        wandb.finish()

# Factory function to get the appropriate logger based on the logger name
def get_logger(logger_name, **kwargs):
    if logger_name == "wandb":
        return WandbLogger(**kwargs)
    elif logger_name == "mlflow":
        return MLflowLogger(**kwargs)
    else:
        raise ValueError(f"Unsupported logger: {logger_name}. Supported loggers are 'wandb' and 'mlflow'.")