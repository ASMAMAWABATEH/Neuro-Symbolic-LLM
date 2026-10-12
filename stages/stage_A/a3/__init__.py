"""Stage A3: predictive-coding latent residual adaptation."""

from .inference import (
    InferenceConfig,
    infer_latent,
    local_prediction_energy,
)
from .latent import initialize_latent, prediction_error
from .predictor import LatentPredictor
from .residual import LatentResidual
from .training import A3TrainingConfig, A3TrainStepResult, a3_train_step

__all__ = [
    "A3TrainStepResult",
    "A3TrainingConfig",
    "InferenceConfig",
    "LatentPredictor",
    "LatentResidual",
    "a3_train_step",
    "infer_latent",
    "initialize_latent",
    "local_prediction_energy",
    "prediction_error",
]
