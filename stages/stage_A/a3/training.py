"""Training step for Stage A3's minimal predictive-coding residual mechanism.

Implements the minimal Appendix A equations:
    h_tilde = h0 + z @ B
    E = L_task(h_tilde) + 0.5 * precision * ||z - f_phi(h0)||^2

The implementation uses row-vector tensors, so B has shape
(latent_size, hidden_size).

Reduction convention:
    - local prediction energy sums over latent dimensions and averages over
      batch/sequence positions;
    - task_energy_fn must return a scalar and defines its own reduction.

These are explicit implementation conventions, not claims that the paper
prescribes these exact reductions.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .inference import InferenceConfig, infer_latent, local_prediction_energy
from .predictor import LatentPredictor
from .residual import LatentResidual


@dataclass(frozen=True)
class A3TrainingConfig:
    """Learning rates for the minimal A3 training step."""

    inference: InferenceConfig = InferenceConfig()
    predictor_learning_rate: float = 1e-3
    residual_learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        if self.predictor_learning_rate <= 0:
            raise ValueError("predictor_learning_rate must be positive")
        if self.residual_learning_rate <= 0:
            raise ValueError("residual_learning_rate must be positive")


@dataclass(frozen=True)
class A3TrainStepResult:
    """Updated A3 parameters and losses measured before the parameter updates."""

    predictor: LatentPredictor
    residual: LatentResidual
    latent: jax.Array
    task_loss: jax.Array
    prediction_energy: jax.Array
    total_energy: jax.Array


def a3_train_step(
    frozen_hidden: jax.Array,
    predictor: LatentPredictor,
    residual: LatentResidual,
    task_energy_fn: Callable[[jax.Array], jax.Array],
    config: A3TrainingConfig,
) -> A3TrainStepResult:
    """Run latent inference, then apply local predictor and residual updates.

    Args:
        frozen_hidden: Cached baseline hidden states h0, shaped [..., hidden].
        predictor: Current predictor f_phi.
        residual: Current projection B, shaped [latent, hidden].
        task_energy_fn: Differentiable scalar task loss as a function of
            modified hidden states. Its gradient must propagate through the
            frozen substrate to the injected hidden states.
        config: Inference and learning-rate settings.

    Returns:
        A result containing the updated predictor and residual, the inferred
        latent state, and losses evaluated before the parameter updates.

    Note:
        This function updates only predictor and residual parameters. It does
        not update the frozen LLM's parameters.
    """
    if frozen_hidden.ndim < 2:
        raise ValueError("frozen_hidden must include batch/position and feature axes")

    hidden_size = frozen_hidden.shape[-1]
    if predictor.weight.shape[0] != hidden_size:
        raise ValueError("Predictor hidden dimension does not match frozen_hidden")
    if residual.projection.shape[1] != hidden_size:
        raise ValueError("Residual hidden dimension does not match frozen_hidden")
    if predictor.weight.shape[1] != residual.projection.shape[0]:
        raise ValueError("Predictor latent dimension does not match residual projection")
    if predictor.bias.shape != (predictor.weight.shape[1],):
        raise ValueError("Predictor bias shape is inconsistent with predictor weight")

    # Validate the task-energy contract before inference calls jax.grad.
    # A non-scalar objective otherwise fails inside JAX with a less useful
    # gradient error.
    initial_modified_hidden = residual(
        frozen_hidden, predictor(frozen_hidden)
    )
    initial_task_loss = task_energy_fn(initial_modified_hidden)
    if initial_task_loss.shape != ():
        raise ValueError(
            "task_energy_fn must return a scalar loss; "
            f"received shape {initial_task_loss.shape}"
        )

    predicted_latent = predictor(frozen_hidden)

    def project_latent(latent: jax.Array) -> jax.Array:
        return residual(frozen_hidden, latent)

    latent = infer_latent(
        initial_latent=predicted_latent,
        predicted_latent=predicted_latent,
        task_energy_fn=task_energy_fn,
        project_latent=project_latent,
        config=config.inference,
    )

    modified_hidden = project_latent(latent)
    task_loss = task_energy_fn(modified_hidden)

    if task_loss.shape != ():
        raise ValueError(
            "task_energy_fn must return a scalar loss; "
            f"received shape {task_loss.shape}"
        )

    prediction_energy = local_prediction_energy(
        latent,
        predicted_latent,
        config.inference.precision,
    )
    total_energy = task_loss + prediction_energy

    # Local predictor learning:
    # gradient descent on 0.5 * precision * ||z - f_phi(h0)||^2.
    # z is held fixed while updating phi.
    def predictor_energy(
        weight: jax.Array, bias: jax.Array
    ) -> jax.Array:
        prediction = frozen_hidden @ weight + bias
        return local_prediction_energy(
            latent,
            prediction,
            config.inference.precision,
        )

    weight_grad, bias_grad = jax.grad(
        predictor_energy, argnums=(0, 1)
    )(predictor.weight, predictor.bias)

    updated_predictor = LatentPredictor(
        weight=predictor.weight
        - config.predictor_learning_rate * weight_grad,
        bias=predictor.bias - config.predictor_learning_rate * bias_grad,
    )

    # Local residual learning in row-vector convention:
    # dL/dB = sum over positions of z[..., :, None] * g[..., None, :],
    # where g = dL_task / d(h0 + z @ B).
    task_hidden_grad = jax.grad(task_energy_fn)(modified_hidden)
    if task_hidden_grad.shape != modified_hidden.shape:
        raise ValueError("Task-loss gradient shape does not match modified hidden")

    projection_grad = jnp.einsum(
        "...l,...h->lh",
        latent,
        task_hidden_grad,
    )

    updated_residual = LatentResidual(
        projection=residual.projection
        - config.residual_learning_rate * projection_grad,
    )

    return A3TrainStepResult(
        predictor=updated_predictor,
        residual=updated_residual,
        latent=latent,
        task_loss=task_loss,
        prediction_energy=prediction_energy,
        total_energy=total_energy,
    )
