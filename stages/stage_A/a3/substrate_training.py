"""Stage A3 training through the real frozen language-model substrate."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .inference import InferenceConfig, local_prediction_energy
from .predictor import LatentPredictor
from .residual import LatentResidual
from .training import A3TrainingConfig, A3TrainStepResult


def a3_substrate_train_step(
    substrate,
    input_ids: jax.Array,
    labels: jax.Array,
    layer: int,
    predictor: LatentPredictor,
    residual: LatentResidual,
    config: A3TrainingConfig,
) -> A3TrainStepResult:
    """Perform one A3 training step using the complete frozen-model loss.

    The substrate parameters remain frozen. Only the predictor and residual
    projection are updated.

    Each latent-inference step evaluates the candidate latent through the
    intercepted Transformer and computes the resulting language-model loss.

    Args:
        substrate: Initialized FrozenSubstrate instance.
        input_ids: Token IDs, shaped [batch, sequence].
        labels: Language-model labels, shaped [batch, sequence].
        layer: Transformer layer where the residual is injected.
        predictor: Current latent predictor.
        residual: Current latent-to-hidden projection.
        config: Inference and parameter-learning configuration.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if labels.shape != input_ids.shape:
        raise ValueError("labels must have the same shape as input_ids")

    # Capture the pristine hidden state without applying a residual.
    baseline = substrate.run_with_interception(
        input_ids,
        modify_fn=lambda hidden, layer_idx: hidden,
        intercept_layers=[layer],
    )
    h0 = baseline.intermediates[layer]

    if predictor.weight.shape[0] != h0.shape[-1]:
        raise ValueError("Predictor hidden dimension does not match substrate")
    if residual.projection.shape[1] != h0.shape[-1]:
        raise ValueError("Residual hidden dimension does not match substrate")
    if predictor.weight.shape[1] != residual.projection.shape[0]:
        raise ValueError("Predictor and residual latent dimensions differ")
    if predictor.bias.shape != (predictor.weight.shape[1],):
        raise ValueError("Predictor bias shape is inconsistent")

    predicted_latent = predictor(h0)

    def task_loss_for(
        latent: jax.Array,
        projection: jax.Array,
    ) -> jax.Array:
        """Run the candidate residual through the complete frozen model."""

        def modify_hidden(hidden, layer_idx):
            if layer_idx == layer:
                return hidden + latent @ projection
            return hidden

        result = substrate.run_with_interception(
            input_ids,
            modify_fn=modify_hidden,
            intercept_layers=[layer],
        )
        return substrate.compute_loss(result.logits, labels)

    def total_energy(latent: jax.Array) -> jax.Array:
        return (
            task_loss_for(latent, residual.projection)
            + local_prediction_energy(
                latent,
                predicted_latent,
                config.inference.precision,
            )
        )

    # Use a Python loop deliberately: it avoids assuming that the
    # TorchAX-backed model can be staged inside jax.lax.scan.
    latent = predicted_latent
    for _ in range(config.inference.steps):
        latent_gradient = jax.grad(total_energy)(latent)
        latent = (
            latent
            - config.inference.learning_rate * latent_gradient
        )

    # Evaluate the inferred latent using the current projection.
    task_loss = task_loss_for(latent, residual.projection)
    if task_loss.shape != ():
        raise ValueError("Substrate task loss must be a scalar")

    prediction_energy = local_prediction_energy(
        latent,
        predicted_latent,
        config.inference.precision,
    )
    total_energy_value = task_loss + prediction_energy

    # Update predictor parameters while holding the inferred latent fixed.
    def predictor_energy(weight, bias):
        prediction = h0 @ weight + bias
        return local_prediction_energy(
            latent,
            prediction,
            config.inference.precision,
        )

    weight_gradient, bias_gradient = jax.grad(
        predictor_energy, argnums=(0, 1)
    )(predictor.weight, predictor.bias)

    updated_predictor = LatentPredictor(
        weight=(
            predictor.weight
            - config.predictor_learning_rate * weight_gradient
        ),
        bias=(
            predictor.bias
            - config.predictor_learning_rate * bias_gradient
        ),
    )

    # Update the residual projection using the task loss only.
    # Differentiate through the complete frozen-model forward pass.
    projection_gradient = jax.grad(
        lambda projection: task_loss_for(latent, projection)
    )(residual.projection)

    updated_residual = LatentResidual(
        projection=(
            residual.projection
            - config.residual_learning_rate * projection_gradient
        )
    )

    return A3TrainStepResult(
        predictor=updated_predictor,
        residual=updated_residual,
        latent=latent,
        task_loss=task_loss,
        prediction_energy=prediction_energy,
        total_energy=total_energy_value,
    )
