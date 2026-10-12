"""Gradient-based latent inference for Stage A3."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class InferenceConfig:
    """Hyperparameters for K-step latent inference."""

    steps: int = 4
    learning_rate: float = 0.01
    precision: float = 1.0

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.precision <= 0:
            raise ValueError("precision must be positive")


def local_prediction_energy(
    latent: jax.Array,
    predicted_latent: jax.Array,
    precision: float,
) -> jax.Array:
    """Mean local energy across batch and sequence positions.

    Sum squared prediction errors across the latent dimension, then average
    across batch and sequence positions.
    """
    error = latent - predicted_latent
    return 0.5 * precision * jnp.mean(
        jnp.sum(jnp.square(error), axis=-1)
    )


def infer_latent(
    initial_latent: jax.Array,
    predicted_latent: jax.Array,
    task_energy_fn: Callable[[jax.Array], jax.Array],
    project_latent: Callable[[jax.Array], jax.Array],
    config: InferenceConfig,
) -> jax.Array:
    """Run K gradient-descent steps on task plus local prediction energy.

    task_energy_fn receives modified hidden states h0 + Bz.
    Its gradient must remain differentiable with respect to those hidden
    states; the caller is responsible for supplying a valid frozen-tail
    gradient path.
    """
    if initial_latent.shape != predicted_latent.shape:
        raise ValueError("Initial latent and predicted latent shapes differ")

    def total_energy(z: jax.Array) -> jax.Array:
        modified_hidden = project_latent(z)
        return (
            task_energy_fn(modified_hidden)
            + local_prediction_energy(
                z, predicted_latent, config.precision
            )
        )

    def step(z: jax.Array, _unused: None) -> tuple[jax.Array, None]:
        gradient = jax.grad(total_energy)(z)
        return z - config.learning_rate * gradient, None

    final_latent, _ = jax.lax.scan(
        step, initial_latent, xs=None, length=config.steps
    )
    return final_latent
