"""Latent-state initialization for Stage A3."""

import jax
import jax.numpy as jnp

from .predictor import LatentPredictor


def initialize_latent(
    predictor: LatentPredictor,
    frozen_hidden: jax.Array,
) -> jax.Array:
    """Initialize z^(0) = f_phi(h^0) without modifying the frozen hidden state."""
    return predictor(frozen_hidden)


def prediction_error(
    latent: jax.Array,
    predicted_latent: jax.Array,
) -> jax.Array:
    """Return the local A3 prediction error z - f_phi(h^0)."""
    if latent.shape != predicted_latent.shape:
        raise ValueError(
            f"Latent shape {latent.shape} does not match predictor output "
            f"shape {predicted_latent.shape}"
        )
    return latent - predicted_latent
