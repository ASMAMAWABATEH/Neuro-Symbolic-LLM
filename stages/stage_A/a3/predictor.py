"""A3 predictor: estimate the latent residual from frozen hidden states."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class LatentPredictor:
    """Affine predictor f_phi(h) = h @ W + b.

    W has shape (hidden_size, latent_size).
    """

    weight: jax.Array
    bias: jax.Array

    @classmethod
    def initialize(
        cls,
        key: jax.Array,
        hidden_size: int,
        latent_size: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> "LatentPredictor":
        if hidden_size <= 0 or latent_size <= 0:
            raise ValueError("hidden_size and latent_size must be positive")

        # Glorot-uniform initialization for a linear predictor.
        limit = jnp.sqrt(6.0 / (hidden_size + latent_size))
        weight = jax.random.uniform(
            key,
            (hidden_size, latent_size),
            minval=-limit,
            maxval=limit,
            dtype=dtype,
        )
        bias = jnp.zeros((latent_size,), dtype=dtype)
        return cls(weight=weight, bias=bias)

    def __call__(self, hidden: jax.Array) -> jax.Array:
        if hidden.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                f"Expected hidden dimension {self.weight.shape[0]}, "
                f"got {hidden.shape[-1]}"
            )
        return hidden @ self.weight + self.bias
