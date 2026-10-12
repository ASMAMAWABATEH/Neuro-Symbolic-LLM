"""A3 latent-to-hidden residual projection."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class LatentResidual:
    """Project latent states into the frozen model's hidden dimension.

    With row-vector tensors:
        residual = z @ projection
        modified_hidden = frozen_hidden + residual

    projection has shape (latent_size, hidden_size).
    """

    projection: jax.Array

    @classmethod
    def initialize(
        cls,
        key: jax.Array,
        latent_size: int,
        hidden_size: int,
        scale: float = 0.01,
        dtype: jnp.dtype = jnp.float32,
    ) -> "LatentResidual":
        if latent_size <= 0 or hidden_size <= 0:
            raise ValueError("latent_size and hidden_size must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")

        projection = scale * jax.random.normal(
            key, (latent_size, hidden_size), dtype=dtype
        )
        return cls(projection=projection)

    def project(self, latent: jax.Array) -> jax.Array:
        if latent.shape[-1] != self.projection.shape[0]:
            raise ValueError(
                f"Expected latent dimension {self.projection.shape[0]}, "
                f"got {latent.shape[-1]}"
            )
        return latent @ self.projection

    def __call__(
        self,
        frozen_hidden: jax.Array,
        latent: jax.Array,
    ) -> jax.Array:
        if frozen_hidden.shape[:-1] != latent.shape[:-1]:
            raise ValueError(
                "Frozen hidden and latent tensors must have matching "
                "batch and sequence dimensions"
            )
        residual = self.project(latent)
        if residual.shape != frozen_hidden.shape:
            raise ValueError(
                f"Residual shape {residual.shape} does not match frozen "
                f"hidden shape {frozen_hidden.shape}"
            )
        return frozen_hidden + residual
