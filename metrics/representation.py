"""Representation / internal-KR metrics (JAX)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def residual_compressibility(delta_small: float, delta_large: float,
                             eps: float = 1e-8) -> float:
    """C_resid(D) = Delta_D(r_small) / (Delta_D(r_large) + eps)  (plan 14.2).

    Delta_D(r) is the improvement on domain D from a rank-r residual; obtain it by
    running the rank sweep r in {1,2,4,...,64} as separate runs.
    """
    return float(delta_small) / (float(delta_large) + eps)


def direction_redundancy(B: jax.Array, eps: float = 1e-8) -> float:
    """Mean |cosine| between distinct residual directions (rows of B: [rank, d]).
    0 = orthogonal directions, 1 = fully redundant."""
    U = B / (jnp.linalg.norm(B, axis=-1, keepdims=True) + eps)
    G = jnp.abs(U @ U.T)
    r = G.shape[0]
    if r < 2:
        return 0.0
    return float((jnp.sum(G) - jnp.trace(G)) / (r * (r - 1)))


def gradient_conflict(grads_a, grads_b, eps: float = 1e-12) -> float:
    """Cosine similarity between two gradient pytrees. Negative = conflict."""
    a, _ = ravel_pytree(grads_a)
    b, _ = ravel_pytree(grads_b)
    return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + eps))