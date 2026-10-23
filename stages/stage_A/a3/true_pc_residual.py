"""Mathematical core for Stage A3: true predictive-coding latent residuals.

This module is intentionally independent of the experiment runner.  It defines
A3's energy, its analytical gradients, and K-step latent inference using JAX.
FabricPC graph/node wiring is kept separate so these equations can be tested
before coupling them to the framework API.

Shapes
------
h0:                  [..., hidden_dim]
z:                   [..., latent_dim]
projection B:        [hidden_dim, latent_dim]
projected residual:  z @ B.T
predictor output mu: same shape as z

The task-loss and optional base-KL callbacks receive the adapted hidden state
h_tilde = h0 + z @ B.T. If a KL term is enabled, its gradient therefore flows
through the same residual path.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array
PredictorFn = Callable[[Any, Array], Array]
HiddenLossFn = Callable[[Array], Array]


class A3Gradients(NamedTuple):
    """Gradients of the A3 energy with respect to z, predictor params, and B."""

    z: Array
    predictor_params: Any
    projection: Array


def project_residual(h0: Array, z: Array, projection: Array) -> Array:
    """Return h0 + B z, with batched row-vector convention z @ B.T."""
    return h0 + jnp.matmul(z, jnp.swapaxes(projection, -1, -2))


def prediction_energy(z: Array, mu: Array, sigma: float | Array) -> Array:
    """Compute sum_l 0.5 / sigma_l^2 * ||z_l - mu_l||^2.

    ``sigma`` must be strictly positive. It may be a scalar or broadcastable
    array. The sum includes all batch and latent dimensions, matching the
    energy definition; callers can normalize losses if their training setup
    requires mean-per-example scaling.
    """
    sigma = jnp.asarray(sigma)
    return 0.5 * jnp.sum(jnp.square(z - mu) / jnp.square(sigma))


def a3_energy(
    z: Array,
    predictor_params: Any,
    projection: Array,
    h0: Array,
    *,
    predictor_fn: PredictorFn,
    task_loss_fn: HiddenLossFn,
    sigma: float | Array = 1.0,
    base_kl_fn: HiddenLossFn | None = None,
    lambda_kl: float | Array = 0.0,
) -> Array:
    """Evaluate the A3 energy for one batch.

    E = task_loss(h0 + Bz) + 0.5/sigma^2 * ||z - f_phi(h0)||^2
        + lambda_kl * base_kl(h0 + Bz).

    The KL term is zero when ``base_kl_fn`` is None. Supplying a KL callback
    with ``lambda_kl=0`` is allowed and contributes zero energy.
    """
    mu = predictor_fn(predictor_params, h0)
    h_tilde = project_residual(h0, z, projection)
    energy = task_loss_fn(h_tilde) + prediction_energy(z, mu, sigma)
    if base_kl_fn is not None:
        energy = energy + jnp.asarray(lambda_kl) * base_kl_fn(h_tilde)
    return energy


def a3_gradients(
    z: Array,
    predictor_params: Any,
    projection: Array,
    h0: Array,
    *,
    predictor_fn: PredictorFn,
    task_loss_fn: HiddenLossFn,
    sigma: float | Array = 1.0,
    base_kl_fn: HiddenLossFn | None = None,
    lambda_kl: float | Array = 0.0,
) -> A3Gradients:
    """Compute analytical A3 gradients for z, predictor parameters, and B.

    The task + KL derivative is first computed with respect to adapted hidden
    states. The z and B derivatives then use the chain rule explicitly:
      grad_z = grad_h @ B + (z - mu) / sigma^2
      grad_B = grad_h.T @ z
    Predictor parameters receive the local prediction-error gradient only.
    This assumes the KL callback depends on h_tilde, not directly on phi or B.
    """
    mu = predictor_fn(predictor_params, h0)
    h_tilde = project_residual(h0, z, projection)

    def downstream_hidden_loss(h: Array) -> Array:
        value = task_loss_fn(h)
        if base_kl_fn is not None:
            value = value + jnp.asarray(lambda_kl) * base_kl_fn(h)
        return value

    grad_h = jax.grad(downstream_hidden_loss)(h_tilde)
    local_precision = 1.0 / jnp.square(jnp.asarray(sigma))
    grad_z = jnp.matmul(grad_h, projection) + (z - mu) * local_precision

    # For h_tilde = h0 + z @ B.T, grad_B = grad_h.T @ z.
    # einsum handles a batch dimension; flatten all leading dimensions first.
    grad_h_2d = jnp.reshape(grad_h, (-1, grad_h.shape[-1]))
    z_2d = jnp.reshape(z, (-1, z.shape[-1]))
    grad_projection = jnp.matmul(jnp.swapaxes(grad_h_2d, 0, 1), z_2d)

    def local_prediction_loss(params: Any) -> Array:
        predicted_mu = predictor_fn(params, h0)
        return prediction_energy(z, predicted_mu, sigma)

    grad_predictor_params = jax.grad(local_prediction_loss)(predictor_params)
    return A3Gradients(grad_z, grad_predictor_params, grad_projection)


def infer_latent(
    z_init: Array,
    predictor_params: Any,
    projection: Array,
    h0: Array,
    *,
    predictor_fn: PredictorFn,
    task_loss_fn: HiddenLossFn,
    num_steps: int = 4,
    step_size: float = 0.1,
    sigma: float | Array = 1.0,
    base_kl_fn: HiddenLossFn | None = None,
    lambda_kl: float | Array = 0.0,
) -> Array:
    """Run K gradient-descent inference steps over z, holding params and B fixed."""
    if num_steps < 0:
        raise ValueError("num_steps must be >= 0")
    if step_size <= 0:
        raise ValueError("step_size must be > 0")

    def step(_: int, z_current: Array) -> Array:
        grad = a3_gradients(
            z_current,
            predictor_params,
            projection,
            h0,
            predictor_fn=predictor_fn,
            task_loss_fn=task_loss_fn,
            sigma=sigma,
            base_kl_fn=base_kl_fn,
            lambda_kl=lambda_kl,
        ).z
        return z_current - step_size * grad

    return jax.lax.fori_loop(0, num_steps, step, z_init)
