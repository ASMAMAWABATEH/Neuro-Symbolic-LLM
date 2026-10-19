"""Performance metrics (JAX). Losses live here.

Conventions: token-mean everywhere; squared latent error is summed over the latent
dimension and averaged over tokens so it is on the same scale as a token-mean CE.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp


def cross_entropy_loss(logits: jax.Array, labels: jax.Array, ignore_index: int = -100
                       ) -> jax.Array:
    """Causal-LM CE: predicts labels[:, 1:] from logits[:, :-1]. Same as
    FrozenSubstrate.compute_loss."""
    lp = jax.nn.log_softmax(logits[:, :-1, :].astype(jnp.float32), axis=-1)
    tgt = labels[:, 1:]
    mask = tgt != ignore_index
    safe = jnp.where(mask, tgt, 0)
    ll = jnp.take_along_axis(lp, safe[..., None], axis=-1).squeeze(-1)
    return -jnp.sum(ll * mask) / jnp.maximum(1.0, jnp.sum(mask))


def kl_to_base(base_logits: jax.Array, logits: jax.Array) -> jax.Array:
    """Token-mean KL(p_base || p_modified). Gradient flows through `logits` only."""
    lp_b = jax.nn.log_softmax(jax.lax.stop_gradient(base_logits).astype(jnp.float32), -1)
    lp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    return jnp.mean(jnp.sum(jnp.exp(lp_b) * (lp_b - lp), axis=-1))


def prediction_error(z: Dict[int, jax.Array], pred: Dict[int, jax.Array],
                     sigma2: float = 1.0) -> jax.Array:
    """sum_l 1/(2 sigma^2) * mean_tokens ||z^l - f_phi,l(h0^l)||^2"""
    total = 0.0
    for l in z:
        total = total + jnp.mean(jnp.sum((z[l] - pred[l]) ** 2, axis=-1)) / (2.0 * sigma2)
    return total


def pc_energy(task_loss: jax.Array, z: Dict[int, jax.Array], pred: Dict[int, jax.Array],
              sigma2: float = 1.0, kl: Optional[jax.Array] = None,
              lambda_kl: float = 0.0) -> Tuple[jax.Array, dict]:
    """E = L_task + sum_l 1/(2 sigma^2)||z^l - f(h0^l)||^2 + lambda_KL * KL (plan 7.5).

    Returns (energy, parts); parts are jax scalars (usable as jax.grad aux).
    """
    pe = prediction_error(z, pred, sigma2)
    energy = task_loss + pe
    parts = {"task": task_loss, "pred_err": pe}
    if kl is not None:
        energy = energy + lambda_kl * kl
        parts["kl"] = kl
    parts["energy"] = energy
    return energy, parts


def training_step_time(fn: Callable, *args, **kwargs):
    """Run fn and return (result, seconds), blocking on JAX async dispatch first."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    return out, time.perf_counter() - t0