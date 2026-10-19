"""Native A3 on FabricPC 0.6.0 (plan sec 9). Run with the 0.6.0 side install first on the path:

    PYTHONPATH=~/fpc060:. pytest tests/test_a3_fabric.py -v
    A3_TEST_MODEL=gpt2 PYTHONPATH=~/fpc060:. pytest tests/test_a3_fabric.py -k substrate -v

A toy frozen model stands in for the substrate (it calls modify_fn(h, layer_idx) the way the
real one does), so no download is needed. Every expectation is computed independently of
FabricPC from the A3 equations, then compared with what FabricPC's engine actually produces:

    z^(0)        = f_phi(h0)
    z <- z - eta * (B^T delta + precision (z - f))         delta = grad of summed NLL at h~
    dE/dphi      = -precision h0^T (z - f) / N
    dE/dB        = sum_t z_t delta_t^T / N,                N = batch * (T - 1)
"""
import os
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("fabricpc.graph_assembly",
                    reason="needs FabricPC >= 0.6 (put ~/fpc060 on PYTHONPATH)")
from fabricpc.core.inference import run_inference
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training.trainer import build_clamps, pc_weight_gradients

from stages.stage_A.a3.fabric_model import A3FabricConfig, FabricA3

BS, T, D, R, V, S2, ETA, K = 2, 5, 6, 3, 7, 0.7, 0.05, 2
PREC = 1.0 / S2
N = BS * (T - 1)
LAYERS = (0, 1)
TOL = dict(atol=5e-4, rtol=1e-3)


class ToySubstrate:
    """Frozen two-layer model with the substrate's interface."""

    def __init__(self, key):
        k = jax.random.split(key, 4)
        self.E = jax.random.normal(k[0], (V, D))
        self.W0 = 0.5 * jax.random.normal(k[1], (D, D))
        self.W1 = 0.5 * jax.random.normal(k[2], (D, D))
        self.Wo = jax.random.normal(k[3], (D, V))
        self.architecture = types.SimpleNamespace(hidden_size=D, num_layers=2)

    def run_with_interception(self, input_ids, modify_fn=None, intercept_layers=None):
        mod = modify_fn or (lambda h, l: h)
        h = mod(jnp.tanh(self.E[input_ids] @ self.W0), 0)
        h = mod(jnp.tanh(h @ self.W1), 1)
        return types.SimpleNamespace(logits=h @ self.Wo)


@pytest.fixture(scope="module")
def env():
    sub = ToySubstrate(jax.random.PRNGKey(0))
    cfg = A3FabricConfig(layers=LAYERS, latent_size=R, K=K, eta_z=ETA, sigma2=S2,
                         projection_scale=0.3)
    model = FabricA3(sub, cfg, seq_len=T)
    params = model.init_params(jax.random.PRNGKey(1))
    ids = jax.random.randint(jax.random.PRNGKey(2), (BS, T), 0, V)
    batch, _ = model.make_batch(ids)
    clamps = build_clamps(batch, model.structure, clamp_target=True)
    state0 = initialize_graph_state(model.structure, BS, jax.random.PRNGKey(3), clamps,
                                    params=params)
    h0 = {l: batch[f"h0_{l}"] for l in LAYERS}
    return types.SimpleNamespace(sub=sub, model=model, params=params, ids=ids, batch=batch,
                                 clamps=clamps, state0=state0, h0=h0)


def f_of(env, l):
    W = env.params.nodes[f"z{l}"].weights[f"h0_{l}->z{l}:in"]
    b = env.params.nodes[f"z{l}"].biases["b"]
    return env.h0[l] @ W + b


def delta_sum(env, z, lam=0.0):
    """Per-token gradient of (summed NLL + lam * summed KL(p_base || p)) at h~_l: the delta in
    B^T delta."""
    probe0 = {l: jnp.zeros((BS, T, D)) for l in LAYERS}

    def loss(probe):
        def mod(h, l):
            return h + z[l] @ env.model.B(env.params, l) + probe[l]
        logits = env.sub.run_with_interception(env.ids, modify_fn=mod).logits[:, :-1]
        logp = jax.nn.log_softmax(logits, axis=-1)
        out = -jnp.sum(jnp.take_along_axis(logp, env.ids[:, 1:, None], axis=-1))
        if lam > 0:
            base_logp = jax.nn.log_softmax(
                env.sub.run_with_interception(env.ids).logits[:, :-1], axis=-1)
            out = out + lam * jnp.sum(jnp.exp(base_logp) * (base_logp - logp))
        return out

    return jax.grad(loss)(probe0)


def settle_by_hand(env, lam=0.0):
    f = {l: f_of(env, l) for l in LAYERS}
    z = dict(f)
    for _ in range(K):
        d = delta_sum(env, z, lam)
        z = {l: z[l] - ETA * (d[l] @ env.model.B(env.params, l).T + PREC * (z[l] - f[l]))
             for l in LAYERS}
    return z, f


# ------------------------------------------------------------------ structure and step 2
def test_graph_parameters(env):
    p = env.params
    for l in LAYERS:
        assert p.nodes[f"z{l}"].weights[f"h0_{l}->z{l}:in"].shape == (D, R)
        assert env.model.B(p, l).shape == (R, D)
        assert float(jnp.abs(env.model.B(p, l)).max()) > 0          # never zero-initialised
    assert not p.nodes["ids"].weights and not p.nodes["h0_0"].weights


def test_predictor_init_std_makes_z0_small(env):
    small = FabricA3(env.sub, A3FabricConfig(layers=LAYERS, latent_size=R, K=K, eta_z=ETA,
                                             sigma2=S2, projection_scale=0.3,
                                             predictor_init_std=1e-3), seq_len=T)
    p = small.init_params(jax.random.PRNGKey(1))
    for l in LAYERS:
        w_small = p.nodes[f"z{l}"].weights[f"h0_{l}->z{l}:in"]
        w_default = env.params.nodes[f"z{l}"].weights[f"h0_{l}->z{l}:in"]
        assert float(jnp.std(w_small)) < 5e-3 < float(jnp.std(w_default))
        assert float(jnp.abs(small.B(p, l)).max()) > 0.05          # B is unaffected


def test_z0_is_predictor_output_and_spring_starts_at_zero(env):
    for l in LAYERS:
        np.testing.assert_allclose(env.state0.nodes[f"z{l}"].z_latent, f_of(env, l), **TOL)
        np.testing.assert_allclose(env.state0.nodes[f"z{l}"].energy, 0.0, atol=1e-6)
    assert env.state0.nodes["tail"].z_latent.shape == (BS, T - 1, V)      # one-hot targets
    np.testing.assert_allclose(env.state0.nodes["tail"].z_latent.sum(-1), 1.0, atol=1e-6)


def test_tail_is_the_frozen_model_with_h0_plus_zB(env):
    z = {l: f_of(env, l) for l in LAYERS}
    logits = env.sub.run_with_interception(
        env.ids, modify_fn=lambda h, l: h + z[l] @ env.model.B(env.params, l)).logits
    probs, _ = env.model.deploy(env.params, env.ids, jax.random.PRNGKey(4))
    np.testing.assert_allclose(probs, jax.nn.softmax(logits[:, :-1], -1), **TOL)


# ------------------------------------------------------------------ step 3
def test_k_step_inference_matches_the_A3_equation(env):
    z_hand, _ = settle_by_hand(env)
    state = run_inference(env.params, env.state0, env.clamps, env.model.structure)
    for l in LAYERS:
        np.testing.assert_allclose(state.nodes[f"z{l}"].z_latent, z_hand[l], **TOL)
    moved = max(float(jnp.abs(z_hand[l] - f_of(env, l)).max()) for l in LAYERS)
    assert moved > 1e-4                                           # z actually moved


def test_kl_term_enters_the_latent_gradient(env):
    """With lambda_kl > 0 the engine must follow  z <- z - eta (B^T (delta_CE + lam delta_KL)
    + precision (z - f)), and the result must differ from the lambda_kl = 0 run."""
    lam = 0.5
    model = FabricA3(env.sub, A3FabricConfig(layers=LAYERS, latent_size=R, K=K, eta_z=ETA,
                                             sigma2=S2, projection_scale=0.3, lambda_kl=lam),
                     seq_len=T)
    batch, _ = model.make_batch(env.ids)
    clamps = build_clamps(batch, model.structure, clamp_target=True)
    state0 = initialize_graph_state(model.structure, BS, jax.random.PRNGKey(3), clamps,
                                    params=env.params)
    state = run_inference(env.params, state0, clamps, model.structure)
    z_kl, _ = settle_by_hand(env, lam)
    z_plain, _ = settle_by_hand(env, 0.0)
    for l in LAYERS:
        np.testing.assert_allclose(state.nodes[f"z{l}"].z_latent, z_kl[l], **TOL)
    assert max(float(jnp.abs(z_kl[l] - z_plain[l]).max()) for l in LAYERS) > 1e-4


def test_inference_lowers_the_energy(env):
    state = run_inference(env.params, env.state0, env.clamps, env.model.structure)
    assert float(env.model.total_energy(state)) < float(env.model.total_energy(env.state0))


def test_energy_curve_starts_at_the_K0_energy_and_decreases(env):
    curve = env.model.energy_curve(env.params, env.ids, jax.random.PRNGKey(7), steps=3)
    assert len(curve) == 4
    np.testing.assert_allclose(curve[0], float(env.model.total_energy(env.state0)) / N, **TOL)
    assert all(a >= b - 1e-6 for a, b in zip(curve, curve[1:])), curve


# ------------------------------------------------------------------ step 4
def test_local_weight_gradients_match_the_closed_forms(env):
    state = run_inference(env.params, env.state0, env.clamps, env.model.structure)
    grads = pc_weight_gradients(env.params, state, env.model.structure, env.clamps)
    zs = {l: state.nodes[f"z{l}"].z_latent for l in LAYERS}
    d = delta_sum(env, zs)
    for l in LAYERS:
        err = zs[l] - f_of(env, l)
        gz = grads.nodes[f"z{l}"]
        np.testing.assert_allclose(                       # dphi: local, spring only
            gz.weights[f"h0_{l}->z{l}:in"],
            -PREC * jnp.einsum("btd,btr->dr", env.h0[l], err) / N, **TOL)
        np.testing.assert_allclose(gz.biases["b"].reshape(-1),
                                   -PREC * err.sum((0, 1)) / N, **TOL)
        np.testing.assert_allclose(                       # dB: task only
            grads.nodes["tail"].weights[f"z{l}->tail:z"],
            jnp.einsum("btr,btd->rd", zs[l], d[l]) / N, **TOL)


def hand_energy(env, params, zs):
    """E(z, phi, B) computed from scratch: spring + summed NLL, no FabricPC involved."""
    total = 0.0
    for l in LAYERS:
        W = params.nodes[f"z{l}"].weights[f"h0_{l}->z{l}:in"]
        b = params.nodes[f"z{l}"].biases["b"]
        total = total + 0.5 * PREC * jnp.sum((zs[l] - (env.h0[l] @ W + b)) ** 2)
    logits = env.sub.run_with_interception(
        env.ids, modify_fn=lambda h, l: h + zs[l] @ params.nodes["tail"].weights[f"z{l}->tail:z"]
    ).logits[:, :-1]
    logp = jax.nn.log_softmax(logits, axis=-1)
    return total - jnp.sum(jnp.take_along_axis(logp, env.ids[:, 1:, None], axis=-1))


def test_descending_the_gradients_lowers_the_energy(env):
    """One small optimizer step on (phi, B) at fixed z* must lower E(z*, phi, B)."""
    state = run_inference(env.params, env.state0, env.clamps, env.model.structure)
    grads = pc_weight_gradients(env.params, state, env.model.structure, env.clamps)
    zs = {l: state.nodes[f"z{l}"].z_latent for l in LAYERS}
    new = jax.tree_util.tree_map(lambda p, g: p - 0.05 * g, env.params, grads)
    assert float(hand_energy(env, new, zs)) < float(hand_energy(env, env.params, zs))


# ------------------------------------------------------------------ the jitted path the runner uses
def test_train_grads_jits_and_matches_the_param_structure(env):
    grads, aux = jax.jit(env.model.train_grads)(env.params, env.ids, jax.random.PRNGKey(6))
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(env.params)
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree_util.tree_leaves(grads))
    assert float(aux["energy_after"]) < float(aux["energy_before"])
    assert int(aux["n_pred"]) == N


# ------------------------------------------------------------------ real substrate (opt-in)
@pytest.mark.skipif(not os.environ.get("A3_TEST_MODEL"), reason="set A3_TEST_MODEL=<hf id>")
def test_substrate_native_a3_runs_and_base_stays_frozen():
    from frozenllm.substrate import FrozenSubstrate
    layers = (2, 3)
    sub = FrozenSubstrate(os.environ["A3_TEST_MODEL"], intercept_layers=layers)
    model = FabricA3(sub, A3FabricConfig(layers=layers, latent_size=8, K=1, eta_z=0.05,
                                         projection_scale=0.05), seq_len=16)
    params = model.init_params(jax.random.PRNGKey(0))
    ids = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, 1000)
    grads, aux = jax.jit(model.train_grads)(params, ids, jax.random.PRNGKey(2))
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree_util.tree_leaves(grads))
    assert float(jnp.abs(grads.nodes["tail"].weights[f"z{layers[0]}->tail:z"]).max()) > 0
    assert sub.verify_frozen()["params_unchanged"]