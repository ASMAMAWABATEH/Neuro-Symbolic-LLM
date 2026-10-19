"""FrozenTail: the one A3-specific node in the FabricPC graph (FabricPC 0.6.0 node contract).

A3 (plan sec 7.5) is a FabricPC graph in which everything is a stock FabricPC piece except this
node:

    h0_l (clamped) --> z_l  Linear + GaussianEnergy      f_phi(h0) and the spring ||z - f||^2/2s^2
    z_l ------------> tail  FrozenTail                   h~_l = h0_l + z_l B_l, frozen model, CE
    ids (clamped) --> tail

What the node does
  * parameters: one B_l [r, d] per incoming z edge (math B_l is d x r, so `z @ B`)
  * predict:    run the frozen substrate with modify_fn adding z_l @ B_l at each intercepted layer;
                z_mu = next-token probabilities at positions 0..T-2 (position t predicts token t+1)
  * energy:     -sum log p(target) + lambda_kl * KL(p_base || p), from log-softmax (aux). FabricPC's stock
                CrossEntropyEnergy clips probabilities at 1e-7, which zeroes the gradient of
                exactly the tokens the model gets most wrong; this does not.
  * gradients:  the engine differentiates this node's energy w.r.t. each z edge (that is
                B^T grad_{h~} L through every frozen layer above) and w.r.t. B (the dB rule).
                forward_and_latent_grads is overridden only to skip the integer `ids` input,
                which autodiff cannot differentiate.

The substrate is bound to a subclass by make_frozen_tail(), so no object has to live in the node
config.
"""
from __future__ import annotations

import re
from typing import Any, Dict

import jax
import jax.numpy as jnp

from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.initializers import NormalInitializer, initialize
from fabricpc.core.types import NodeParams
from fabricpc.nodes.base import NodeBase, SlotSpec

_Z_EDGE = re.compile(r"(?:^|/)z(\d+)->")


def layer_of(edge_key: str) -> int:
    """Intercepted-layer index from a z edge key such as 'z4->tail:z'."""
    m = _Z_EDGE.search(edge_key)
    if m is None:
        raise ValueError(f"cannot read a layer index from edge key {edge_key!r}")
    return int(m.group(1))


class FrozenTail(NodeBase):
    _substrate: Any = None  # bound by make_frozen_tail()

    def __init__(self, shape, name, *, layers, hidden_size, projection_scale=0.01,
                 lambda_kl=0.0, activation=IdentityActivation(), energy=CrossEntropyEnergy(),
                 latent_init=NormalInitializer(), weight_init=None):
        # `energy` is only the declared label: the node overrides energy() below.
        super().__init__(
            shape=shape, name=name, activation=activation, energy=energy,
            latent_init=latent_init,
            weight_init=weight_init or NormalInitializer(mean=0.0, std=projection_scale),
            layers=tuple(int(l) for l in layers), hidden_size=int(hidden_size),
            lambda_kl=float(lambda_kl))

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"z": SlotSpec(name="z", is_multi_input=True),
                "ids": SlotSpec(name="ids", is_multi_input=False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config) -> NodeParams:
        """One B_l [r, d] per z edge, small and random (never zero: with B = 0 the gradient on z
        is exactly zero and inference could not move z)."""
        d = int(config["hidden_size"])
        z_edges = [k for k in input_shapes if k.endswith(":z")]
        keys = jax.random.split(key, max(len(z_edges), 1))
        weights = {}
        for k, edge_key in zip(keys, z_edges):
            r = int(input_shapes[edge_key][-1])
            weights[edge_key] = initialize(k, (r, d), weight_init)
        return NodeParams(weights=weights, biases={})

    @staticmethod
    def predict(params, inputs, state, node_info):
        sub = node_info.node_class._substrate
        layers = tuple(node_info.node_config["layers"])
        ids, z_in = None, {}
        for edge_key, x in inputs.items():
            if edge_key.endswith(":ids"):
                ids = x
            else:
                z_in[layer_of(edge_key)] = (x, params.weights[edge_key])
        ids = ids.astype(jnp.int32)

        def modify(h, layer_idx):                      # h~_l = h0_l + z_l B_l
            if layer_idx not in z_in:
                return h
            z, B = z_in[layer_idx]
            return h + (z @ B).astype(h.dtype)

        logits = sub.run_with_interception(ids, modify_fn=modify, intercept_layers=layers).logits
        logp = jax.nn.log_softmax(logits[:, :-1, :].astype(jnp.float32), axis=-1)
        base_logp = None
        if float(node_info.node_config.get("lambda_kl", 0.0)) > 0.0:
            # KL(p_base || p): the same frozen model with no injection, no gradient. It costs one
            # extra forward per evaluation and only exists when lambda_kl > 0.
            base = sub.run_with_interception(ids, intercept_layers=layers).logits
            base_logp = jax.nn.log_softmax(
                jax.lax.stop_gradient(base)[:, :-1, :].astype(jnp.float32), axis=-1)
        return jnp.exp(logp), (logp, base_logp)        # (z_mu, aux)

    @staticmethod
    def energy(params, inputs, state, aux, node_info):
        """Per-sample  -sum_t log p(y_t) + lambda_kl * sum_t KL(p_base || p)  (z_latent is the
        one-hot target clamp). The KL term is the paper's trust region; it is absent at 0."""
        if aux is None:
            return NodeBase.energy(params, inputs, state, aux, node_info)
        logp, base_logp = aux
        axes = tuple(range(1, logp.ndim))
        e = -jnp.sum(state.z_latent * logp, axis=axes)
        if base_logp is not None:
            lam = float(node_info.node_config["lambda_kl"])
            e = e + lam * jnp.sum(jnp.exp(base_logp) * (base_logp - logp), axis=axes)
        return e

    @staticmethod
    def forward_and_latent_grads(params, inputs, state, node_info, is_clamped):
        node_class = node_info.node_class
        z_inputs = {k: v for k, v in inputs.items() if k.endswith(":z")}
        ids_inputs = {k: v for k, v in inputs.items() if k.endswith(":ids")}

        def energy_fn(z_in, z_latent):
            s = state._replace(z_latent=z_latent)
            new_s = node_class.forward(params, {**z_in, **ids_inputs}, s, node_info)
            return jnp.sum(new_s.energy), new_s

        (_, new_state), (z_grads, self_grad) = jax.value_and_grad(
            energy_fn, argnums=(0, 1), has_aux=True)(z_inputs, state.z_latent)
        input_grads = dict(z_grads)
        for k, v in ids_inputs.items():                # ids are data: no gradient
            input_grads[k] = jnp.zeros(v.shape, jnp.float32)
        return new_state, input_grads, self_grad


def make_frozen_tail(substrate):
    """FrozenTail class bound to a substrate (anything with run_with_interception)."""
    return type("FrozenTail", (FrozenTail,), {"_substrate": substrate})