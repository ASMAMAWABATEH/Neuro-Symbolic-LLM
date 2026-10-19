"""A3 on FabricPC 0.6.0: the graph and the per-batch protocol (plan sec 7.5, steps 1-4).

    1. run the frozen model once, untouched          -> h0_l           _clean_pass
    2. z_l^(0) = f_phi(h0_l)                          FabricPC's FeedforwardStateInit
    3. K steps  z <- z - eta_z * grad_z E             run_inference with InferenceSGD
    4. local updates of f_phi and B from z*           pc_weight_gradients
                                                      (means per prediction, ready for optax)

Everything except the tail node is a stock FabricPC node and engine function. The optimizer step
and logging live in experiments/run_stage_a3.py.

Shapes (T = seq_len, d = hidden size, r = latent size, V = vocabulary):
    ids (T,) int  |  h0_l (T, d)  |  z_l (T, r)  |  tail (T-1, V)
Targets are ids[:, 1:]; build_clamps one-hot encodes them to (B, T-1, V).

Energies sum over tokens, so every gradient is per token, and eta_z is a per-token step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.training.trainer import build_clamps, grad_denominator, pc_weight_gradients

from .fabric_tail import make_frozen_tail


@dataclass(frozen=True)
class A3FabricConfig:
    layers: Tuple[int, ...] = (4, 5, 6, 7, 8)   # intercepted layers: a chosen hyperparameter
    latent_size: int = 64                        # r
    K: int = 4                                   # inference steps; sweep {1, 2, 4, 8}
    eta_z: float = 0.1                           # per-token inference step
    sigma2: float = 1.0                          # precision = 1/sigma2 on the spring
    projection_scale: float = 0.01               # init std of B_l (must be > 0)
    lambda_kl: float = 0.0                       # KL trust-region weight; 0 = off (plan sec 2)
    predictor_init_std: Optional[float] = None   # init std of f_phi's weights; None = FabricPC's
                                                 # Kaiming default. Small values make z^(0) small,
                                                 # so the initial injection z B is small while B
                                                 # (and the inference gradient B^T delta) stays large


class FabricA3:
    def __init__(self, sub, cfg: A3FabricConfig, seq_len: int):
        self.sub, self.cfg, self.seq_len = sub, cfg, int(seq_len)
        self.layers = tuple(int(l) for l in cfg.layers)
        d = int(sub.architecture.hidden_size)
        T, r = self.seq_len, cfg.latent_size
        vocab = int(sub.run_with_interception(jnp.zeros((1, 2), jnp.int32)).logits.shape[-1])

        ids = IdentityNode(shape=(T,), name="ids")
        h0 = {l: IdentityNode(shape=(T, d), name=f"h0_{l}") for l in self.layers}
        pred_kw = ({} if cfg.predictor_init_std is None else
                   {"weight_init": NormalInitializer(mean=0.0, std=cfg.predictor_init_std)})
        z = {l: Linear(shape=(T, r), name=f"z{l}",
                       energy=GaussianEnergy(precision=1.0 / cfg.sigma2), **pred_kw)
             for l in self.layers}
        tail = make_frozen_tail(sub)(
            shape=(T - 1, vocab), name="tail", layers=self.layers, hidden_size=d,
            projection_scale=cfg.projection_scale, lambda_kl=cfg.lambda_kl)

        edges = [Edge(source=ids, target=tail.slot("ids"))]
        for l in self.layers:
            edges += [Edge(source=h0[l], target=z[l]), Edge(source=z[l], target=tail.slot("z"))]
        task_map = TaskMap(x=ids, y=tail, **{f"h0_{l}": h0[l] for l in self.layers})
        self.structure = graph(
            nodes=[ids, *h0.values(), *z.values(), tail], edges=edges, task_map=task_map,
            inference=InferenceSGD(eta_infer=cfg.eta_z, infer_steps=cfg.K))

    # ---------------------------------------------------------------- parameters
    def init_params(self, key):
        return initialize_params(self.structure, key)

    # ---------------------------------------------------------------- step 1
    def _clean_pass(self, ids):
        """Frozen model, untouched: (pristine h0_l, base logits)."""
        store = {}

        def record(h, layer_idx):
            if layer_idx in self.layers:
                store[layer_idx] = h
            return h

        logits = self.sub.run_with_interception(
            ids, modify_fn=record, intercept_layers=self.layers).logits
        return ({l: jax.lax.stop_gradient(store[l]) for l in self.layers},
                jax.lax.stop_gradient(logits))

    def make_batch(self, ids):
        """Batch dict keyed by task-map names, plus the base logits."""
        ids = ids.astype(jnp.int32)
        h0, base_logits = self._clean_pass(ids)
        batch = {"x": ids, "y": ids[:, 1:],
                 **{f"h0_{l}": h0[l].astype(jnp.float32) for l in self.layers}}
        return batch, base_logits

    # ---------------------------------------------------------------- energies
    def total_energy(self, state):
        """Sum of the energies of every node with in_degree > 0 (what training sums)."""
        return sum(jnp.sum(state.nodes[n].energy) for n, node in self.structure.nodes.items()
                   if node.node_info.in_degree > 0)

    # ---------------------------------------------------------------- steps 2-4
    def train_grads(self, params, ids, key):
        """One A3 batch. Returns (grads, aux); grads is the gradient of E as means per
        prediction, so the optimizer descends it. Pure in its array arguments (jit-able)."""
        batch, _ = self.make_batch(ids)
        clamps = build_clamps(batch, self.structure, clamp_target=True)
        state = initialize_graph_state(self.structure, ids.shape[0], key, clamps, params=params)
        e_before = self.total_energy(state)                 # z = f_phi(h0): the K = 0 energy
        state = run_inference(params, state, clamps, self.structure)
        grads = pc_weight_gradients(params, state, self.structure, clamps)
        aux = {
            "n_pred": grad_denominator(self.structure, clamps),
            "energy_before": e_before,
            "energy_after": self.total_energy(state),       # from the last step's forward
            "task_sum": jnp.sum(state.nodes["tail"].energy),
            "ce_sum": -jnp.sum(state.nodes["tail"].z_latent
                               * jnp.log(jnp.maximum(state.nodes["tail"].z_mu, 1e-30))),
            "z_star": {l: state.nodes[f"z{l}"].z_latent for l in self.layers},
            "pred": {l: state.nodes[f"z{l}"].z_mu for l in self.layers},
        }
        return grads, aux

    def deploy(self, params, ids, key):
        """Label-free path: z = f_phi(h0), no inference. Returns (next-token probs [B,T-1,V],
        base logits [B,T,V])."""
        batch, base_logits = self.make_batch(ids)
        clamps = build_clamps(batch, self.structure, clamp_target=False)
        state = initialize_graph_state(self.structure, ids.shape[0], key, clamps, params=params)
        return state.nodes["tail"].z_mu, base_logits

    def energy_curve(self, params, ids, key, steps=None):
        """Total energy per prediction at z^(0), z^(1), ..., z^(K): the per-step curve.

        Eager diagnostic (not jitted). Each inference_step evaluates every node's energy at the
        CURRENT z before updating it, so K+1 calls give the energy at z^(0)..z^(K) exactly.
        curve[0] is the K = 0 energy (z = f_phi(h0)); the training aux fields are one step stale.
        """
        batch, _ = self.make_batch(ids)
        clamps = build_clamps(batch, self.structure, clamp_target=True)
        state = initialize_graph_state(self.structure, ids.shape[0], key, clamps, params=params)
        inference = self.structure.config["inference"]
        cls, config = type(inference), inference.config
        n = grad_denominator(self.structure, clamps)
        steps = int(config["infer_steps"]) if steps is None else int(steps)
        curve = []
        for _ in range(steps + 1):
            state = cls.inference_step(params, state, clamps, self.structure, config)
            curve.append(float(self.total_energy(state)) / n)
        return curve

    # ---------------------------------------------------------------- views of params
    def B(self, params, l: int):
        return params.nodes["tail"].weights[f"z{l}->tail:z"]