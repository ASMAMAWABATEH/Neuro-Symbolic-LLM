"""
Freezing base-model nodes during FabricPC training.

`fabricpc.graph.graph_net.compute_local_weight_gradients` (v0.3.0) computes
a gradient for every node with in_degree > 0, uniformly, and
`fabricpc.training.train.train_step` applies `optax.apply_updates` to the
whole `GraphParams` pytree. There is no built-in notion of "this node's
weights are frozen" -- if the base transformer is represented as FabricPC
nodes (see a3_residual.py's docstring), something has to stop those nodes'
weights from moving during A3 training, or "frozen base" stops being true
after the first optimizer step.

`optax.masked` is the right primitive: unlike zeroing the gradient (which,
for stateful optimizers like Adam, still lets momentum/variance terms decay
the parameters over time), `optax.masked(inner, mask)` passes masked-out
leaves through completely unchanged -- a true freeze.
"""
from __future__ import annotations

from typing import Iterable

import optax

from fabricpc.core.types import GraphParams, NodeParams


def build_trainable_mask(
    params: GraphParams,
    trainable_node_names: Iterable[str],
) -> GraphParams:
    """Build a boolean GraphParams pytree matching `params`'s structure,
    True for every weight/bias leaf belonging to a node in
    `trainable_node_names`, False everywhere else.

    Pass the result to `optax.masked` (see `freeze_optimizer` below) rather
    than using it to zero gradients directly.
    """
    trainable = set(trainable_node_names)
    masked_nodes = {}
    for name, node_params in params.nodes.items():
        is_trainable = name in trainable
        masked_nodes[name] = NodeParams(
            weights={k: is_trainable for k in node_params.weights},
            biases={k: is_trainable for k in node_params.biases},
        )
    return GraphParams(nodes=masked_nodes)


def freeze_optimizer(
    inner_optimizer: optax.GradientTransformation,
    params: GraphParams,
    trainable_node_names: Iterable[str],
) -> optax.GradientTransformation:
    """Wrap `inner_optimizer` so it only ever updates the nodes named in
    `trainable_node_names` (e.g. your A3 predictor/residual nodes);
    every other node's weights (the frozen base) pass through untouched.

    Usage (in run_stage_a3.py):
        base_optimizer = optax.adamw(learning_rate=lr)
        optimizer = freeze_optimizer(base_optimizer, params, a3_node_names)
        opt_state = optimizer.init(params)   # or pass straight to train_pcn
    """
    mask = build_trainable_mask(params, trainable_node_names)
    return optax.masked(inner_optimizer, mask)