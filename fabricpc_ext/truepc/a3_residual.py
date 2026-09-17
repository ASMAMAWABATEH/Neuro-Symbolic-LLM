"""
Stage A3 — "True PC latent residual layer" (research plan, Section 7.5),
built directly out of FabricPC v0.3.0 primitives.

    h_tilde^l = h0^l + B_l z^l
    z^l ~ f_{phi,l}(h0^l)

Rather than reimplementing predictive-coding settling by hand, this maps
the paper's equation directly onto two existing FabricPC node types:

    f_{phi,l}   ->  fabricpc.nodes.linear.Linear
                    (predictor: h0^l -> initial z^l guess)

    h_tilde^l   ->  fabricpc.nodes.linear_residual.LinearResidual
                    z_mu = activation(W @ z_in + b) + x_skip
                    with activation=Identity, "in"=predictor output,
                    "skip"=h0^l  =>  z_mu = B_l @ z + h0^l, exactly.

The K-step settling loop, the energy-minimization gradient flow, and the
z-initialization (z^l <- f_{phi,l}(h0^l) via FeedforwardStateInit) all come
from FabricPC's own `InferenceSGD` / `graph()` machinery -- see
fabricpc.core.inference.InferenceSGD(eta_infer=eta_z, infer_steps=K).

IMPORTANT CAVEAT (do not skip): the LinearResidual node R used for h_tilde^l
has its OWN free latent state (R.z_latent), separate from R.z_mu = h0^l +
B_l z^l. The paper treats h_tilde^l as a deterministic readout, not
something independently inferred. R is given a high-precision GaussianEnergy
(precision = residual_damping / eta_z, damping < 1) so R.z_latent tracks
R.z_mu quickly without exact one-step collapse -- exact collapse
(damping=1) gives the backward gradient into z_l unity gain and empirically
diverges by K=2 (undamped feedback through B_l^T B_l). See
`residual_precision`'s docstring in build_a3_residual for the derivation.

What THIS module owns is just:
  1. Building the two new nodes for one hooked layer (`build_a3_residual`).
  2. Splicing the residual node into an existing base graph's edge list so
     everything downstream of h0^l now consumes h_tilde^l instead
     (`splice_residual_downstream`).

What this module deliberately does NOT own (still open -- see README /
run_stage_a3.py for the TODO): how the frozen, pretrained transformer
itself gets represented as FabricPC nodes with pretrained (non-random)
NodeParams, and how N layers get stacked into one base graph. Whatever
builds that base graph just needs to hand this module the FabricPC node
object representing h0^l at each layer you want to attach a residual to,
plus the list of Edge objects that currently read from it.
"""
from __future__ import annotations

from typing import List, Tuple

from fabricpc.builder.edge import Edge
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import KaimingInitializer, NormalInitializer
from fabricpc.nodes.base import NodeBase
from fabricpc.nodes.linear import Linear
from fabricpc.nodes.linear_residual import LinearResidual


def build_a3_residual(
    h0_node: NodeBase,
    layer_name: str,
    z_dim: int,
    eta_z: float,
    predictor_hidden_activation=None,
    residual_precision: float = None,
    residual_damping: float = 0.5,
) -> Tuple[NodeBase, NodeBase, List[Edge]]:
    """Build the predictor (f_phi,l) and write-matrix (B_l) nodes for one
    hooked layer, per Section 7.5.

    Args:
        h0_node: the existing FabricPC node whose output IS h0^l -- the
            frozen base's hidden state at this layer. Not modified.
        layer_name: unique identifier for this hooked layer, e.g. "block6".
            Used to build unique node names ("{layer_name}_a3_predictor",
            "{layer_name}_a3_residual").
        z_dim: dimensionality of the latent residual state z^l. Per
            Section 2.1 / 6.2, keep this small relative to h0_node's
            d_model -- the whole framing of the residual system is
            1e-4 to 1e-2 of the base parameter count.
        eta_z: the SAME eta_infer you pass to InferenceSGD for this graph.
            Required so residual_precision (below) can be set correctly --
            InferenceSGD.eta_infer is a single global setting shared by
            every node in the graph, so the residual node's precision is
            the only per-node lever available to control how tightly it
            tracks its own z_mu.
        predictor_hidden_activation: activation for the predictor's own
            output. Defaults to Identity -- the predictor's job is to
            produce a z^l guess, not to apply a nonlinearity on top of it.
        residual_precision: GaussianEnergy precision for the residual node
            R. Defaults to `residual_damping / eta_z`.

            CORRECTED from an earlier version of this function, which
            defaulted to exactly `1 / eta_z`. That choice forces R.z_latent
            to collapse EXACTLY onto R.z_mu every step (since
            1 - eta*precision = 0 in InferenceSGD's update) -- but it also
            gives the backward gradient this exerts on z_l an UNDAMPED
            unity gain (eta * precision_R = 1 exactly), turning the
            predictor/residual pair into an undamped feedback loop whose
            stability depends entirely on the spectral radius of B_l^T B_l.
            Empirically this diverges by K=2 (confirmed: energy went from
            ~500 at K=1 to ~1e15 at K=2 with exact collapse). Using
            `residual_damping < 1` (default 0.5) keeps R fast-tracking its
            z_mu without giving the feedback path unity gain, at the cost
            of R no longer being an *exact* readout of h_tilde^l -- a
            deliberate, documented trade-off, not an oversight.
        residual_damping: see residual_precision above. Must be in (0, 1);
            values close to 1 approach the (unstable) exact-collapse
            behavior, values well below 1 trade fidelity for stability.

    Returns:
        (predictor_node, residual_node, new_edges) where new_edges are the
        THREE edges needed to wire these two nodes to h0_node:
            h0_node -> predictor.slot("in")
            predictor -> residual.slot("in")
            h0_node -> residual.slot("skip")
        Add these to your graph's edge list. `residual_node` is what you
        splice downstream via `splice_residual_downstream` below --
        `residual_node`'s output is h_tilde^l.
    """
    d_model = h0_node.shape[-1]
    leading_shape = h0_node.shape[:-1]  # e.g. (seq_len,) for per-token nodes, () otherwise

    predictor = Linear(
        shape=leading_shape + (z_dim,),
        name=f"{layer_name}_a3_predictor",
        activation=predictor_hidden_activation or IdentityActivation(),
        use_bias=True,
        weight_init=KaimingInitializer(),
        latent_init=NormalInitializer(),
    )

    if residual_precision is None:
        if not (0.0 < residual_damping < 1.0):
            raise ValueError(
                f"residual_damping must be in (0, 1), got {residual_damping}. "
                "1.0 reproduces the undamped, empirically-unstable exact-collapse case."
            )
        residual_precision = residual_damping / eta_z

    # LinearResidual: z_mu = activation(W @ z_in + b) + x_skip.
    # activation=Identity so z_mu = B_l @ z + h0^l, matching the paper exactly
    # (no nonlinearity injected into the residual stream by the write matrix).
    # High precision (see residual_precision docstring above) forces
    # z_latent -> z_mu exactly each step, so this node acts as a faithful
    # deterministic readout of h_tilde^l rather than a loosely-coupled
    # independent latent.
    residual = LinearResidual(
        shape=h0_node.shape,  # exactly matches h0^l's shape, e.g. (seq_len, d_model)
        name=f"{layer_name}_a3_residual",
        activation=IdentityActivation(),
        energy=GaussianEnergy(precision=residual_precision),
        use_bias=True,
        weight_init=KaimingInitializer(),
        latent_init=NormalInitializer(),
    )

    new_edges = [
        Edge(source=h0_node, target=predictor.slot("in")),
        Edge(source=predictor, target=residual.slot("in")),
        Edge(source=h0_node, target=residual.slot("skip")),
    ]

    return predictor, residual, new_edges


def splice_residual_downstream(
    downstream_edges: List[Edge],
    h0_node: NodeBase,
    residual_node: NodeBase,
) -> List[Edge]:
    """Rewrite `downstream_edges` so that anything currently reading from
    `h0_node` reads from `residual_node` (i.e. h_tilde^l) instead.

    This is what actually inserts the residual into the base model's
    forward path: everything after layer l now sees h0^l + B_l z^l rather
    than the frozen base's raw hidden state.

    Args:
        downstream_edges: the base graph's existing Edge objects that
            currently have `h0_node` as their source (e.g. the edge feeding
            the next transformer block). Do NOT include the edges you just
            built in `build_a3_residual` -- those must keep reading from
            the real h0_node, not the residual.
        h0_node: the frozen base node being replaced downstream.
        residual_node: the A3 residual node from `build_a3_residual`.

    Returns:
        A new list of Edge objects with `h0_node` swapped for
        `residual_node` as source, preserving each original target/slot.
        Edges whose source isn't `h0_node` are passed through unchanged.
    """
    spliced = []
    for edge in downstream_edges:
        if edge.source is h0_node:
            spliced.append(
                Edge(
                    source=residual_node,
                    target=edge.target_node.slot(edge.target_slot),
                )
            )
        else:
            spliced.append(edge)
    return spliced