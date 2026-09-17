#!/usr/bin/env python3
"""
Stage A3 experiment: "True PC latent residual layer" (Section 7.5),
attached to a frozen transformer represented as FabricPC nodes.

Sweeps K (= InferenceSGD.infer_steps) per Section 7.5's headline
experimental variable, and logs energy/accuracy per K so the Stage A7
question -- "is K <= 4 practical?" -- can be read straight off the output.

=============================================================================
OPEN INTEGRATION POINT -- read before running
=============================================================================
`load_or_build_base_graph()` below is a PLACEHOLDER. It builds a small
*randomly initialized* transformer_v2 pipeline (Embedding -> MhaResidual ->
LnMlp1 -> Mlp2Residual -> VocabProjection) purely so this script is
runnable end-to-end for testing the A3 wiring and the freeze mask.

It is NOT the frozen, pretrained base model the research plan calls for.
Before this is a real A3 run, someone needs to answer: how do frozen
GPT-2/Pythia weights (loaded via frozenllm.substrate.loader) get converted
into NodeParams for these same node classes? That conversion -- weight
transposition/reshaping per node type, matching transformer_v2's exact
parameter layout (see fabricpc/nodes/transformer_v2.py) -- is a separate
piece of work from what's in this file. Swap `load_or_build_base_graph()`
for the real thing once that exists; nothing else in this script should
need to change.
=============================================================================
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import jax
import optax

from fabricpc.builder.edge import Edge
from fabricpc.builder.graph_builder import TaskMap, graph
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.types import GraphStructure
from fabricpc.graph.graph_net import initialize_params
from fabricpc.nodes.base import NodeBase
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.transformer_v2 import (
    EmbeddingNode,
    LnMlp1Node,
    Mlp2ResidualNode,
    MhaResidualNode,
    VocabProjectionNode,
)
from fabricpc.training.train import train_pcn

from fabricpc_ext.truepc import build_a3_residual, splice_residual_downstream, freeze_optimizer


def load_or_build_base_graph(
    vocab_size: int, embed_dim: int, ff_dim: int, num_heads: int, seq_len: int
) -> Tuple[List[NodeBase], List[Edge], NodeBase]:
    """PLACEHOLDER base graph -- see module docstring. Returns
    (nodes, edges, hooked_node) where `hooked_node` is the node A3's
    residual gets attached to (here: the MhaResidualNode's output).

    `tok_ids` is a genuine terminal node (in_degree=0) holding the raw,
    clamped token indices -- task_map["x"] must point HERE, not at
    `tok_in`. Terminal nodes never call their own forward() (see
    NodeBase.forward_inference's in_degree==0 branch), so if the raw
    indices were clamped directly onto the EmbeddingNode instead, its
    embedding lookup would never run.
    """
    tok_ids = IdentityNode(shape=(seq_len, 1), name="token_ids")
    tok_in = EmbeddingNode(
        shape=(seq_len, embed_dim),
        name="tokens",
        vocab_size=vocab_size,
        embed_dim=embed_dim,
    )
    mha = MhaResidualNode(
        shape=(seq_len, embed_dim), name="block0_mha", embed_dim=embed_dim, num_heads=num_heads
    )
    ln_mlp1 = LnMlp1Node(
        shape=(seq_len, ff_dim), name="block0_mlp1", embed_dim=embed_dim, ff_dim=ff_dim
    )
    mlp2 = Mlp2ResidualNode(
        shape=(seq_len, embed_dim), name="block0_mlp2", embed_dim=embed_dim, ff_dim=ff_dim
    )
    vocab = VocabProjectionNode(
        shape=(seq_len, vocab_size), name="logits", vocab_size=vocab_size, embed_dim=embed_dim
    )

    edges = [
        Edge(source=tok_ids, target=tok_in.slot("in")),
        Edge(source=tok_in, target=mha.slot("in")),
        Edge(source=mha, target=ln_mlp1.slot("in")),
        Edge(source=ln_mlp1, target=mlp2.slot("in")),
        Edge(source=mha, target=mlp2.slot("residual")),
        Edge(source=mlp2, target=vocab.slot("in")),
    ]
    nodes = [tok_ids, tok_in, mha, ln_mlp1, mlp2, vocab]
    return nodes, edges, mlp2


def load_minimal_diagnostic_graph(
    vocab_size: int, embed_dim: int, seq_len: int
) -> Tuple[List[NodeBase], List[Edge], NodeBase]:
    """DIAGNOSTIC ONLY -- not the real placeholder graph, not for normal
    use. Bypasses MhaResidualNode/Mlp2ResidualNode entirely: tok_ids ->
    embedding -> [A3 attaches here] -> vocab, a single-hop chain whose
    only residual/skip-connection depth (L in muPC's formula) comes from
    A3's own LinearResidual node.

    Purpose: isolate whether K>=4 divergence (observed with the full
    load_or_build_base_graph, even with muPC scaling enabled and a fixed
    parameter seed across the K-sweep) is caused by fabricpc_ext/truepc's
    own A3 residual mechanism, or by MhaResidualNode's hardcoded internal
    residual (z_mu = x + mha, no explicit is_skip_connection slot -- see
    a3_residual.py's caveats) being invisible to muPC's depth accounting.
    If THIS graph stays stable at K=4/K=8 where the full graph didn't,
    the fault is isolated to MhaResidualNode/transformer_v2.py, not A3.
    """
    tok_ids = IdentityNode(shape=(seq_len, 1), name="token_ids")
    tok_in = EmbeddingNode(
        shape=(seq_len, embed_dim),
        name="tokens",
        vocab_size=vocab_size,
        embed_dim=embed_dim,
    )
    vocab = VocabProjectionNode(
        shape=(seq_len, vocab_size), name="logits", vocab_size=vocab_size, embed_dim=embed_dim
    )
    edges = [
        Edge(source=tok_ids, target=tok_in.slot("in")),
        Edge(source=tok_in, target=vocab.slot("in")),
    ]
    nodes = [tok_ids, tok_in, vocab]
    return nodes, edges, tok_in  # hook A3 onto the embedding node's output


def build_a3_graph(
    base_nodes: List[NodeBase],
    base_edges: List[Edge],
    hooked_node: NodeBase,
    task_x_node: NodeBase,
    task_y_node: NodeBase,
    z_dim: int,
    eta_z: float,
    K: int,
) -> Tuple[GraphStructure, List[str]]:
    """Attach an A3 residual at `hooked_node` and assemble the full graph.

    Returns (structure, a3_node_names) -- a3_node_names is what you pass to
    fabricpc_ext.training.freeze_optimizer so only these nodes train.
    """
    predictor, residual, new_edges = build_a3_residual(
        hooked_node, layer_name="hook0", z_dim=z_dim, eta_z=eta_z
    )

    # Edges that currently read FROM hooked_node (excluding the ones
    # build_a3_residual just created, which must keep reading the real h0).
    unspliced_new_edge_ids = {id(e) for e in new_edges}
    downstream = [
        e
        for e in base_edges
        if e.source is hooked_node and id(e) not in unspliced_new_edge_ids
    ]
    other_edges = [e for e in base_edges if e not in downstream]
    spliced = splice_residual_downstream(downstream, hooked_node, residual)

    all_nodes = base_nodes + [predictor, residual]
    all_edges = other_edges + spliced + new_edges

    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=task_x_node, y=task_y_node),
        inference=InferenceSGD(eta_infer=eta_z, infer_steps=K),
        # VocabProjectionNode uses SoftmaxActivation + KLDivergenceEnergy
        # (softmax+CE-style) -- MuPCConfig's own docstring recommends
        # include_output=False for this energy family (True is for
        # MSE/Gaussian output energies). See fabricpc/core/mupc.py.
        scaling=MuPCConfig(include_output=False),
    )
    return structure, [predictor.name, residual.name]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--ff-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--z-dim", type=int, default=16, help="A3 latent residual dim")
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--eta-z", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/a3_sweep.csv")
    p.add_argument(
        "--minimal-graph",
        action="store_true",
        help="DIAGNOSTIC: bypass MhaResidualNode/Mlp2ResidualNode, attach A3 "
        "directly to the embedding node's output. See load_minimal_diagnostic_graph.",
    )
    args = p.parse_args()

    # TODO: replace with a real (tokenized) dataloader once available;
    # synthetic random-token loader here only exercises the wiring.
    import jax.numpy as jnp

    class _SyntheticLoader:
        def __init__(self, n_batches=20, batch_size=8):
            self.n_batches = n_batches
            self.batch_size = batch_size

        def __len__(self):
            return self.n_batches

        def __iter__(self):
            key = jax.random.PRNGKey(123)
            for _ in range(self.n_batches):
                key, kx = jax.random.split(key)
                x = jax.random.randint(
                    kx, (self.batch_size, args.seq_len, 1), 0, args.vocab_size
                )
                y = jax.nn.one_hot(
                    jnp.squeeze(x, axis=-1), args.vocab_size
                )  # trivial next-token stand-in
                yield {"x": x, "y": y}

    if args.minimal_graph:
        base_nodes, base_edges, hooked_node = load_minimal_diagnostic_graph(
            args.vocab_size, args.embed_dim, args.seq_len
        )
    else:
        base_nodes, base_edges, hooked_node = load_or_build_base_graph(
            args.vocab_size, args.embed_dim, args.ff_dim, args.num_heads, args.seq_len
        )
    tok_ids_node = base_nodes[0]  # terminal IdentityNode -- the real clamp target
    vocab_node = base_nodes[-1]
    base_node_names = {n.name for n in base_nodes}

    # Derive init_key and the loader's train_key ONCE, outside the K-loop,
    # and reuse them for every K. This is required for the sweep to
    # actually test K's effect in isolation: if each K instead got a fresh
    # random parameter draw (as an earlier version of this script did, by
    # splitting `rng` inside the loop), any difference observed between K
    # values would be confounded with random-initialization variance and
    # would not license any conclusion about K itself -- exactly the
    # comparison Section 7.5's K-sweep is meant to make.
    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    rng, train_key = jax.random.split(rng)
    rows = []

    for K in args.k_values:
        print(f"\n=== Stage A3, K={K} ===")
        structure, a3_names = build_a3_graph(
            base_nodes,
            base_edges,
            hooked_node,
            tok_ids_node,
            vocab_node,
            z_dim=args.z_dim,
            eta_z=args.eta_z,
            K=K,
        )

        params = initialize_params(structure, init_key)

        base_optimizer = optax.adamw(learning_rate=args.lr)
        optimizer = freeze_optimizer(base_optimizer, params, a3_names)

        loader = _SyntheticLoader()
        trained_params, iter_energies, _ = train_pcn(
            params,
            structure,
            loader,
            optimizer,
            config={"num_epochs": args.num_epochs},
            rng_key=train_key,
            verbose=True,
        )

        final_energy = iter_energies[-1][-1] if iter_energies and iter_energies[-1] else None
        rows.append({"K": K, "final_energy": final_energy, "a3_nodes": ",".join(a3_names)})
        print(f"K={K}: final energy={final_energy}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()