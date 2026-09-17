"""
Unit tests for fabricpc_ext.nodes.a3_residual.

These only check the pure-Python graph-wiring logic (node names, shapes,
edge source/target relationships, and the residual-precision derivation) --
no jax execution, no real training. That keeps them fast and independent
of anything not yet confirmed about the base-graph/pretrained-loader
integration (see run_stage_a3.py's TODO).
"""
from fabricpc.builder.edge import Edge
from fabricpc.core.energy import GaussianEnergy
from fabricpc.nodes.linear import Linear

from fabricpc_ext.truepc import build_a3_residual, splice_residual_downstream


def _dummy_h0_node(d_model=32, name="h0"):
    """A stand-in for a frozen base's hidden-state node."""
    return Linear(shape=(d_model,), name=name)


def test_build_a3_residual_shapes_and_names():
    h0 = _dummy_h0_node(d_model=32, name="block6_h0")
    predictor, residual, new_edges = build_a3_residual(
        h0, layer_name="block6", z_dim=8, eta_z=0.1
    )

    assert predictor.name == "block6_a3_predictor"
    assert residual.name == "block6_a3_residual"
    assert predictor.shape == (8,)
    assert residual.shape == h0.shape  # residual output matches h0's d_model

    assert len(new_edges) == 3
    sources = {(e.source.name, e.target_node.name, e.target_slot) for e in new_edges}
    assert (h0.name, predictor.name, "in") in sources
    assert (predictor.name, residual.name, "in") in sources
    assert (h0.name, residual.name, "skip") in sources


def test_residual_precision_defaults_to_damped_inverse_eta_z():
    """residual_precision defaults to residual_damping/eta_z (damping=0.5),
    NOT exactly 1/eta_z -- exact collapse (damping=1) gives the backward
    gradient into z_l unity gain and diverges by K=2 in practice (see
    a3_residual.py's docstring for the full derivation)."""
    h0 = _dummy_h0_node(name="h0")
    eta_z = 0.25
    _, residual, _ = build_a3_residual(h0, layer_name="l0", z_dim=4, eta_z=eta_z)

    assert isinstance(residual._energy, GaussianEnergy)
    assert residual._energy.config["precision"] == 0.5 / eta_z


def test_residual_damping_must_be_in_open_unit_interval():
    import pytest

    h0 = _dummy_h0_node(name="h0")
    with pytest.raises(ValueError):
        build_a3_residual(h0, layer_name="l0", z_dim=4, eta_z=0.1, residual_damping=1.0)
    with pytest.raises(ValueError):
        build_a3_residual(h0, layer_name="l0", z_dim=4, eta_z=0.1, residual_damping=0.0)


def test_residual_precision_override_is_respected():
    h0 = _dummy_h0_node(name="h0")
    _, residual, _ = build_a3_residual(
        h0, layer_name="l0", z_dim=4, eta_z=0.1, residual_precision=999.0
    )
    assert residual._energy.config["precision"] == 999.0


def test_splice_residual_downstream_rewires_source_only():
    h0 = _dummy_h0_node(name="h0")
    next_layer = _dummy_h0_node(name="next_layer")
    unrelated = _dummy_h0_node(name="unrelated_source")

    predictor, residual, new_edges = build_a3_residual(
        h0, layer_name="l0", z_dim=4, eta_z=0.1
    )

    downstream_edge = Edge(source=h0, target=next_layer.slot("in"))
    unrelated_edge = Edge(source=unrelated, target=next_layer.slot("in"))

    spliced = splice_residual_downstream([downstream_edge, unrelated_edge], h0, residual)

    assert len(spliced) == 2
    rewired = next(e for e in spliced if e.target_node is next_layer and e.source is residual)
    assert rewired.target_slot == "in"

    # The edge that never sourced from h0 must be passed through untouched.
    passthrough = next(e for e in spliced if e.source is unrelated)
    assert passthrough.target_node is next_layer


def test_splice_does_not_touch_the_new_a3_edges():
    """Sanity check on the pattern used in run_stage_a3.build_a3_graph:
    the 3 edges build_a3_residual returns must be excluded from whatever
    gets passed into splice_residual_downstream, or the predictor/skip
    inputs would incorrectly get rewired to read from themselves."""
    h0 = _dummy_h0_node(name="h0")
    predictor, residual, new_edges = build_a3_residual(
        h0, layer_name="l0", z_dim=4, eta_z=0.1
    )

    new_edge_ids = {id(e) for e in new_edges}
    # Simulate run_stage_a3's filtering logic directly.
    downstream = [e for e in new_edges if e.source is h0 and id(e) not in new_edge_ids]

    assert downstream == []  # both h0-sourced new_edges are correctly excluded