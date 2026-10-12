"""Tests for the minimal Stage A3 training step."""

import jax
import jax.numpy as jnp
import pytest

from stages.stage_A.a3 import (
    A3TrainingConfig,
    InferenceConfig,
    LatentPredictor,
    LatentResidual,
    a3_train_step,
)


def _example():
    hidden = jnp.array(
        [[[0.4, -0.2], [0.1, 0.7], [-0.5, 0.3]]],
        dtype=jnp.float32,
    )
    predictor = LatentPredictor(
        weight=jnp.array([[0.3], [-0.2]], dtype=jnp.float32),
        bias=jnp.array([0.05], dtype=jnp.float32),
    )
    residual = LatentResidual(
        projection=jnp.array([[0.2, -0.15]], dtype=jnp.float32)
    )
    target = jnp.full_like(hidden, 0.25)

    def task_energy(modified_hidden):
        return 0.5 * jnp.mean(jnp.square(modified_hidden - target))

    config = A3TrainingConfig(
        inference=InferenceConfig(
            steps=2,
            learning_rate=0.05,
            precision=1.0,
        ),
        predictor_learning_rate=0.01,
        residual_learning_rate=0.01,
    )
    return hidden, predictor, residual, task_energy, config


def test_training_step_returns_correct_shapes_and_finite_losses():
    hidden, predictor, residual, task_energy, config = _example()

    result = a3_train_step(
        hidden, predictor, residual, task_energy, config
    )

    assert result.latent.shape == (1, 3, 1)
    assert result.predictor.weight.shape == (2, 1)
    assert result.predictor.bias.shape == (1,)
    assert result.residual.projection.shape == (1, 2)

    for value in (
        result.latent,
        result.predictor.weight,
        result.predictor.bias,
        result.residual.projection,
        result.task_loss,
        result.prediction_energy,
        result.total_energy,
    ):
        assert bool(jnp.all(jnp.isfinite(value)))


def test_training_step_updates_predictor_and_residual():
    hidden, predictor, residual, task_energy, config = _example()

    result = a3_train_step(
        hidden, predictor, residual, task_energy, config
    )

    assert not jnp.allclose(result.predictor.weight, predictor.weight)
    assert not jnp.allclose(result.predictor.bias, predictor.bias)
    assert not jnp.allclose(
        result.residual.projection, residual.projection
    )


def test_training_step_does_not_mutate_input_arrays():
    hidden, predictor, residual, task_energy, config = _example()
    hidden_before = hidden.copy()
    weight_before = predictor.weight.copy()
    projection_before = residual.projection.copy()

    a3_train_step(hidden, predictor, residual, task_energy, config)

    assert jnp.array_equal(hidden, hidden_before)
    assert jnp.array_equal(predictor.weight, weight_before)
    assert jnp.array_equal(residual.projection, projection_before)


def test_non_scalar_task_energy_is_rejected():
    hidden, predictor, residual, _, config = _example()

    def invalid_energy(modified_hidden):
        return jnp.square(modified_hidden)

    with pytest.raises(ValueError, match="must return a scalar"):
        a3_train_step(
            hidden, predictor, residual, invalid_energy, config
        )


def test_predictor_and_residual_dimensions_are_checked():
    hidden, predictor, residual, task_energy, config = _example()

    wrong_predictor = LatentPredictor(
        weight=jnp.ones((3, 1)),
        bias=jnp.zeros((1,)),
    )

    with pytest.raises(ValueError, match="Predictor hidden dimension"):
        a3_train_step(
            hidden, wrong_predictor, residual, task_energy, config
        )
