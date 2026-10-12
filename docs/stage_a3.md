# Stage A3: Predictive-Coding Latent Residual Adaptation

## 1. Purpose

This document records the current Stage A3 implementation on
`feature/stage-a3`.

The implementation explores predictive-coding adaptation of a latent
residual while keeping the underlying language-model substrate frozen.
It contains a generic JAX training path and a separate path that integrates
with the frozen language-model substrate.

**Status:** The training modules and tests exist. The experiment runner
(`experiments/run_stage_a3.py`) and configuration
(`configs/stage_A/a3_true_pc.yaml`) are currently empty. A complete,
configuration-driven experiment is therefore not yet available.

## 2. Mathematical formulation

Let:

- `h0` be the hidden representation from the frozen substrate.
- `z` be the inferred latent state.
- `B` be the latent-to-hidden residual projection.
- `f_phi` be the latent predictor.
- `sigma^2` be the prediction-error variance.
- `L_task` be the downstream task loss.

The residual-modified hidden representation is

    h_tilde = h0 + B z

The minimal A3 energy is

    E = L_task + (1 / (2 sigma^2)) ||z - f_phi(h0)||^2

The latent-state gradient is

    grad_z E = B^T grad_h_tilde L_task
               + (1 / sigma^2) (z - f_phi(h0))

The latent is updated for K inference steps:

    z_next = z - eta_z grad_z E

The predictor is trained using the local prediction error, while the
projection is updated using the downstream task-loss signal. The frozen
substrate parameters must remain unchanged.

This is the minimal formulation currently targeted by the implementation;
additional objective terms and experiment-level policies are not implied.

## 3. Current module layout

### `stages/stage_A/a3/latent.py`

Provides latent initialization and prediction-error calculation.

### `stages/stage_A/a3/predictor.py`

Defines `LatentPredictor`, an affine predictor with a weight matrix and
bias that predicts the latent from the frozen hidden representation.

### `stages/stage_A/a3/residual.py`

Defines `LatentResidual`, which projects the latent into hidden space and
adds the resulting residual to the original hidden representation.

### `stages/stage_A/a3/inference.py`

Defines `InferenceConfig`, the local prediction-energy calculation, and
iterative latent inference. The generic inference path uses JAX operations
and `jax.lax.scan`.

### `stages/stage_A/a3/training.py`

Implements the generic A3 training step. It infers the latent, updates the
predictor using the local prediction objective, and updates the residual
projection using the task-loss gradient.

### `stages/stage_A/a3/substrate_training.py`

Implements the substrate-integrated training path. It obtains hidden
representations from the frozen substrate, applies the latent residual at
a selected layer, evaluates the downstream task loss, and updates the
latent, predictor, and residual projection.

The substrate-integrated path uses a Python loop for latent inference
rather than staging the TorchAX-backed model through `jax.lax.scan`.

### `stages/stage_A/a3/__init__.py`

Exports the generic A3 API, including the predictor, residual, inference
configuration, and generic training step. The substrate-integrated training
function is currently imported directly from `substrate_training.py`.

### `tests/stages/stage_A/a3/test_training.py`

Contains tests for the generic A3 training path.

## 4. Frozen-substrate integration

The substrate-integrated implementation is intended to adapt the residual
path without training the language-model parameters.

The integration uses the substrate's hidden-state interception mechanism
to obtain the original hidden representation and apply the residual at the
selected layer. The task loss is evaluated using the resulting model logits.

A successful smoke test was previously run with GPT-2, a short input
sequence, a selected hidden layer, and a small latent dimension. The test
checked that the latent and updated parameters were finite, the predictor
and projection changed, and the frozen substrate parameters remained
unchanged.

This smoke test is evidence of a working integration path, not evidence
that the full Stage A3 experiment or a training benchmark is complete.

## 5. Validation recorded so far

The following tests previously passed:

    pytest -q tests/stages/stage_A/a3/test_training.py
    # 5 passed

    pytest -q tests/frozenllm/unit/test_interception.py
    # 16 passed

A GPT-2 substrate-training smoke test also completed successfully and
verified finite outputs, updates to the predictor and residual projection,
and unchanged frozen-substrate parameters.

These results should be reproduced in the current environment before
committing.

## 6. Current limitations and pending work

- `experiments/run_stage_a3.py` is empty.
- `configs/stage_A/a3_true_pc.yaml` is empty.
- A complete experiment orchestration and configuration-driven run have
  not yet been implemented.
- FabricPC API and capability compatibility still need to be audited
  before deciding which existing mechanisms to reuse.
- FabricPC is currently installed at version 0.3.0 in the local
  environment; available package versions and dependency compatibility
  need to be investigated before changing it.
- A full experiment must wait until the implementation and metrics are
  validated.

## 7. Recommended next steps

1. Preserve and document the current implementation.
2. Review and commit the current work to `feature/stage-a3`.
3. Push the branch to the user's fork.
4. Audit the latest FabricPC release and its APIs.
5. Inspect existing FabricPC metrics and learning mechanisms.
6. Compare those capabilities with the A3 equations.
7. Refactor to reuse existing mechanisms and implement only missing pieces.
8. Run regression tests and a real GPT-2 smoke test.
9. Run the full experiment only after the implementation and metrics are
   validated.
