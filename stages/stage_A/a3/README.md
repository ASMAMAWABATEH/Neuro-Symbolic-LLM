# Stage A3: true predictive-coding latent residual, on FabricPC

Research plan v8, section 7.5. The frozen model F0 is never modified. At chosen layers l a latent
`z_l` is inferred and injected as `h~_l = h0_l + B_l z_l`.

```
E(z, phi) = L_task(F0 + Bz) + sum_l (1/2 sigma_l^2) || z_l - f_phi,l(h0_l) ||^2 + lambda_KL * KL(p_base || p)
```

## A3 as a FabricPC graph

```
h0_l (clamped, from a clean frozen pass) --> z_l  : Linear + GaussianEnergy(precision = 1/sigma^2)
z_l --------------------------------------> tail : FrozenTail (custom node)
ids (clamped token ids) ------------------> tail
tail is clamped to the next-token targets during training
```

| Plan section 7.5 | FabricPC |
|---|---|
| `h0_l` from the frozen model (step 1) | clamped `IdentityNode`, filled by a clean substrate pass |
| `f_phi(h0)` and the spring term | one stock `Linear` node with `GaussianEnergy`; phi is its weights |
| `z^(0) = f_phi(h0)` (step 2) | `FeedforwardStateInit` (`initialize_graph_state`) |
| K steps `z <- z - eta_z * grad_z E` (step 3) | `run_inference` with `InferenceSGD(eta_infer, infer_steps=K)` |
| `B^T grad_h~ L_task` | gradient the `FrozenTail` sends back to `z_l`, through every frozen layer above |
| `dphi` (local) and `dB` (task) (step 4) | `pc_weight_gradients`: each node differentiates only its own energy |
| `lambda_KL * KL` | optional term inside the `FrozenTail` energy |

The only A3-specific FabricPC code is `FrozenTail`. It owns `B_l`, runs the frozen substrate with
`modify_fn` adding `z_l @ B_l` at the intercepted layers, and returns next-token probabilities.

## Conventions

- Shapes (T tokens, d hidden, r latent, V vocabulary): ids `(T,)`, h0 `(T, d)`, z `(T, r)`,
  tail `(T-1, V)`. Position t predicts token t+1; targets are `ids[:, 1:]`, one-hot encoded by
  FabricPC's `build_clamps`.
- Energies sum over tokens, so gradients are per token and `eta_z` is a per-token step.
  `pc_weight_gradients` divides by the number of predictions (batch x (T-1)).
- The tail energy uses log-softmax. FabricPC's stock `CrossEntropyEnergy` clips probabilities at
  1e-7, which zeroes the gradient of the tokens the model gets most wrong.
- `B` is initialised small and random, never zero: with `B = 0` the gradient on `z` is exactly
  zero. To keep the initial injection `z B` small anyway, set `predictor_init_std` small, so
  `z^(0)` starts near zero while `B` stays large enough for inference to matter.
- Evaluation has no labels, so it uses `z = f_phi(h0)` with no inference steps.

## Files

| File | Role |
|---|---|
| `fabric_tail.py` | `FrozenTail` node |
| `fabric_model.py` | `FabricA3`: graph builder, `train_grads` (steps 1-4), `deploy`, `energy_curve` |
| `experiments/run_stage_a3.py` | wiring only: optimizer, K sweep, evaluation, logging |
| `tests/.../test_a3_fabric.py` | toy-model tests of the engine against the equations; real-GPT-2 test (opt-in) |
| `tools/substrate_gate.py`, `tools/fabricpc_probe.py` | substrate-in-FabricPC feasibility check; FabricPC version probe |

## Running (FabricPC 0.6.0)

The repository environment has FabricPC 0.3.0. A3 uses 0.6.0, installed beside it without touching
the shared environment:

```bash
pip install --no-deps --target ~/fpc060 "fabricpc==0.6.0"
PYTHONPATH=~/fpc060:. pytest tests/stages/stage_A/a3/test_a3_fabric.py -v
A3_TEST_MODEL=gpt2 PYTHONPATH=~/fpc060:. pytest tests/stages/stage_A/a3/test_a3_fabric.py -k substrate -v
PYTHONPATH=~/fpc060:. python experiments/run_stage_a3.py --model gpt2 --train-text TRAIN.txt \
    --val-text VAL.txt --K 2 --projection-scale 0.1 --predictor-init-std 0.001 --eta-z 0.5 --lr 1e-4
```

## Verification status

Passed on the project machine (CPU, GPT-2 small):
- toy-model tests (11 passed): `z^(0) = f_phi(h0)`, the K-step update
  `z <- z - eta (B^T delta + p (z - f))`, the same update with the KL term, the local `dphi` and
  `dB` gradients, energy decreases, the exact per-step energy curve, the jitted training path;
- substrate gate: gradients match finite differences (relative error 8e-5), work under `jit` and
  inside `lax.fori_loop` with traced token ids, and the base model stays frozen;
- end-to-end runs on GPT-2 with `K` in {1, 2, 4}, and with `lambda_kl` = 0.1: the KL measured
  inside the tail node (0.019, 0.031) agrees with the independently computed `val_kl_to_base`
  (0.019, 0.028); `pc_energy` decomposes as CE + spring + lambda_kl * KL to six digits; setting
  `lambda_kl = 0` reproduces the earlier run digit for digit.

## Findings so far (smoke runs on 1,550 tokens; train equals val, so no performance claim)

- A3 is sensitive to initialisation. With a default predictor and `B = 0.1`, the initial model has
  `val_ce` 10.26 against 4.59 for the frozen base (KL 6.0). With `predictor_init_std = 0.001` it
  starts at 4.62 (KL 0.017) and inference still lowers the energy by 1.5% to 5.3%.
- Shrinking `B` alone does not fix it: it starts 0.37 nats above the base and inference does
  almost nothing (energy drop 0.01% to 0.06%).
- `pred_err` grows during early training: the predictor lags the point inference moves `z` to.
  Whether it later falls is the first scientific question for a longer run.

## Limitations and open decisions

- No dataset is named in the plan (section 17.1: "small sequential language/modeling or
  classification tasks"); one must be chosen.
- A1 (ordinary adapter) and A2 baselines are needed for the plan's comparisons; the loss terms and
  training protocol must then be matched across A1, A2 and A3.
- Cost: about K+3 full forward/backward passes of the frozen model per step; each K compiles once
  (about two minutes for GPT-2 small). Measured on CPU: about 2.9 s per step at K=2, and
  about 3.1 s with `lambda_kl` > 0 (the base-model pass costs less than expected).
- Memory: the tail node stores several `(batch, T-1, V)` float32 arrays.
- FabricPC version: 0.6.0 is used through a side install; moving the shared environment from 0.3.0
  is a team decision. 0.3.0 is not on PyPI, so keep a copy of it before replacing it.
- `energy_after_inference` in the training aux is one update stale; `energy_curve` gives the exact
  per-step energies.