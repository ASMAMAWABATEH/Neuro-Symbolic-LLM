"""Stage A3 experiment runner. Wiring only: no architecture and no loss are defined here.

    frozenllm.substrate -> stages.stage_A.a3 (FabricPC graph, modify_fn inside the tail node)
                        -> frozenllm.substrate -> logits

It loads the frozen base (frozenllm/), builds the A3 graph (stages/stage_A/a3/fabric_model.py),
applies the optimizer, evaluates, and logs metrics from metrics/. Use FabricPC 0.6.0:

    PYTHONPATH=~/fpc060:. python experiments/run_stage_a3.py --model gpt2 \
        --train-text T.txt --val-text V.txt --K 1 2 4

CPU note: one step costs about (K + 3) full forward/backward passes of the frozen model, and
each K compiles once (about a minute for GPT-2). Defaults are deliberately small.

Comparisons (plan sec 7-8): frozen base (val_base_ce, every row), the K sweep, step_time_s per K.
Not produced here: the ordinary-residual (A1) baseline, and the rank sweep behind
residual_compressibility (run once per --rank and compare val_ce_gain_vs_base).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from frozenllm.substrate import FrozenSubstrate
from metrics.performance import (cross_entropy_loss, kl_to_base, pc_energy,
                                 prediction_error, training_step_time)
from metrics.representation import direction_redundancy
from stages.stage_A.a3.fabric_model import A3FabricConfig, FabricA3


def token_batches(path, tokenizer, seq_len, batch_size, seed):
    ids = np.asarray(tokenizer(Path(path).read_text())["input_ids"], dtype=np.int32)
    x = ids[: (len(ids) // seq_len) * seq_len].reshape(-1, seq_len)
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(x))
        for i in range(0, len(x) - batch_size + 1, batch_size):
            yield jnp.asarray(x[order[i:i + batch_size]])


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train-text", required=True)
    p.add_argument("--val-text", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[4, 5, 6])
    p.add_argument("--rank", type=int, default=32, help="latent size r")
    p.add_argument("--K", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--eta-z", type=float, default=0.1, help="PER-TOKEN inference step")
    p.add_argument("--sigma2", type=float, default=1.0)
    p.add_argument("--projection-scale", type=float, default=0.01)
    p.add_argument("--lambda-kl", type=float, default=0.0,
                   help="KL trust-region weight; 0 = off (plan sec 2)")
    p.add_argument("--predictor-init-std", type=float, default=None,
                   help="small value (e.g. 0.001) starts z near 0 so the model starts near the base")
    p.add_argument("--lr", type=float, default=1e-3, help="optimizer learning rate")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--energy-curve", action="store_true",
                   help="at eval steps also log the exact energy at z^(0)..z^(K) (eager, slower)")
    p.add_argument("--out", default="results/stage_a3")
    return p.parse_args()


def run_one(args, sub, K):
    layers = tuple(args.layers)
    model = FabricA3(sub, A3FabricConfig(
        layers=layers, latent_size=args.rank, K=K, eta_z=args.eta_z, sigma2=args.sigma2,
        projection_scale=args.projection_scale, lambda_kl=args.lambda_kl,
        predictor_init_std=args.predictor_init_std), seq_len=args.seq_len)
    params = model.init_params(jax.random.PRNGKey(args.seed))
    opt = optax.adamw(args.lr)
    opt_state = opt.init(params)
    train = token_batches(args.train_text, sub.tokenizer, args.seq_len, args.batch_size, args.seed)
    val = token_batches(args.val_text, sub.tokenizer, args.seq_len, args.batch_size, args.seed + 1)
    base_key = jax.random.PRNGKey(args.seed + 100)
    val_batches = [next(val) for _ in range(args.eval_batches)]   # same batches at every eval

    @jax.jit
    def train_step(params, opt_state, ids, key):
        grads, aux = model.train_grads(params, ids, key)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, aux

    deploy = jax.jit(model.deploy)

    def evaluate(params, step):
        """Deployment mode: no labels at inference, so z = f_phi(h0) with no inference steps."""
        ce = base_ce = kl = 0.0
        for i, ids in enumerate(val_batches):
            probs, base_logits = deploy(params, ids, jax.random.fold_in(base_key, 10_000 * step + i))
            logp = jnp.log(jnp.maximum(probs, 1e-30))
            padded = jnp.concatenate([logp, jnp.zeros_like(logp[:, :1])], axis=1)
            ce += float(cross_entropy_loss(padded, ids))
            base_ce += float(cross_entropy_loss(base_logits, ids))
            kl += float(kl_to_base(base_logits[:, :-1], logp))
        n = args.eval_batches
        return {"val_ce": ce / n, "val_base_ce": base_ce / n, "val_kl_to_base": kl / n,
                "val_ce_gain_vs_base": (base_ce - ce) / n}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, times = [], []
    with open(out / f"K{K}_r{args.rank}.jsonl", "w") as log:
        # step 0: the model BEFORE any update. val_ce vs val_base_ce here is how far A3's
        # initialisation alone has moved the frozen model (they should be nearly equal).
        init = {"step": 0, "K": K, "rank": args.rank, **evaluate(params, 0)}
        print(json.dumps(init))
        log.write(json.dumps(init) + "\n")
        for step in range(1, args.steps + 1):
            ids = next(train)
            key = jax.random.fold_in(base_key, step)
            (params, opt_state, aux), dt = training_step_time(train_step, params, opt_state, ids, key)
            if step > 1:
                times.append(dt)                       # step 1 includes the compile
            n = int(aux["n_pred"])
            task_ce = float(aux["ce_sum"]) / n                    # cross-entropy only
            kl_term = ((float(aux["task_sum"]) - float(aux["ce_sum"])) / (args.lambda_kl * n)
                       if args.lambda_kl > 0 else None)       # KL(p_base||p) per prediction
            perr = float(prediction_error(aux["z_star"], aux["pred"], args.sigma2))
            e_before, e_after = float(aux["energy_before"]) / n, float(aux["energy_after"]) / n
            rec = {"step": step, "K": K, "rank": args.rank, "step_time_s": dt,
                   "includes_compile": step == 1, "task_ce_at_z_star": task_ce,
                   "pred_err": perr, "kl_at_z_star": kl_term, "lambda_kl": args.lambda_kl,
                   "pc_energy": float(pc_energy(task_ce, aux["z_star"], aux["pred"], args.sigma2,
                                                kl_term, args.lambda_kl)[0]),
                   "energy_before_inference": e_before, "energy_after_inference": e_after,
                   "energy_rose_during_inference": e_after > e_before + 1e-9}
            if step % args.eval_every == 0 or step == args.steps:
                rec.update(evaluate(params, step))
                if args.energy_curve:
                    rec["energy_curve"] = model.energy_curve(params, ids, key)
                rec["direction_redundancy"] = float(np.mean(
                    [direction_redundancy(model.B(params, l)) for l in layers]))
                print(json.dumps(rec))
                rows.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()

    frozen = sub.verify_frozen()["params_unchanged"]
    assert frozen, "frozen base parameters changed during A3 training"
    return {"K": K, "rank": args.rank, "frozen_params_unchanged": frozen,
            "mean_step_time_s": float(np.mean(times)) if times else None, "final": rows[-1]}


def main():
    args = parse()
    sub = FrozenSubstrate(args.model, intercept_layers=tuple(args.layers))
    summary = [run_one(args, sub, K) for K in args.K]
    Path(args.out, f"summary_r{args.rank}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()