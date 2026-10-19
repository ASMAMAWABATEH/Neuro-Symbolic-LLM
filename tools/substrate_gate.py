"""Can the frozen substrate sit inside FabricPC's engine?

FabricPC's engine differentiates a node with jax.value_and_grad, runs inference inside
jax.lax.fori_loop, and jits the trainer. A native A3 node would call the substrate inside all
three, so each transformation is tested here, plus an autodiff-vs-finite-difference check and
frozen-parameter integrity. Run in the project environment:

    PYTHONPATH=. python tools/substrate_gate.py --model gpt2
"""
import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from frozenllm.substrate import FrozenSubstrate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--layers", type=int, nargs="+", default=[2, 3])
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=16)
    a = p.parse_args()
    layers = tuple(a.layers)

    sub = FrozenSubstrate(a.model, intercept_layers=layers)
    d = sub.architecture.hidden_size
    ids = jax.random.randint(jax.random.PRNGKey(0), (a.batch, a.seq), 0, 1000)
    print(sub, "| devices:", jax.devices())

    def loss_of(delta):
        def fn(h, l):
            return h + delta[l].astype(h.dtype) if l in delta else h
        logits = sub.run_with_interception(ids, modify_fn=fn, intercept_layers=layers).logits
        return sub.compute_loss(logits, ids)

    zero = {l: jnp.zeros((a.batch, a.seq, d), jnp.float32) for l in layers}

    results = []

    def check(name, fn):
        t0 = time.perf_counter()
        try:
            detail, ok = fn(), True
        except Exception as e:
            detail, ok = f"{type(e).__name__}: {str(e)[:250]}", False
        results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}  ({time.perf_counter() - t0:.1f}s)")

    def t_hook():
        seen = []

        def spy(h, l):
            seen.append((l, type(h).__name__, tuple(h.shape), str(h.dtype)))
            return h
        sub.run_with_interception(ids, modify_fn=spy, intercept_layers=layers)
        assert [s[0] for s in seen] == list(layers), seen
        assert all(s[2] == (a.batch, a.seq, d) for s in seen), seen
        return f"modify_fn sees {seen[0]}"

    def t_grad():
        L, g = jax.value_and_grad(loss_of)(zero)
        n = {l: float(jnp.linalg.norm(g[l])) for l in layers}
        assert all(np.isfinite(v) and v > 0 for v in n.values()), n
        return f"loss={float(L):.4f}, grad norms={ {l: round(v, 4) for l, v in n.items()} }"

    def t_det():
        l1, l2 = float(loss_of(zero)), float(loss_of(zero))
        assert abs(l1 - l2) < 1e-5, f"loss differs between identical calls: {l1} vs {l2} (dropout on?)"
        return f"same input twice: {l1:.6f} and {l2:.6f}"

    def t_fd():
        # Along u = g/|g| the true directional derivative is |g|, far above float32 noise
        # (a random direction in this ~50k-dim space gives ~1e-4, which float32 cannot resolve).
        g = jax.grad(loss_of)(zero)
        gn = float(jnp.sqrt(sum(jnp.sum(g[l] ** 2) for l in layers)))
        u = {l: g[l] / gn for l in layers}
        rows, best = [], 1e9
        for eps in (1e-3, 1e-2, 1e-1):
            fd = float(loss_of({l: eps * u[l] for l in layers})
                       - loss_of({l: -eps * u[l] for l in layers})) / (2 * eps)
            rows.append(f"eps={eps:g}: {fd:.4e}")
            best = min(best, abs(fd - gn) / gn)
        assert best < 0.05, f"|g|={gn:.4e} but finite differences give " + "; ".join(rows)
        return f"|g|={gn:.4e}; finite differences " + "; ".join(rows) + f" (best rel_err {best:.1e})"

    def t_time():
        t0 = time.perf_counter()
        jax.block_until_ready(jax.grad(loss_of)(zero))
        return f"second eager forward+backward: {time.perf_counter() - t0:.1f}s"

    def t_jit():
        jax.block_until_ready(jax.jit(jax.grad(loss_of))(zero))
        return "jax.jit(jax.grad(...)) runs"

    def t_fori():
        def body(i, dlt):
            g = jax.grad(loss_of)(dlt)
            return {l: dlt[l] - 0.1 * g[l] for l in dlt}
        out = jax.lax.fori_loop(0, 2, body, zero)
        jax.block_until_ready(out)
        assert any(float(jnp.abs(out[l]).max()) > 0 for l in out)
        return "2 gradient steps inside lax.fori_loop"

    def loss_ids(delta, ids_):
        def fn(h, l):
            return h + delta[l].astype(h.dtype) if l in delta else h
        logits = sub.run_with_interception(ids_, modify_fn=fn, intercept_layers=layers).logits
        return sub.compute_loss(logits, ids_)

    def t_traced_ids():
        # Inside FabricPC the token ids reach the substrate as TRACED values (jit argument and
        # lax.fori_loop carry), not as closed-over constants as in the checks above.
        g = jax.jit(jax.grad(loss_ids))(zero, ids)
        assert all(bool(jnp.isfinite(g[l]).all()) and float(jnp.abs(g[l]).max()) > 0 for l in layers)

        def body(i, carry):
            dlt, ids_ = carry
            gg = jax.grad(loss_ids)(dlt, ids_)
            return ({l: dlt[l] - 0.1 * gg[l] for l in dlt}, ids_)
        out, _ = jax.lax.fori_loop(0, 2, body, (zero, ids))
        jax.block_until_ready(out)
        assert any(float(jnp.abs(out[l]).max()) > 0 for l in out)
        return "ids as jit argument and as fori_loop carry"

    def t_frozen():
        assert sub.verify_frozen()["params_unchanged"]
        return "base parameters unchanged after all of the above"

    check("hook is called once per layer with [B,T,d]", t_hook)
    check("gradient reaches the injected delta", t_grad)
    check("forward is deterministic", t_det)
    check("autodiff matches finite differences along the gradient", t_fd)
    check("steady-state cost (informational)", t_time)
    check("jit around grad", t_jit)
    check("grad inside lax.fori_loop (what run_inference does)", t_fori)
    check("token ids traced (jit argument + fori_loop carry)", t_traced_ids)
    check("frozen parameters unchanged", t_frozen)

    must = {"gradient reaches the injected delta", "autodiff matches finite differences along the gradient",
            "grad inside lax.fori_loop (what run_inference does)",
            "token ids traced (jit argument + fori_loop carry)", "frozen parameters unchanged"}
    blocked = [n for n, ok in results if n in must and not ok]
    print("\nNATIVE A3 BLOCKED by:" if blocked else "\nnative A3 is feasible on this substrate",
          *blocked, sep="\n  ")
    sys.exit(1 if blocked else 0)


if __name__ == "__main__":
    main()