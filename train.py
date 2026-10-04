"""Seed-0 toy training for superposition linearity and joint contrastive decoding.

Phase 1: next-token cross-entropy on each stream alone.
Phase 2: KL self-distillation on mixed pairs. Teacher is a stop-gradient copy
of the phase-1 weights.
Phase 3: joint contrastive loss. alpha and beta start at 1 and are trained
with the large backbone. The small model is the frozen phase-1 copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import jax
import jax.numpy as jnp
import optax
from jax import random

from superposition.decode import joint_contrastive_loss
from superposition.mix import TAU, distillation_loss, make_toy_batch
from superposition.model import copy_module, init_transformer, lm_cross_entropy


def main():
    key = random.PRNGKey(0)
    model = init_transformer(
        key,
        vocab=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        max_len=32,
    )
    tokens_a, tokens_b = make_toy_batch(batch=16, length=8)

    # 1e-2 so 40 steps actually fit the bigrams on this tiny net.
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(model)

    def phase1_loss(module):
        return 0.5 * (
            lm_cross_entropy(module, tokens_a) + lm_cross_entropy(module, tokens_b)
        )

    @jax.jit
    def phase1_step(module, state):
        loss, grads = jax.value_and_grad(phase1_loss)(module)
        updates, state = optimizer.update(grads, state, module)
        module = optax.apply_updates(module, updates)
        return module, state, loss

    for _ in range(40):
        model, opt_state, _loss = phase1_step(model, opt_state)

    teacher = copy_module(model)
    opt_state = optimizer.init(model)

    def phase2_loss(module):
        return distillation_loss(module, teacher, tokens_a, tokens_b, tau=TAU)

    kl_start = float(phase2_loss(model))
    print(f"phase2_kl_start {kl_start:.6f}")

    @jax.jit
    def phase2_step(module, state):
        loss, grads = jax.value_and_grad(phase2_loss)(module)
        updates, state = optimizer.update(grads, state, module)
        module = optax.apply_updates(module, updates)
        return module, state, loss

    for _ in range(40):
        model, opt_state, _loss = phase2_step(model, opt_state)

    kl_end = float(phase2_loss(model))
    print(f"phase2_kl_end {kl_end:.6f}")

    small = teacher
    variables = {
        "model": model,
        "alpha": jnp.asarray(1.0),
        "beta": jnp.asarray(1.0),
    }
    opt_state = optimizer.init(variables)

    def phase3_loss(state):
        return joint_contrastive_loss(
            state["model"],
            small,
            tokens_a,
            tokens_b,
            state["alpha"],
            state["beta"],
        )

    @jax.jit
    def phase3_step(state, opt_state):
        loss, grads = jax.value_and_grad(phase3_loss)(state)
        updates, opt_state = optimizer.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        return state, opt_state, loss

    for _ in range(20):
        variables, opt_state, _loss = phase3_step(variables, opt_state)

    alpha = float(variables["alpha"])
    beta = float(variables["beta"])
    print(f"alpha {alpha:.6f}")
    print(f"beta {beta:.6f}")


if __name__ == "__main__":
    main()
