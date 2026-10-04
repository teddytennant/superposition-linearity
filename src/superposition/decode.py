"""Geometric-mean obstruction and joint contrastive decoding (eq 6 and eq 7)."""

from __future__ import annotations

import jax.numpy as jnp

from superposition.mix import mix_embeddings
from superposition.model import Transformer, next_token_cross_entropy


def geometric_mean_distribution(p_a, p_b):
    """Normalized geometric mean of two next-token distributions.

    P_geo(v) = sqrt(P_A(v) * P_B(v)) / sum_u sqrt(P_A(u) * P_B(u))

    This is the distribution induced by averaging logits, and it is not the
    arithmetic mean 0.5 * (P_A + P_B) unless P_A equals P_B. The last axis
    is the vocabulary axis.
    """
    p_a = jnp.asarray(p_a)
    p_b = jnp.asarray(p_b)
    unnormalized = jnp.sqrt(jnp.clip(p_a, 0.0) * jnp.clip(p_b, 0.0))
    denom = jnp.sum(unnormalized, axis=-1, keepdims=True)
    return unnormalized / jnp.maximum(denom, jnp.asarray(1e-12, dtype=unnormalized.dtype))


def joint_contrastive_logits(ell_large_z, ell_small_a, ell_small_b, alpha, beta):
    """Pure joint contrastive map (eq 7).

    ell_A = ell_large(z) + alpha * ell_small(A) - beta * ell_small(B)
    ell_B = ell_large(z) + alpha * ell_small(B) - beta * ell_small(A)

    alpha and beta are scalars. Decoding takes argmax of each result
    independently at each position.
    """
    alpha = jnp.asarray(alpha, dtype=ell_large_z.dtype)
    beta = jnp.asarray(beta, dtype=ell_large_z.dtype)
    ell_a = ell_large_z + alpha * ell_small_a - beta * ell_small_b
    ell_b = ell_large_z + alpha * ell_small_b - beta * ell_small_a
    return ell_a, ell_b


def _as_module(module):
    if isinstance(module, Transformer):
        return module
    raise TypeError("expected a Transformer module")


def joint_contrastive_forward(large, small, tokens_a, tokens_b, alpha, beta):
    """Three forwards: large on the mixed embedding, small on A, small on B.

    The mix uses the large module's embedding table (positions included).
    The small module may be a different module, including a different depth.
    Vocabulary size must match so the logit tensors add.
    """
    large = _as_module(large)
    small = _as_module(small)
    mixed = mix_embeddings(large, tokens_a, tokens_b)
    ell_large_z = large.forward_from_embeddings(mixed)
    ell_small_a = small(tokens_a)
    ell_small_b = small(tokens_b)
    return joint_contrastive_logits(ell_large_z, ell_small_a, ell_small_b, alpha, beta)


def joint_contrastive_loss(large, small, tokens_a, tokens_b, alpha, beta):
    """Average of per-stream next-token cross-entropy on the contrastive logits.

    alpha and beta are trained jointly with the large backbone when this loss
    is differentiated with respect to both. The small module receives gradients
    only if it is the same traced object as large, or is otherwise an argument
    of the differentiated function.
    """
    ell_a, ell_b = joint_contrastive_forward(
        large, small, tokens_a, tokens_b, alpha, beta
    )
    loss_a = next_token_cross_entropy(ell_a, tokens_a)
    loss_b = next_token_cross_entropy(ell_b, tokens_b)
    return 0.5 * (loss_a + loss_b)


def generate(large, small, prefix_a, prefix_b, alpha=1.0, beta=1.0):
    """One-step joint contrastive decode.

    Embeds both prefixes with the large module, mixes them, runs the three
    forwards, and appends argmax(ell_A) and argmax(ell_B) at the last position.
    Prefixes may be [length] or [batch, length] and must share a shape.
    Returns the extended sequences in the same rank as the inputs.
    """
    large = _as_module(large)
    small = _as_module(small)
    prefix_a = jnp.asarray(prefix_a, dtype=jnp.int32)
    prefix_b = jnp.asarray(prefix_b, dtype=jnp.int32)
    squeeze = False
    if prefix_a.ndim == 1 and prefix_b.ndim == 1:
        prefix_a = prefix_a[None, :]
        prefix_b = prefix_b[None, :]
        squeeze = True
    if prefix_a.shape != prefix_b.shape:
        raise ValueError(
            f"prefixes must have the same shape, got {prefix_a.shape} and {prefix_b.shape}"
        )
    ell_a, ell_b = joint_contrastive_forward(
        large, small, prefix_a, prefix_b, alpha, beta
    )
    next_a = jnp.argmax(ell_a[:, -1, :], axis=-1).astype(jnp.int32)
    next_b = jnp.argmax(ell_b[:, -1, :], axis=-1).astype(jnp.int32)
    extended_a = jnp.concatenate([prefix_a, next_a[:, None]], axis=1)
    extended_b = jnp.concatenate([prefix_b, next_b[:, None]], axis=1)
    if squeeze:
        return extended_a[0], extended_b[0]
    return extended_a, extended_b
