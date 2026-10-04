"""Input embedding mix and KL self-distillation (eq 1 and eq 5).

Mixing happens only here, at the input embeddings. Hidden states and logits
are not averaged. The distillation temperature is 2.0 (Appendix J). The loss
multiplies KL by tau^2, the usual distillation scaling. The paper writes
KL(P_target || M(z)) without spelling that factor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from superposition.model import Transformer, embed

# Appendix J distillation temperature.
TAU = 2.0


def _as_module(module):
    if isinstance(module, Transformer):
        return module
    raise TypeError("expected a Transformer module")


def mix_embeddings(module, tokens_a, tokens_b):
    """z_t = 0.5 * (E(A)_t + E(B)_t), with positions included in E.

    A and B must have the same length. The mixed sequence is what the
    transformer consumes. Do not mix at the logit level.
    """
    module = _as_module(module)
    tokens_a = jnp.asarray(tokens_a, dtype=jnp.int32)
    tokens_b = jnp.asarray(tokens_b, dtype=jnp.int32)
    if tokens_a.shape != tokens_b.shape:
        raise ValueError(
            f"A and B must have the same shape, got {tokens_a.shape} and {tokens_b.shape}"
        )
    if tokens_a.ndim != 2:
        raise ValueError("tokens must have shape [batch, length]")
    embedded_a = embed(module.params, tokens_a)
    embedded_b = embed(module.params, tokens_b)
    return 0.5 * (embedded_a + embedded_b)


def kl_distill_from_logits(student_logits, teacher_logits_a, teacher_logits_b, tau=TAU):
    """Temperature-scaled KL from the arithmetic mean of two teacher distributions.

    P_target = 0.5 * (softmax(teacher_A / tau) + softmax(teacher_B / tau))
    L = KL(P_target || softmax(student / tau)) * tau^2

    KL is sum_v P_target(v) * (log P_target(v) - log P_student(v)), averaged
    over batch and position. tau^2 is the standard distillation multiplier
    (the paper writes the KL without that factor).
    """
    tau = jnp.asarray(tau, dtype=student_logits.dtype)
    log_p_a = jax.nn.log_softmax(teacher_logits_a / tau, axis=-1)
    log_p_b = jax.nn.log_softmax(teacher_logits_b / tau, axis=-1)
    # log(0.5 * (p_a + p_b)) via a stable two-term log-sum-exp mix.
    log_half = jnp.asarray(jnp.log(0.5), dtype=student_logits.dtype)
    log_p_target = log_half + jnp.logaddexp(log_p_a, log_p_b)
    log_p_student = jax.nn.log_softmax(student_logits / tau, axis=-1)
    p_target = jnp.exp(log_p_target)
    kl = jnp.sum(p_target * (log_p_target - log_p_student), axis=-1)
    return jnp.mean(kl) * (tau ** 2)


def distillation_loss(student, teacher, tokens_a, tokens_b, tau=TAU):
    """Eq 5 self-distillation. Teacher outputs are stop-gradient.

    The student embeds A and B with its own table, averages them, and runs
    one forward on that mix. The teacher is evaluated on A and on B alone.
    Gradients do not flow into teacher parameters.
    """
    student = _as_module(student)
    teacher = _as_module(teacher)
    teacher_logits_a = jax.lax.stop_gradient(teacher(tokens_a))
    teacher_logits_b = jax.lax.stop_gradient(teacher(tokens_b))
    mixed = mix_embeddings(student, tokens_a, tokens_b)
    student_logits = student.forward_from_embeddings(mixed)
    return kl_distill_from_logits(
        student_logits, teacher_logits_a, teacher_logits_b, tau=tau
    )


def make_toy_batch(batch=16, length=8):
    """Two streams of repeating bigrams, each row starting with token 0.

    Stream A alternates 1, 2. Stream B alternates 3, 4. Even and odd rows
    use opposite phases so the batch is not a single repeated string.
    """
    if length < 2:
        raise ValueError("length must be at least 2")
    rows_a = []
    rows_b = []
    for row in range(batch):
        phase = row % 2
        seq_a = [0]
        seq_b = [0]
        for step in range(length - 1):
            if (step + phase) % 2 == 0:
                seq_a.append(1)
                seq_b.append(3)
            else:
                seq_a.append(2)
                seq_b.append(4)
        rows_a.append(seq_a)
        rows_b.append(seq_b)
    tokens_a = jnp.asarray(rows_a, dtype=jnp.int32)
    tokens_b = jnp.asarray(rows_b, dtype=jnp.int32)
    return tokens_a, tokens_b
