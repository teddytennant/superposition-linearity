"""Checks for embedding mix, KL distillation, and joint contrastive decoding."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

from superposition.decode import (
    generate,
    geometric_mean_distribution,
    joint_contrastive_forward,
    joint_contrastive_logits,
    joint_contrastive_loss,
)
from superposition.mix import (
    distillation_loss,
    kl_distill_from_logits,
    make_toy_batch,
    mix_embeddings,
)
from superposition.model import Transformer, init_transformer


def _float_leaves(tree):
    leaves = []
    for leaf in jax.tree_util.tree_leaves(tree):
        leaf = jnp.asarray(leaf)
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            leaves.append(leaf)
    return leaves


def test_mix_matches_embedding_average_and_logits_do_not():
    model = init_transformer(random.PRNGKey(0), init_scale=0.1)
    tokens_a = random.randint(random.PRNGKey(1), (4, 8), 0, 32)
    tokens_b = random.randint(random.PRNGKey(2), (4, 8), 0, 32)
    tokens_b = tokens_b.at[:, 0].set((tokens_a[:, 0] + 1) % 32)
    assert not bool(jnp.array_equal(tokens_a, tokens_b))

    mixed = mix_embeddings(model, tokens_a, tokens_b)
    embedded_a = model.embed(tokens_a)
    embedded_b = model.embed(tokens_b)
    np.testing.assert_allclose(mixed, 0.5 * (embedded_a + embedded_b), atol=1e-6)

    length = tokens_a.shape[1]
    pos = model.params["pos"][:length]
    manual = 0.5 * (
        model.params["embed"][tokens_a]
        + pos[None, :, :]
        + model.params["embed"][tokens_b]
        + pos[None, :, :]
    )
    np.testing.assert_allclose(mixed, manual, atol=1e-6)
    # Positions are actually in E: a nonzero pos table changes the embedding.
    assert float(jnp.max(jnp.abs(model.params["pos"]))) > 0.0
    raw = model.params["embed"][tokens_a]
    assert float(jnp.max(jnp.abs(embedded_a - raw))) > 1e-6

    logits_mixed = model.forward_from_embeddings(mixed)
    logits_avg = 0.5 * (model(tokens_a) + model(tokens_b))
    # Superposition is not free at the logits on a random model.
    diff = float(jnp.max(jnp.abs(logits_mixed - logits_avg)))
    assert diff > 1e-3, diff


def test_kl_zero_when_student_matches_target_and_positive_otherwise():
    tau = 2.0
    rng = np.random.default_rng(0)
    teacher_a = rng.normal(size=(2, 4, 8)).astype(np.float32)
    teacher_b = rng.normal(size=(2, 4, 8)).astype(np.float32)
    log_p_a = _log_softmax(teacher_a / tau)
    log_p_b = _log_softmax(teacher_b / tau)
    log_p_target = np.log(0.5) + np.logaddexp(log_p_a, log_p_b)
    p_target = np.exp(log_p_target)
    student = (tau * log_p_target).astype(np.float32)
    loss = kl_distill_from_logits(
        jnp.asarray(student),
        jnp.asarray(teacher_a),
        jnp.asarray(teacher_b),
        tau=tau,
    )
    assert float(loss) < 1e-4

    student_bad = np.zeros_like(teacher_a)
    student_bad[..., 0] = 30.0
    loss_bad = kl_distill_from_logits(
        jnp.asarray(student_bad),
        jnp.asarray(teacher_a),
        jnp.asarray(teacher_b),
        tau=tau,
    )
    assert float(loss_bad) > 0.0

    # Hand KL, including the tau^2 distillation factor.
    log_p_student = _log_softmax(student_bad / tau)
    kl = np.sum(p_target * (log_p_target - log_p_student), axis=-1).mean() * (tau ** 2)
    np.testing.assert_allclose(float(loss_bad), float(kl), rtol=1e-5, atol=1e-5)


def test_teacher_is_stop_gradient():
    student = init_transformer(random.PRNGKey(2))
    teacher = init_transformer(random.PRNGKey(3))
    other = init_transformer(random.PRNGKey(4))
    tokens_a = jnp.asarray([[1, 2, 1, 2, 1, 2]], dtype=jnp.int32)
    tokens_b = jnp.asarray([[3, 4, 3, 4, 3, 4]], dtype=jnp.int32)

    def loss_fn(s, t):
        return distillation_loss(s, t, tokens_a, tokens_b, tau=2.0)

    grad_student = jax.grad(loss_fn, argnums=0)(student, teacher)
    grad_teacher = jax.grad(loss_fn, argnums=1)(student, teacher)
    student_leaves = _float_leaves(grad_student)
    teacher_leaves = _float_leaves(grad_teacher)
    assert teacher_leaves
    assert student_leaves
    assert all(bool(jnp.all(leaf == 0)) for leaf in teacher_leaves)
    assert any(bool(jnp.any(leaf != 0)) for leaf in student_leaves)

    loss_teacher = loss_fn(student, teacher)
    loss_other = loss_fn(student, other)
    assert not bool(jnp.allclose(loss_teacher, loss_other))


def test_geometric_mean_matches_hand_computation():
    p_a = np.array([0.1, 0.2, 0.7], dtype=np.float64)
    p_b = np.array([0.5, 0.4, 0.1], dtype=np.float64)
    assert not np.allclose(p_a, p_b)
    hand = []
    for index in range(p_a.shape[0]):
        hand.append(np.sqrt(p_a[index] * p_b[index]))
    hand = np.asarray(hand, dtype=np.float64)
    hand = hand / hand.sum()
    got = geometric_mean_distribution(jnp.asarray(p_a), jnp.asarray(p_b))
    np.testing.assert_allclose(np.asarray(got), hand, rtol=1e-6, atol=1e-6)
    arithmetic = 0.5 * (p_a + p_b)
    assert not np.allclose(np.asarray(got), arithmetic)

    batched_a = np.stack([p_a, p_b])
    batched_b = np.stack([p_b, p_a])
    batched = geometric_mean_distribution(jnp.asarray(batched_a), jnp.asarray(batched_b))
    assert batched.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(batched[0]), hand, rtol=1e-6, atol=1e-6)


def test_joint_contrastive_logits_match_formula():
    ell_z = jnp.asarray([1.0, 2.0, 3.0, -1.0])
    ell_a = jnp.asarray([0.5, 0.0, -1.0, 0.25])
    ell_b = jnp.asarray([1.0, -0.5, 0.25, 2.0])
    for alpha, beta in ((1.0, 1.0), (0.5, 2.0)):
        got_a, got_b = joint_contrastive_logits(ell_z, ell_a, ell_b, alpha, beta)
        np.testing.assert_allclose(got_a, ell_z + alpha * ell_a - beta * ell_b)
        np.testing.assert_allclose(got_b, ell_z + alpha * ell_b - beta * ell_a)

    key = random.PRNGKey(5)
    ell_z = random.normal(key, (2, 3, 4))
    ell_a = random.normal(random.PRNGKey(6), (2, 3, 4))
    ell_b = random.normal(random.PRNGKey(7), (2, 3, 4))
    alpha = jnp.asarray(0.5)
    beta = jnp.asarray(2.0)
    got_a, got_b = joint_contrastive_logits(ell_z, ell_a, ell_b, alpha, beta)
    np.testing.assert_allclose(got_a, ell_z + alpha * ell_a - beta * ell_b, atol=1e-6)
    np.testing.assert_allclose(got_b, ell_z + alpha * ell_b - beta * ell_a, atol=1e-6)

    # Two different modules are accepted. Logit shapes still match on vocab.
    large = init_transformer(random.PRNGKey(8), n_layers=1, d_model=32, n_heads=4)
    small = init_transformer(
        random.PRNGKey(9), n_layers=2, d_model=16, n_heads=4, d_ff=32
    )
    tokens_a = jnp.asarray([[1, 2, 1, 2]], dtype=jnp.int32)
    tokens_b = jnp.asarray([[3, 4, 3, 4]], dtype=jnp.int32)
    out_a, out_b = joint_contrastive_forward(large, small, tokens_a, tokens_b, 1.0, 1.0)
    assert out_a.shape == out_b.shape == (1, 4, 32)
    assert jnp.all(jnp.isfinite(out_a))
    assert jnp.all(jnp.isfinite(out_b))


def test_generate_appends_different_argmax_tokens():
    large = _identity_copy_model(random.PRNGKey(0))
    small = _identity_copy_model(random.PRNGKey(1))
    prefix_a = jnp.asarray([0, 1, 2, 1], dtype=jnp.int32)
    prefix_b = jnp.asarray([0, 3, 4, 3], dtype=jnp.int32)
    alpha = jnp.asarray(1.0)
    beta = jnp.asarray(1.0)
    ell_a, ell_b = joint_contrastive_forward(
        large, small, prefix_a[None, :], prefix_b[None, :], alpha, beta
    )
    token_a = int(jnp.argmax(ell_a[0, -1]))
    token_b = int(jnp.argmax(ell_b[0, -1]))
    assert token_a != token_b

    new_a, new_b = generate(large, small, prefix_a, prefix_b, alpha, beta)
    assert new_a.shape == (5,)
    assert new_b.shape == (5,)
    np.testing.assert_array_equal(np.asarray(new_a[:-1]), np.asarray(prefix_a))
    np.testing.assert_array_equal(np.asarray(new_b[:-1]), np.asarray(prefix_b))
    assert int(new_a[-1]) == token_a
    assert int(new_b[-1]) == token_b
    assert int(new_a[-1]) != int(new_b[-1])


def test_alpha_beta_receive_nonzero_gradients():
    model = init_transformer(random.PRNGKey(0), init_scale=0.1)
    tokens_a, tokens_b = make_toy_batch(batch=4, length=6)
    alpha = jnp.asarray(1.0)
    beta = jnp.asarray(1.0)

    def loss_fn(a, b):
        return joint_contrastive_loss(model, model, tokens_a, tokens_b, a, b)

    grad_alpha, grad_beta = jax.grad(loss_fn, argnums=(0, 1))(alpha, beta)
    assert float(jnp.abs(grad_alpha)) > 1e-6
    assert float(jnp.abs(grad_beta)) > 1e-6


def test_phase2_smoke_does_not_nan():
    model = init_transformer(random.PRNGKey(0))
    teacher = Transformer(
        jax.tree_util.tree_map(lambda x: jnp.array(x), model.params),
        model.n_heads,
    )
    tokens_a, tokens_b = make_toy_batch(batch=16, length=8)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(model)

    def loss_fn(student):
        return distillation_loss(student, teacher, tokens_a, tokens_b, tau=2.0)

    for _ in range(5):
        loss, grads = jax.value_and_grad(loss_fn)(model)
        assert bool(jnp.isfinite(loss))
        for leaf in _float_leaves(grads):
            assert bool(jnp.all(jnp.isfinite(leaf)))
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)
        for leaf in _float_leaves(model):
            assert bool(jnp.all(jnp.isfinite(leaf)))
    final = loss_fn(model)
    assert bool(jnp.isfinite(final))


def _identity_copy_model(key):
    """Residual branches zeroed and E = I, so logits copy the current token."""
    model = init_transformer(key, vocab=32, d_model=32, n_layers=1, n_heads=4, d_ff=64)
    params = dict(model.params)
    params["embed"] = jnp.eye(32, 32)
    params["pos"] = jnp.zeros_like(model.params["pos"])
    layers = []
    for layer in model.params["layers"]:
        layer = dict(layer)
        layer["w_o"] = jnp.zeros_like(layer["w_o"])
        layer["b_o"] = jnp.zeros_like(layer["b_o"])
        layer["w_down"] = jnp.zeros_like(layer["w_down"])
        layer["b_down"] = jnp.zeros_like(layer["b_down"])
        layers.append(layer)
    params["layers"] = tuple(layers)
    return Transformer(params, model.n_heads)


def _log_softmax(values, axis=-1):
    peak = np.max(values, axis=axis, keepdims=True)
    return values - peak - np.log(np.sum(np.exp(values - peak), axis=axis, keepdims=True))
