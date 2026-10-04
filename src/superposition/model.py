"""Tiny decoder-only transformer.

Causal self-attention, pre-norm residual blocks, GELU MLP, tied output
projection (logits = h @ E.T). No token-level or hidden-state mixing lives
here; mixing is only the input embedding average in mix.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random


def _as_key(key):
    if isinstance(key, int):
        return random.PRNGKey(key)
    return key


def layer_norm(x, scale, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return scale * (x - mean) * jnp.reciprocal(jnp.sqrt(var + eps)) + bias


def _init_layer(key, d_model, d_ff, init_scale):
    k_q, k_k, k_v, k_o, k_up, k_down = random.split(key, 6)
    return {
        "ln1_scale": jnp.ones((d_model,)),
        "ln1_bias": jnp.zeros((d_model,)),
        "w_q": random.normal(k_q, (d_model, d_model)) * init_scale,
        "w_k": random.normal(k_k, (d_model, d_model)) * init_scale,
        "w_v": random.normal(k_v, (d_model, d_model)) * init_scale,
        "w_o": random.normal(k_o, (d_model, d_model)) * init_scale,
        "b_q": jnp.zeros((d_model,)),
        "b_k": jnp.zeros((d_model,)),
        "b_v": jnp.zeros((d_model,)),
        "b_o": jnp.zeros((d_model,)),
        "ln2_scale": jnp.ones((d_model,)),
        "ln2_bias": jnp.zeros((d_model,)),
        "w_up": random.normal(k_up, (d_model, d_ff)) * init_scale,
        "w_down": random.normal(k_down, (d_ff, d_model)) * init_scale,
        "b_up": jnp.zeros((d_ff,)),
        "b_down": jnp.zeros((d_model,)),
    }


def init_params(
    key,
    vocab=32,
    d_model=32,
    n_layers=1,
    n_heads=4,
    d_ff=64,
    max_len=32,
    init_scale=0.02,
):
    """Initialize a parameter pytree. n_heads is not stored in the tree."""
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    key = _as_key(key)
    key, k_embed, k_pos = random.split(key, 3)
    layer_keys = random.split(key, n_layers)
    layers = tuple(
        _init_layer(layer_keys[i], d_model, d_ff, init_scale) for i in range(n_layers)
    )
    return {
        "embed": random.normal(k_embed, (vocab, d_model)) * init_scale,
        "pos": random.normal(k_pos, (max_len, d_model)) * init_scale,
        "layers": layers,
    }


def embed(params, tokens):
    """Token embedding plus the same absolute positional embedding.

    tokens: int array [batch, length]. Positions are added before any mix so
    that z = 0.5 * (E(A) + E(B)) includes positions.
    """
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, length]")
    length = tokens.shape[1]
    max_len = params["pos"].shape[0]
    if length > max_len:
        raise ValueError(f"sequence length {length} exceeds max_len {max_len}")
    tok = params["embed"][tokens]
    pos = params["pos"][:length]
    return tok + pos[None, :, :]


def _attention(x, layer, n_heads):
    batch, length, d_model = x.shape
    d_head = d_model // n_heads
    query = (x @ layer["w_q"] + layer["b_q"]).reshape(batch, length, n_heads, d_head)
    key = (x @ layer["w_k"] + layer["b_k"]).reshape(batch, length, n_heads, d_head)
    value = (x @ layer["w_v"] + layer["b_v"]).reshape(batch, length, n_heads, d_head)
    query = jnp.transpose(query, (0, 2, 1, 3))
    key = jnp.transpose(key, (0, 2, 1, 3))
    value = jnp.transpose(value, (0, 2, 1, 3))
    scale = jnp.reciprocal(jnp.sqrt(jnp.asarray(d_head, dtype=x.dtype)))
    scores = jnp.einsum("bhld,bhmd->bhlm", query, key) * scale
    causal = jnp.tril(jnp.ones((length, length), dtype=bool))
    neg = jnp.asarray(-1.0e9, dtype=scores.dtype)
    scores = jnp.where(causal[None, None, :, :], scores, neg)
    weights = jax.nn.softmax(scores, axis=-1)
    mixed = jnp.einsum("bhlm,bhmd->bhld", weights, value)
    mixed = jnp.transpose(mixed, (0, 2, 1, 3)).reshape(batch, length, d_model)
    return mixed @ layer["w_o"] + layer["b_o"]


def _mlp(x, layer):
    hidden = jax.nn.gelu(x @ layer["w_up"] + layer["b_up"])
    return hidden @ layer["w_down"] + layer["b_down"]


def _block(h, layer, n_heads):
    normed = layer_norm(h, layer["ln1_scale"], layer["ln1_bias"])
    h = h + _attention(normed, layer, n_heads)
    normed = layer_norm(h, layer["ln2_scale"], layer["ln2_bias"])
    h = h + _mlp(normed, layer)
    return h


def forward_from_embeddings(params, hidden, n_heads):
    """Run the trunk on already-embedded inputs and tie the output to E.

    hidden: [batch, length, d_model]. logits = h @ E.T, no extra output bias.
    """
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, length, d_model]")
    h = hidden
    for layer in params["layers"]:
        h = _block(h, layer, n_heads)
    return h @ jnp.transpose(params["embed"])


def forward(params, tokens, n_heads):
    return forward_from_embeddings(params, embed(params, tokens), n_heads)


def next_token_cross_entropy(logits, tokens):
    """Mean cross-entropy of logits[:, t] against tokens[:, t + 1]."""
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    log_probs = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
    targets = tokens[:, 1:]
    gathered = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
    return -jnp.mean(gathered)


@jax.tree_util.register_pytree_node_class
class Transformer:
    """Decoder-only module. Parameter leaves are differentiable; n_heads is static."""

    def __init__(self, params, n_heads):
        self.params = params
        self.n_heads = int(n_heads)

    def tree_flatten(self):
        return (self.params,), (self.n_heads,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (params,) = children
        (n_heads,) = aux
        return cls(params, n_heads)

    def embed(self, tokens):
        return embed(self.params, tokens)

    def forward_from_embeddings(self, hidden):
        return forward_from_embeddings(self.params, hidden, self.n_heads)

    def __call__(self, tokens):
        return forward(self.params, tokens, self.n_heads)


def init_transformer(
    key,
    vocab=32,
    d_model=32,
    n_layers=1,
    n_heads=4,
    d_ff=64,
    max_len=32,
    init_scale=0.02,
):
    params = init_params(
        key,
        vocab=vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max_len,
        init_scale=init_scale,
    )
    return Transformer(params, n_heads)


def copy_module(module):
    """Value copy of a module (frozen teacher / small-model snapshot)."""
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), module.params)
    return Transformer(params, module.n_heads)


def lm_cross_entropy(module, tokens):
    return next_token_cross_entropy(module(tokens), tokens)
