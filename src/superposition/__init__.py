"""Superposition linearity hypothesis and joint contrastive decoding.

Implements embedding averaging, KL self-distillation, the geometric-mean
obstruction, and joint contrastive decoding from arXiv 2609.29845.
"""

from superposition.decode import (
    generate,
    geometric_mean_distribution,
    joint_contrastive_forward,
    joint_contrastive_logits,
    joint_contrastive_loss,
)
from superposition.mix import (
    TAU,
    distillation_loss,
    kl_distill_from_logits,
    make_toy_batch,
    mix_embeddings,
)
from superposition.model import (
    Transformer,
    copy_module,
    embed,
    forward,
    forward_from_embeddings,
    init_transformer,
    lm_cross_entropy,
    next_token_cross_entropy,
)

__all__ = [
    "TAU",
    "Transformer",
    "copy_module",
    "distillation_loss",
    "embed",
    "forward",
    "forward_from_embeddings",
    "generate",
    "geometric_mean_distribution",
    "init_transformer",
    "joint_contrastive_forward",
    "joint_contrastive_logits",
    "joint_contrastive_loss",
    "kl_distill_from_logits",
    "lm_cross_entropy",
    "make_toy_batch",
    "mix_embeddings",
    "next_token_cross_entropy",
]
