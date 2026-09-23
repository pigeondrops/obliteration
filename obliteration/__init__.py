"""Oblique abliteration: remove the refusal direction while preserving a chosen
second direction."""
from .ablator import COMPONENTS, ObliqueAblator, layer_weight
from .data import grounding_pairs
from .directions import (build_directions, find_decoder_layers, residual_means,
                         separability)
from .export import apply_to_weights, restore_missing_tensors
from .metrics import count_hedges, count_refusals, first_token_logprobs, kl_from_baseline
from .projector import (interpolate_direction, leverage, oblique_covector,
                        preserve_subspace)

__version__ = "0.1.0"
__all__ = [
    "ObliqueAblator", "COMPONENTS", "layer_weight",
    "oblique_covector", "preserve_subspace", "interpolate_direction", "leverage",
    "residual_means", "build_directions", "separability", "find_decoder_layers",
    "first_token_logprobs", "kl_from_baseline", "count_refusals", "count_hedges",
    "apply_to_weights", "restore_missing_tensors", "grounding_pairs",
]
