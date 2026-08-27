"""Cache construction and mmap-only datasets for tactile reconstruction."""

from .cache_schema import CACHE_VERSION, TactileCache, validate_cache
from .contact import CONTACT_METHOD, compute_contact_mask, neutral_residual_topk_score
from .npy_tactile_dataset import (
    TactileMixtureDataset,
    TactileNpyDataset,
    WeightedMixtureSampler,
    resolve_tactile_dataset,
)

__all__ = [
    "CACHE_VERSION",
    "CONTACT_METHOD",
    "TactileCache",
    "TactileMixtureDataset",
    "TactileNpyDataset",
    "WeightedMixtureSampler",
    "compute_contact_mask",
    "neutral_residual_topk_score",
    "resolve_tactile_dataset",
    "validate_cache",
]
