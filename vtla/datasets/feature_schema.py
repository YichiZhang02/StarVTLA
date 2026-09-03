"""Training-contract comparison for dataset mixture features."""

from __future__ import annotations

from typing import Any


MIXTURE_FEATURE_SCHEMA_FIELDS = (
    "dtype",
    "shape",
    "names",
    "tactile_encoding",
    "storage_dtype",
)


def mixture_feature_schema_diff(
    reference: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]
) -> list[str]:
    """Compare fields that determine the tensors and semantics seen by a policy."""
    differences = []
    for key in sorted(set(reference) | set(candidate)):
        if key not in reference:
            differences.append(f"extra feature {key!r}")
            continue
        if key not in candidate:
            differences.append(f"missing feature {key!r}")
            continue
        for field in MIXTURE_FEATURE_SCHEMA_FIELDS:
            expected = reference[key].get(field)
            actual = candidate[key].get(field)
            if expected != actual:
                differences.append(
                    f"feature {key!r} field {field!r}: expected {expected!r}, got {actual!r}"
                )
    return differences
