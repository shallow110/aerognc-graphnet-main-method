from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def acquisition_score(
    epistemic_uncertainty,
    constraint_boundary,
    predicted_improvement,
    diversity_distance,
    high_gradient_region,
) -> np.ndarray:
    """AeroGNC-GraphNet acquisition rule from the paper."""
    return (
        0.30 * np.asarray(epistemic_uncertainty)
        + 0.25 * np.asarray(constraint_boundary)
        + 0.20 * np.asarray(predicted_improvement)
        + 0.15 * np.asarray(diversity_distance)
        + 0.10 * np.asarray(high_gradient_region)
    )


def select_top_k(candidate_metrics: Dict[str, np.ndarray], k: int) -> Tuple[np.ndarray, np.ndarray]:
    scores = acquisition_score(
        normalize01(candidate_metrics["epistemic_uncertainty"]),
        normalize01(candidate_metrics["constraint_boundary"]),
        normalize01(candidate_metrics["predicted_improvement"]),
        normalize01(candidate_metrics["diversity_distance"]),
        normalize01(candidate_metrics["high_gradient_region"]),
    )
    order = np.argsort(scores)[::-1]
    return order[: int(k)], scores
