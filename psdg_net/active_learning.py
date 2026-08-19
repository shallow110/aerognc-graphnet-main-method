from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float64)
    low = np.nanmin(values[finite])
    high = np.nanmax(values[finite])
    if high - low < 1.0e-12:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


@dataclass
class AcquisitionInputs:
    predictive_std: np.ndarray
    feasibility_probability: np.ndarray
    predicted_objective: np.ndarray
    best_observed: float
    embeddings: np.ndarray
    constraint_margin: Optional[np.ndarray] = None
    physics_residual: Optional[np.ndarray] = None
    verification_cost: Optional[np.ndarray] = None


class PhysicsAwareAcquisition:
    """Domain-independent, cost-aware active-learning interface."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        default = {
            "uncertainty": 0.25,
            "boundary": 0.20,
            "improvement": 0.20,
            "diversity": 0.15,
            "physics_residual": 0.10,
            "verification_cost": 0.10,
        }
        if weights:
            default.update(weights)
        total = sum(default.values())
        self.weights = {key: value / total for key, value in default.items()}

    @staticmethod
    def _diversity(embeddings: np.ndarray, selected_embeddings: Optional[np.ndarray]) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if selected_embeddings is None or len(selected_embeddings) == 0:
            center = np.mean(embeddings, axis=0, keepdims=True)
            return np.linalg.norm(embeddings - center, axis=1)
        selected_embeddings = np.asarray(selected_embeddings, dtype=np.float64)
        distance = np.linalg.norm(embeddings[:, None, :] - selected_embeddings[None, :, :], axis=-1)
        return np.min(distance, axis=1)

    def score(
        self,
        inputs: AcquisitionInputs,
        selected_embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        uncertainty = _normalize(np.asarray(inputs.predictive_std).reshape(-1))
        probability = np.asarray(inputs.feasibility_probability).reshape(-1)
        boundary = _normalize(np.exp(-5.0 * np.abs(probability - 0.5)))
        if inputs.constraint_margin is not None:
            margin = np.min(np.asarray(inputs.constraint_margin), axis=-1)
            boundary = _normalize(boundary + np.exp(-np.abs(margin)))
        improvement = _normalize(np.maximum(0.0, float(inputs.best_observed) - np.asarray(inputs.predicted_objective).reshape(-1)))
        diversity = _normalize(self._diversity(inputs.embeddings, selected_embeddings))
        residual = np.zeros_like(uncertainty)
        if inputs.physics_residual is not None:
            residual = _normalize(np.max(np.abs(np.asarray(inputs.physics_residual)), axis=-1))
        cost_utility = np.ones_like(uncertainty)
        if inputs.verification_cost is not None:
            cost_utility = 1.0 - _normalize(np.asarray(inputs.verification_cost).reshape(-1))
        components = {
            "uncertainty": uncertainty,
            "boundary": boundary,
            "improvement": improvement,
            "diversity": diversity,
            "physics_residual": residual,
            "verification_cost": cost_utility,
        }
        score = sum(self.weights[name] * values for name, values in components.items())
        return np.asarray(score, dtype=np.float64), components

    def select(
        self,
        inputs: AcquisitionInputs,
        batch_size: int,
        selected_embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        score, components = self.score(inputs, selected_embeddings=selected_embeddings)
        order = np.argsort(score)[::-1][: int(batch_size)]
        return order.astype(np.int64), components
