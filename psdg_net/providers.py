from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from .schema import GraphSchema


@dataclass
class PhysicsBatch:
    input_std: np.ndarray
    node_extras: np.ndarray
    equation_residuals: np.ndarray
    constraint_margins: np.ndarray
    relation_strength: np.ndarray
    fidelity: np.ndarray

    def torch(self, device: torch.device) -> dict:
        result = {
            "input_std": torch.as_tensor(self.input_std, dtype=torch.float32, device=device),
            "node_extras": torch.as_tensor(self.node_extras, dtype=torch.float32, device=device),
            "equation_residuals": torch.as_tensor(self.equation_residuals, dtype=torch.float32, device=device),
            "constraint_margins": torch.as_tensor(self.constraint_margins, dtype=torch.float32, device=device),
            "relation_strength": torch.as_tensor(self.relation_strength, dtype=torch.float32, device=device),
            "fidelity": torch.as_tensor(self.fidelity, dtype=torch.float32, device=device),
        }
        return result

    def subset(self, indices: np.ndarray) -> "PhysicsBatch":
        return PhysicsBatch(
            input_std=self.input_std[indices],
            node_extras=self.node_extras[indices],
            equation_residuals=self.equation_residuals[indices],
            constraint_margins=self.constraint_margins[indices],
            relation_strength=self.relation_strength[indices],
            fidelity=self.fidelity[indices],
        )


class PhysicsFeatureProvider(Protocol):
    schema: GraphSchema

    def evaluate(self, x: np.ndarray) -> PhysicsBatch:
        ...


def default_relation_strength(schema: GraphSchema, n: int) -> np.ndarray:
    prior = np.asarray([relation.prior_confidence for relation in schema.relations], dtype=np.float32)
    return np.repeat(prior.reshape(1, -1), int(n), axis=0)


def default_fidelity(schema: GraphSchema, n: int, value: float = 1.0) -> np.ndarray:
    return np.full((int(n), len(schema.relations)), float(value), dtype=np.float32)
