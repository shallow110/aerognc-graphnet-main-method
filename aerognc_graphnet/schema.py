from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    node: str
    low: float
    high: float
    role: str


PARAMETER_SPECS: List[ParameterSpec] = [
    ParameterSpec("speed_scale", "InitialNode", 0.92, 1.08, "condition"),
    ParameterSpec("pitch_delta_deg", "InitialNode", -3.0, 3.0, "condition"),
    ParameterSpec("length_diameter_ratio", "GeometryNode", 4.0, 8.0, "design"),
    ParameterSpec("nose_haack_c", "GeometryNode", 0.0, 1.0, "design"),
    ParameterSpec("fin_position", "GeometryNode", 0.55, 1.00, "design"),
    ParameterSpec("fin_span", "GeometryNode", 0.05, 0.18, "design"),
    ParameterSpec("fin_area_scale", "GeometryNode", 0.50, 1.50, "design"),
    ParameterSpec("pos_sigma", "NavBeliefNode", 0.0, 15.0, "condition"),
    ParameterSpec("vel_sigma", "NavBeliefNode", 0.0, 3.0, "condition"),
    ParameterSpec("att_sigma_deg", "NavBeliefNode", 0.0, 0.50, "condition"),
    ParameterSpec("ctrl_gain", "ControlNode", 0.8, 4.0, "design"),
    ParameterSpec("delta_gain", "ActuatorNode", 0.01, 0.12, "design"),
    ParameterSpec("max_deflection_deg", "ActuatorNode", 5.0, 25.0, "design"),
    ParameterSpec("track_weight", "ControlNode", 0.30, 3.00, "design"),
    ParameterSpec("terminal_weight", "ControlNode", 0.30, 3.00, "design"),
    ParameterSpec("control_weight", "ControlNode", 0.30, 3.00, "design"),
    ParameterSpec("q_max", "SafetyNode", 5.0e4, 8.0e5, "constraint"),
    ParameterSpec("alpha_max_deg", "SafetyNode", 5.0, 30.0, "constraint"),
    ParameterSpec("p_trace_max", "SafetyNode", 10.0, 3000.0, "constraint"),
]


DEFAULTS: Dict[str, float] = {
    "speed_scale": 1.0,
    "pitch_delta_deg": 0.0,
    "length_diameter_ratio": 6.0,
    "nose_haack_c": 0.0,
    "fin_position": 0.6,
    "fin_span": 0.1,
    "fin_area_scale": 1.0,
    "pos_sigma": 0.0,
    "vel_sigma": 0.0,
    "att_sigma_deg": 0.0,
    "ctrl_gain": 2.4,
    "delta_gain": 0.05,
    "max_deflection_deg": 15.0,
    "track_weight": 1.0,
    "terminal_weight": 1.0,
    "control_weight": 1.0,
    "q_max": 5.0e5,
    "alpha_max_deg": 15.0,
    "p_trace_max": 2000.0,
}


LABEL_NAMES: List[str] = [
    "terminal_error",
    "mean_same_time_error",
    "control_energy",
    "barrier_cost",
    "nav_uncertainty_cost",
    "total_cost",
    "feasible",
]


def vector_to_parameter_dict(vector, specs: Iterable[ParameterSpec] = PARAMETER_SPECS) -> Dict[str, float]:
    return {spec.name: float(value) for spec, value in zip(specs, vector)}


def default_vector() -> np.ndarray:
    return np.asarray([DEFAULTS[spec.name] for spec in PARAMETER_SPECS], dtype=np.float32)


def bounds() -> Tuple[np.ndarray, np.ndarray]:
    lows = np.asarray([spec.low for spec in PARAMETER_SPECS], dtype=np.float32)
    highs = np.asarray([spec.high for spec in PARAMETER_SPECS], dtype=np.float32)
    return lows, highs


def parameter_index(name: str) -> int:
    for index, spec in enumerate(PARAMETER_SPECS):
        if spec.name == name:
            return index
    raise KeyError(name)


def index_groups() -> Tuple[np.ndarray, np.ndarray]:
    condition = []
    design = []
    for index, spec in enumerate(PARAMETER_SPECS):
        if spec.role in {"condition", "constraint"}:
            condition.append(index)
        elif spec.role == "design":
            design.append(index)
    return np.asarray(condition, dtype=np.int64), np.asarray(design, dtype=np.int64)


def condition_full_from_matrix(x: np.ndarray, condition_indices: np.ndarray) -> np.ndarray:
    condition_full = np.repeat(default_vector().reshape(1, -1), int(x.shape[0]), axis=0).astype(np.float32)
    condition_full[:, condition_indices] = x[:, condition_indices]
    return condition_full
