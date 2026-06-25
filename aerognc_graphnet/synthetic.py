from __future__ import annotations

from typing import Tuple

import numpy as np

from .schema import LABEL_NAMES, PARAMETER_SPECS, bounds


def sample_parameter_matrix(n: int, seed: int = 0) -> np.ndarray:
    lows, highs = bounds()
    rng = np.random.default_rng(seed)
    return rng.uniform(lows, highs, size=(int(n), len(PARAMETER_SPECS))).astype(np.float32)


def synthetic_aero_node_features(x: np.ndarray) -> np.ndarray:
    """Self-contained toy six-channel AeroMapNode fixture for examples and tests.

    These are dimensionless synthetic values, not aerodynamic tables or
    vehicle-specific coefficients. In a real reproduction, replace them with
    public log-scaled drag, lift, moment, control-force, control-moment and
    roll-damping channels from the chosen solver or dataset.
    """
    idx = {spec.name: i for i, spec in enumerate(PARAMETER_SPECS)}
    lows, highs = bounds()
    x01 = np.clip((x - lows.reshape(1, -1)) / np.maximum(highs - lows, 1.0e-6).reshape(1, -1), 0.0, 1.0)
    geometry = x01[:, [idx["length_diameter_ratio"], idx["nose_haack_c"], idx["fin_position"], idx["fin_span"], idx["fin_area_scale"]]]
    channels = np.stack(
        [
            0.60 * geometry[:, 0] - 0.30 * geometry[:, 1],
            0.45 * geometry[:, 3] + 0.20 * geometry[:, 4],
            np.sin(np.pi * geometry[:, 2]) * 0.35,
            0.55 * geometry[:, 3] * geometry[:, 4],
            0.40 * geometry[:, 2] * geometry[:, 4],
            0.25 + 0.30 * geometry[:, 3],
        ],
        axis=1,
    )
    return np.clip(channels * 2.0 - 1.0, -3.0, 3.0).astype(np.float32)


def synthetic_labels(x: np.ndarray, aero_features: np.ndarray = None) -> np.ndarray:
    """Small deterministic label function so the package runs without data files."""
    if aero_features is None:
        aero_features = synthetic_aero_node_features(x)
    idx = {spec.name: i for i, spec in enumerate(PARAMETER_SPECS)}

    fin_authority = x[:, idx["fin_span"]] * x[:, idx["fin_area_scale"]] * x[:, idx["delta_gain"]] * x[:, idx["max_deflection_deg"]]
    control_authority = x[:, idx["ctrl_gain"]] * fin_authority
    nav_trace = x[:, idx["pos_sigma"]] ** 2 + x[:, idx["vel_sigma"]] ** 2 + x[:, idx["att_sigma_deg"]] ** 2
    safety_slack = (
        x[:, idx["q_max"]] / 8.0e5
        + x[:, idx["alpha_max_deg"]] / 30.0
        + x[:, idx["p_trace_max"]] / 3000.0
    ) / 3.0
    speed_pitch = np.abs(x[:, idx["speed_scale"]] - 1.0) + np.abs(x[:, idx["pitch_delta_deg"]]) / 3.0
    aero_penalty = np.maximum(aero_features[:, 0] - aero_features[:, 3], 0.0)

    terminal_error = 15.0 + 3.0 * speed_pitch + 4.0 / np.maximum(control_authority, 1.0e-3) + 0.4 * aero_penalty
    same_time_error = 8.0 + 0.12 * nav_trace + 0.8 * speed_pitch
    control_energy = 2.0 + 0.04 * x[:, idx["max_deflection_deg"]] ** 2 + 0.6 * x[:, idx["ctrl_gain"]]
    barrier_cost = np.maximum(0.65 - safety_slack, 0.0) * 30.0
    nav_cost = 0.08 * nav_trace
    total_cost = (
        x[:, idx["terminal_weight"]] * terminal_error
        + x[:, idx["track_weight"]] * same_time_error
        + x[:, idx["control_weight"]] * control_energy
        + barrier_cost
        + nav_cost
    )
    feasible = ((terminal_error < 42.0) & (barrier_cost < 8.0) & (nav_cost < 20.0)).astype(np.float32)
    y = np.stack([terminal_error, same_time_error, control_energy, barrier_cost, nav_cost, total_cost, feasible], axis=1)
    return y.astype(np.float32)


def synthetic_dataset(n: int = 128, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = sample_parameter_matrix(n, seed=seed)
    aero = synthetic_aero_node_features(x)
    y = synthetic_labels(x, aero)
    if y.shape[1] != len(LABEL_NAMES):
        raise RuntimeError("synthetic label dimension mismatch")
    return x, aero, y
