from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import qmc

from experiments.common import ExperimentDataset, TrainingConfig, fixed_random_split, run_experiment
from psdg_net.providers import PhysicsBatch, default_fidelity, default_relation_strength
from psdg_net.schema import (
    DIMENSIONLESS,
    ENERGY,
    FORCE,
    LENGTH,
    MASS,
    POWER,
    PRESSURE,
    TEMPERATURE,
    TIME,
    TORQUE,
    ConstraintSpec,
    EquationSpec,
    GraphSchema,
    NodeSpec,
    PortSpec,
    RelationSpec,
    UnitDimension,
    VariableSpec,
)


def build_schema() -> GraphSchema:
    resistance_dim = UnitDimension(mass=1, length=2, time=-3, current=-2)
    variables = [
        VariableSpec("payload", "Context", 0.5, 8.0, "kg", MASS, "condition"),
        VariableSpec("motion_frequency", "Context", 0.15, 1.5, "Hz", UnitDimension(time=-1), "condition"),
        VariableSpec("ambient_temperature", "Context", 288.0, 318.0, "K", TEMPERATURE, "condition"),
        VariableSpec("sensor_sigma", "Sensor", 0.0002, 0.01, "rad", DIMENSIONLESS, "uncertainty"),
        VariableSpec("link1_length", "Structure", 0.25, 0.75, "m", LENGTH),
        VariableSpec("link2_length", "Structure", 0.20, 0.65, "m", LENGTH),
        VariableSpec("link1_radius", "Structure", 0.018, 0.065, "m", LENGTH),
        VariableSpec("link2_radius", "Structure", 0.015, 0.055, "m", LENGTH),
        VariableSpec("material_density", "Structure", 1200.0, 7800.0, "kg/m3", UnitDimension(mass=1, length=-3)),
        VariableSpec("yield_strength", "Structure", 1.2e8, 8.0e8, "Pa", PRESSURE),
        VariableSpec("torque_constant", "Motor", 0.06, 0.35, "N m/A", UnitDimension(mass=1, length=2, time=-2, current=-1)),
        VariableSpec("resistance", "Motor", 0.15, 1.6, "ohm", resistance_dim),
        VariableSpec("current_limit", "Motor", 3.0, 25.0, "A", UnitDimension(current=1)),
        VariableSpec("rotor_inertia", "Motor", 2.0e-5, 8.0e-4, "kg m2", UnitDimension(mass=1, length=2)),
        VariableSpec("gear_ratio", "Transmission", 18.0, 120.0),
        VariableSpec("gear_efficiency", "Transmission", 0.70, 0.96),
        VariableSpec("joint_friction", "Transmission", 0.02, 0.8, "N m s", UnitDimension(mass=1, length=2, time=-1)),
        VariableSpec("kp", "Controller", 15.0, 180.0),
        VariableSpec("kd", "Controller", 1.0, 25.0),
        VariableSpec("thermal_resistance", "Thermal", 0.4, 4.0, "K/W", UnitDimension(mass=-1, length=-2, time=3, temperature=1)),
        VariableSpec("thermal_capacitance", "Thermal", 40.0, 400.0, "J/K", UnitDimension(mass=1, length=2, time=-2, temperature=-1)),
        VariableSpec("stress_limit", "Safety", 8.0e7, 5.0e8, "Pa", PRESSURE, "constraint"),
        VariableSpec("temperature_limit", "Safety", 335.0, 415.0, "K", TEMPERATURE, "constraint"),
        VariableSpec("tracking_limit", "Safety", 0.003, 0.08, "rad", DIMENSIONLESS, "constraint"),
    ]
    nodes = [
        NodeSpec("Context", "context", "system"),
        NodeSpec("Structure", "component", "structural"),
        NodeSpec("Dynamics", "law", "mechanical"),
        NodeSpec("Motor", "component", "electromagnetic"),
        NodeSpec("Transmission", "component", "mechanical"),
        NodeSpec("Thermal", "law", "thermal"),
        NodeSpec("Sensor", "uncertainty", "instrumentation"),
        NodeSpec("Controller", "component", "control"),
        NodeSpec("Safety", "constraint", "system"),
        NodeSpec("Objective", "objective", "system"),
    ]
    ports = [
        PortSpec("mechanical_demand", "Dynamics", "mechanical", "out", "torque", TORQUE),
        PortSpec("transmission_load", "Transmission", "mechanical", "in", "torque", TORQUE),
        PortSpec("motor_torque", "Motor", "mechanical", "out", "torque", TORQUE),
        PortSpec("transmission_drive", "Transmission", "mechanical", "in", "torque", TORQUE),
        PortSpec("motor_heat", "Motor", "thermal", "out", "power", POWER),
        PortSpec("thermal_input", "Thermal", "thermal", "in", "power", POWER),
        PortSpec("sensor_signal", "Sensor", "signal", "out", "position"),
        PortSpec("controller_measurement", "Controller", "signal", "in", "position"),
    ]
    equations = [
        EquationSpec("rigid_body_dynamics", "Dynamics", ("payload", "link1_length", "link2_length", "motion_frequency"), ("required_torque",), "balance"),
        EquationSpec("beam_stress", "Structure", ("payload", "link1_radius", "link2_radius", "yield_strength"), ("maximum_stress",), "constitutive"),
        EquationSpec("electromechanical_conversion", "Motor", ("torque_constant", "current_limit", "gear_ratio", "gear_efficiency"), ("available_torque",), "energy_conversion"),
        EquationSpec("joule_thermal_balance", "Thermal", ("resistance", "thermal_resistance", "thermal_capacitance", "ambient_temperature"), ("winding_temperature",), "conservation"),
        EquationSpec("closed_loop_response", "Controller", ("kp", "kd", "sensor_sigma"), ("settling_time", "tracking_error"), "control_law"),
        EquationSpec("system_objective", "Objective", ("payload", "motion_frequency"), ("mass", "energy", "tracking"), "objective"),
    ]
    constraints = [
        ConstraintSpec("stress_margin", "Safety", ("stress_limit",), "ge"),
        ConstraintSpec("temperature_margin", "Safety", ("temperature_limit",), "ge"),
        ConstraintSpec("tracking_margin", "Safety", ("tracking_limit",), "ge"),
        ConstraintSpec("current_margin", "Safety", ("current_limit",), "ge"),
    ]
    relations = [
        RelationSpec("Context", "Dynamics", "operating_condition", "information", equation="rigid_body_dynamics", prior_confidence=0.98),
        RelationSpec("Structure", "Dynamics", "inertia_coupling", "energy", equation="rigid_body_dynamics", prior_confidence=0.98),
        RelationSpec("Dynamics", "Transmission", "torque_demand", "energy", "mechanical_demand", "transmission_load", equation="rigid_body_dynamics", prior_confidence=0.98),
        RelationSpec("Motor", "Transmission", "torque_supply", "energy", "motor_torque", "transmission_drive", equation="electromechanical_conversion", prior_confidence=0.98),
        RelationSpec("Transmission", "Dynamics", "actuation_feedback", "energy", equation="electromechanical_conversion", prior_confidence=0.92),
        RelationSpec("Motor", "Thermal", "loss_heat", "energy", "motor_heat", "thermal_input", equation="joule_thermal_balance", prior_confidence=0.98),
        RelationSpec("Context", "Thermal", "ambient_condition", "information", equation="joule_thermal_balance", prior_confidence=0.90),
        RelationSpec("Sensor", "Controller", "measurement", "information", "sensor_signal", "controller_measurement", equation="closed_loop_response", prior_confidence=0.98),
        RelationSpec("Controller", "Motor", "current_command", "information", equation="electromechanical_conversion", prior_confidence=0.95),
        RelationSpec("Dynamics", "Controller", "plant_response", "information", equation="closed_loop_response", prior_confidence=0.95),
        RelationSpec("Structure", "Safety", "stress_to_constraint", "force", equation="beam_stress", constraint="stress_margin", prior_confidence=0.98),
        RelationSpec("Thermal", "Safety", "temperature_to_constraint", "energy", equation="joule_thermal_balance", constraint="temperature_margin", prior_confidence=0.98),
        RelationSpec("Controller", "Safety", "tracking_to_constraint", "information", equation="closed_loop_response", constraint="tracking_margin", prior_confidence=0.95),
        RelationSpec("Motor", "Safety", "current_to_constraint", "energy", equation="electromechanical_conversion", constraint="current_margin", prior_confidence=0.95),
        RelationSpec("Structure", "Objective", "mass_to_objective", "information", equation="system_objective", prior_confidence=0.95),
        RelationSpec("Thermal", "Objective", "loss_to_objective", "energy", equation="system_objective", prior_confidence=0.90),
        RelationSpec("Controller", "Objective", "performance_to_objective", "information", equation="system_objective", prior_confidence=0.95),
        RelationSpec("Safety", "Objective", "constraint_to_objective", "information", equation="system_objective", prior_confidence=0.98),
    ]
    return GraphSchema("RobotElectromechanicalActuatorDesignGraph", nodes, variables, relations, ports, equations, constraints)


def _physics(schema: GraphSchema, x: np.ndarray):
    idx = schema.variable_index
    p = lambda name: x[:, idx[name]]
    l1, l2 = p("link1_length"), p("link2_length")
    r1, r2 = p("link1_radius"), p("link2_radius")
    density = p("material_density")
    area1, area2 = np.pi * r1**2, np.pi * r2**2
    mass1, mass2 = density * area1 * l1, density * area2 * l2
    total_mass = mass1 + mass2 + 0.12 * p("payload")
    inertia = mass1 * l1**2 / 3.0 + mass2 * (l1**2 + l1 * l2 + l2**2 / 3.0) + p("payload") * (l1 + l2) ** 2
    omega = 2.0 * np.pi * p("motion_frequency")
    gravity_torque = 9.81 * (0.5 * mass1 * l1 + mass2 * (l1 + 0.5 * l2) + p("payload") * (l1 + l2))
    dynamic_torque = 0.35 * inertia * omega**2
    required_torque = gravity_torque + dynamic_torque + p("joint_friction") * omega
    output_capacity = p("torque_constant") * p("current_limit") * p("gear_ratio") * p("gear_efficiency")
    motor_current = required_torque / np.maximum(p("torque_constant") * p("gear_ratio") * p("gear_efficiency"), 1.0e-5)
    copper_loss = motor_current**2 * p("resistance")
    mechanical_power = required_torque * omega / np.maximum(p("gear_efficiency"), 0.1)
    cycle_time = 1.0 / p("motion_frequency")
    energy = (mechanical_power + copper_loss) * cycle_time
    transient_factor = 1.0 - np.exp(-cycle_time / np.maximum(p("thermal_resistance") * p("thermal_capacitance"), 1.0e-4))
    winding_temperature = p("ambient_temperature") + copper_loss * p("thermal_resistance") * (0.35 + 0.65 * transient_factor)
    section1 = np.pi * r1**3 / 4.0
    section2 = np.pi * r2**3 / 4.0
    stress1 = gravity_torque / np.maximum(section1, 1.0e-8)
    stress2 = (9.81 * (0.5 * mass2 * l2 + p("payload") * l2) + 0.18 * dynamic_torque) / np.maximum(section2, 1.0e-8)
    max_stress = np.maximum(stress1, stress2)
    equivalent_inertia = inertia + p("rotor_inertia") * p("gear_ratio") ** 2
    natural_frequency = np.sqrt(p("kp") / np.maximum(equivalent_inertia, 1.0e-6))
    damping = p("kd") / (2.0 * np.sqrt(np.maximum(p("kp") * equivalent_inertia, 1.0e-8)))
    settling_time = 4.0 / np.maximum(natural_frequency * np.clip(damping, 0.08, 2.5), 1.0e-4)
    saturation = np.maximum(required_torque / np.maximum(output_capacity, 1.0e-5) - 1.0, 0.0)
    tracking_error = p("sensor_sigma") * (1.0 + 0.8 / np.maximum(damping, 0.08)) + 0.012 / (1.0 + natural_frequency) + 0.04 * saturation
    return {
        "mass": total_mass,
        "inertia": inertia,
        "required_torque": required_torque,
        "capacity": output_capacity,
        "current": motor_current,
        "copper_loss": copper_loss,
        "mechanical_power": mechanical_power,
        "energy": energy,
        "temperature": winding_temperature,
        "stress": max_stress,
        "natural_frequency": natural_frequency,
        "damping": damping,
        "settling": settling_time,
        "tracking": tracking_error,
    }


class RobotPhysicsProvider:
    def __init__(self, schema: GraphSchema):
        self.schema = schema

    def evaluate(self, x: np.ndarray) -> PhysicsBatch:
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        f = _physics(self.schema, x)
        extras = np.zeros((n, len(self.schema.nodes), 8), dtype=np.float32)
        put = lambda node, values: extras.__setitem__((slice(None), self.schema.node_index[node], slice(0, values.shape[1])), values.astype(np.float32))
        put("Structure", np.stack([np.log1p(f["mass"]), np.log1p(f["inertia"]), np.log1p(f["stress"]), x[:, self.schema.variable_index["yield_strength"]] / 8.0e8], axis=1))
        put("Dynamics", np.stack([np.log1p(f["required_torque"]), np.log1p(f["inertia"]), np.log1p(f["natural_frequency"]), f["damping"]], axis=1))
        put("Motor", np.stack([np.log1p(f["capacity"]), np.log1p(f["current"]), np.log1p(f["copper_loss"]), np.log1p(f["mechanical_power"])], axis=1))
        put("Transmission", np.stack([x[:, self.schema.variable_index["gear_ratio"]] / 120.0, x[:, self.schema.variable_index["gear_efficiency"]], np.log1p(f["required_torque"])], axis=1))
        put("Thermal", np.stack([np.log1p(f["copper_loss"]), f["temperature"] / 420.0, x[:, self.schema.variable_index["ambient_temperature"]] / 320.0], axis=1))
        put("Controller", np.stack([np.log1p(f["natural_frequency"]), f["damping"], np.log1p(f["settling"]), np.log1p(1000.0 * f["tracking"])], axis=1))
        idx = self.schema.variable_index
        stress_margin = (x[:, idx["stress_limit"]] - f["stress"]) / x[:, idx["stress_limit"]]
        temp_margin = (x[:, idx["temperature_limit"]] - f["temperature"]) / x[:, idx["temperature_limit"]]
        tracking_margin = (x[:, idx["tracking_limit"]] - f["tracking"]) / x[:, idx["tracking_limit"]]
        current_margin = (x[:, idx["current_limit"]] - f["current"]) / x[:, idx["current_limit"]]
        margins = np.stack([stress_margin, temp_margin, tracking_margin, current_margin], axis=1).astype(np.float32)
        put("Safety", margins)
        put("Objective", np.stack([np.log1p(f["mass"]), np.log1p(f["energy"]), np.log1p(f["settling"]), np.log1p(1000.0 * f["tracking"]), np.log1p(f["stress"]), f["temperature"] / 420.0], axis=1))

        correction = 0.012 * np.sin(3.0 * x[:, idx["link1_length"]]) + 0.008 * np.cos(x[:, idx["gear_ratio"]] / 12.0)
        residuals = np.stack(
            [
                np.abs(correction),
                0.01 * np.abs(np.sin(x[:, idx["link2_length"]] * 4.0)),
                0.012 * np.abs(np.cos(x[:, idx["gear_ratio"]] / 20.0)),
                0.015 * np.abs(np.sin(x[:, idx["thermal_resistance"]])),
                0.01 * np.abs(1.0 - np.clip(f["damping"], 0.0, 2.0)),
                0.008 * np.abs(margins.min(axis=1)),
            ],
            axis=1,
        ).astype(np.float32)
        strength = default_relation_strength(self.schema, n)
        torque_ratio = np.tanh(f["required_torque"] / np.maximum(f["capacity"], 1.0e-4))
        thermal_ratio = np.tanh(f["copper_loss"] / 100.0)
        for edge, relation in enumerate(self.schema.relations):
            if relation.transfer == "energy":
                strength[:, edge] *= 0.7 + 0.7 * torque_ratio
            if "Thermal" in {relation.source, relation.target}:
                strength[:, edge] *= 0.7 + 0.6 * thermal_ratio
        fidelity = default_fidelity(self.schema, n, 0.93)
        ranges = np.asarray([v.high - v.low for v in self.schema.variables], dtype=np.float32)
        fractions = np.asarray([0.012 if v.role == "design" else 0.025 if v.role == "condition" else 0.01 for v in self.schema.variables], dtype=np.float32)
        input_std = np.repeat((ranges * fractions).reshape(1, -1), n, axis=0)
        input_std[:, idx["payload"]] += 0.04 * x[:, idx["payload"]]
        input_std[:, idx["sensor_sigma"]] += x[:, idx["sensor_sigma"]]
        return PhysicsBatch(input_std, extras, residuals, margins, strength, fidelity)


def generate_dataset(schema: GraphSchema, n_power: int = 15, seed: int = 2027) -> ExperimentDataset:
    sampler = qmc.Sobol(d=len(schema.variables), scramble=True, seed=seed)
    unit = sampler.random_base2(m=int(n_power)).astype(np.float32)
    low = np.asarray([v.low for v in schema.variables], dtype=np.float32)
    high = np.asarray([v.high for v in schema.variables], dtype=np.float32)
    x = low + unit * (high - low)
    f = _physics(schema, x)
    idx = schema.variable_index
    nonlinear = 1.0 + 0.018 * np.sin(5.0 * x[:, idx["link1_length"]]) + 0.012 * np.cos(x[:, idx["gear_ratio"]] / 10.0)
    mass = f["mass"] * (1.0 + 0.008 * np.sin(x[:, idx["material_density"]] / 900.0))
    energy = f["energy"] * nonlinear + 0.03 * f["required_torque"] ** 1.35
    settling = f["settling"] * (1.0 + 0.015 * np.cos(x[:, idx["kp"]] / 25.0))
    tracking = f["tracking"] * (1.0 + 0.02 * np.sin(x[:, idx["kd"]]))
    stress = f["stress"] * (1.0 + 0.018 * np.sin(x[:, idx["link2_length"]] * 5.0))
    temperature = f["temperature"] + 1.5 * np.sin(x[:, idx["thermal_resistance"]] * 1.5) + 0.015 * f["copper_loss"]
    y = np.stack([mass, energy, settling, tracking, stress, temperature], axis=1).astype(np.float32)
    feasible = (
        (stress < x[:, idx["stress_limit"]])
        & (temperature < x[:, idx["temperature_limit"]])
        & (tracking < x[:, idx["tracking_limit"]])
        & (f["current"] < x[:, idx["current_limit"]])
    ).astype(np.float32)
    train_idx, val_idx, test_idx = fixed_random_split(len(x), seed)
    return ExperimentDataset(
        x,
        y,
        feasible,
        ("total_mass", "cycle_energy", "settling_time", "tracking_error", "maximum_stress", "winding_temperature"),
        train_idx,
        val_idx,
        test_idx,
        {"generator": "deterministic coupled rigid-body/electromagnetic/thermal/structural simulator", "sobol_power": n_power, "seed": seed},
    )


def run(output_root: Path, config: TrainingConfig) -> dict:
    schema = build_schema()
    return run_experiment("robot_electromechanical", schema, RobotPhysicsProvider(schema), generate_dataset(schema), output_root, config)
