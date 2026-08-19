from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import qmc

from experiments.common import ExperimentDataset, TrainingConfig, fixed_random_split, run_experiment
from psdg_net.providers import PhysicsBatch, default_fidelity, default_relation_strength
from psdg_net.schema import (
    DIMENSIONLESS,
    FLOW_RATE,
    LENGTH,
    MASS,
    POWER,
    PRESSURE,
    TEMPERATURE,
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
    viscosity = UnitDimension(mass=1, length=-1, time=-1)
    conductivity = UnitDimension(mass=1, length=1, time=-3, temperature=-1)
    heat_capacity = UnitDimension(length=2, time=-2, temperature=-1)
    variables = [
        VariableSpec("heat_load", "Context", 500.0, 12000.0, "W", POWER, "condition"),
        VariableSpec("ambient_temperature", "Context", 283.0, 318.0, "K", TEMPERATURE, "condition"),
        VariableSpec("channel_length", "Geometry", 0.25, 2.0, "m", LENGTH),
        VariableSpec("hydraulic_diameter", "Geometry", 0.003, 0.03, "m", LENGTH),
        VariableSpec("flow_area", "Geometry", 2.0e-5, 1.5e-3, "m2", UnitDimension(length=2)),
        VariableSpec("wall_thickness", "Geometry", 0.0008, 0.009, "m", LENGTH),
        VariableSpec("solid_width", "Geometry", 0.02, 0.20, "m", LENGTH),
        VariableSpec("roughness", "Geometry", 1.0e-6, 1.5e-4, "m", LENGTH),
        VariableSpec("mass_flow", "Fluid", 0.015, 0.65, "kg/s", UnitDimension(mass=1, time=-1)),
        VariableSpec("fluid_density", "Fluid", 750.0, 1100.0, "kg/m3", UnitDimension(mass=1, length=-3)),
        VariableSpec("fluid_viscosity", "Fluid", 0.00035, 0.004, "Pa s", viscosity),
        VariableSpec("fluid_cp", "Fluid", 1800.0, 4300.0, "J/kg/K", heat_capacity),
        VariableSpec("fluid_conductivity", "Fluid", 0.12, 0.70, "W/m/K", conductivity),
        VariableSpec("pump_efficiency", "Pump", 0.45, 0.88),
        VariableSpec("wall_conductivity", "Material", 12.0, 230.0, "W/m/K", conductivity),
        VariableSpec("elastic_modulus", "Material", 3.0e9, 210.0e9, "Pa", PRESSURE),
        VariableSpec("thermal_expansion", "Material", 4.0e-6, 28.0e-6, "1/K", UnitDimension(temperature=-1)),
        VariableSpec("yield_strength", "Material", 7.0e7, 8.0e8, "Pa", PRESSURE),
        VariableSpec("material_density", "Material", 1200.0, 8900.0, "kg/m3", UnitDimension(mass=1, length=-3)),
        VariableSpec("pressure_limit", "Safety", 3.0e4, 1.2e6, "Pa", PRESSURE, "constraint"),
        VariableSpec("temperature_limit", "Safety", 320.0, 500.0, "K", TEMPERATURE, "constraint"),
        VariableSpec("stress_limit", "Safety", 5.0e7, 6.0e8, "Pa", PRESSURE, "constraint"),
        VariableSpec("deformation_limit", "Safety", 2.0e-5, 4.0e-3, "m", LENGTH, "constraint"),
    ]
    nodes = [
        NodeSpec("Context", "context", "system"),
        NodeSpec("Geometry", "component", "mechanical"),
        NodeSpec("Fluid", "state", "fluid"),
        NodeSpec("Hydraulics", "law", "fluid"),
        NodeSpec("Pump", "component", "fluid"),
        NodeSpec("Thermal", "law", "thermal"),
        NodeSpec("Material", "component", "materials"),
        NodeSpec("Structure", "law", "structural"),
        NodeSpec("Safety", "constraint", "system"),
        NodeSpec("Objective", "objective", "system"),
    ]
    ports = [
        PortSpec("fluid_out", "Hydraulics", "fluid", "out", "pressure_flow"),
        PortSpec("pump_in", "Pump", "fluid", "in", "pressure_flow"),
        PortSpec("heat_source", "Context", "thermal", "out", "heat", POWER),
        PortSpec("thermal_in", "Thermal", "thermal", "in", "heat", POWER),
        PortSpec("temperature_out", "Thermal", "thermal", "out", "temperature", TEMPERATURE),
        PortSpec("structure_temperature", "Structure", "thermal", "in", "temperature", TEMPERATURE),
    ]
    equations = [
        EquationSpec("darcy_weisbach", "Hydraulics", ("channel_length", "hydraulic_diameter", "flow_area", "roughness", "mass_flow", "fluid_density", "fluid_viscosity"), ("pressure_drop",), "conservation"),
        EquationSpec("pump_power", "Pump", ("mass_flow", "fluid_density", "pump_efficiency"), ("pump_power",), "energy_conversion"),
        EquationSpec("convective_heat_transfer", "Thermal", ("heat_load", "mass_flow", "fluid_cp", "fluid_conductivity", "hydraulic_diameter"), ("outlet_temperature", "wall_temperature"), "conservation"),
        EquationSpec("thermoelastic_stress", "Structure", ("elastic_modulus", "thermal_expansion", "wall_thickness"), ("combined_stress", "deformation"), "constitutive"),
        EquationSpec("system_objective", "Objective", ("heat_load", "pump_efficiency", "material_density"), ("energy", "mass", "performance"), "objective"),
    ]
    constraints = [
        ConstraintSpec("pressure_margin", "Safety", ("pressure_limit",), "ge"),
        ConstraintSpec("temperature_margin", "Safety", ("temperature_limit",), "ge"),
        ConstraintSpec("stress_margin", "Safety", ("stress_limit",), "ge"),
        ConstraintSpec("deformation_margin", "Safety", ("deformation_limit",), "ge"),
    ]
    relations = [
        RelationSpec("Geometry", "Hydraulics", "flow_geometry", "material", equation="darcy_weisbach", prior_confidence=0.98),
        RelationSpec("Fluid", "Hydraulics", "fluid_state", "material", equation="darcy_weisbach", prior_confidence=0.98),
        RelationSpec("Hydraulics", "Pump", "pressure_flow", "energy", "fluid_out", "pump_in", equation="pump_power", prior_confidence=0.98),
        RelationSpec("Context", "Thermal", "heat_input", "energy", "heat_source", "thermal_in", equation="convective_heat_transfer", prior_confidence=0.98),
        RelationSpec("Fluid", "Thermal", "convective_transport", "energy", equation="convective_heat_transfer", prior_confidence=0.98),
        RelationSpec("Geometry", "Thermal", "heat_transfer_area", "energy", equation="convective_heat_transfer", prior_confidence=0.94),
        RelationSpec("Material", "Thermal", "conduction", "energy", equation="convective_heat_transfer", prior_confidence=0.95),
        RelationSpec("Thermal", "Structure", "thermal_load", "energy", "temperature_out", "structure_temperature", equation="thermoelastic_stress", prior_confidence=0.98),
        RelationSpec("Geometry", "Structure", "structural_geometry", "force", equation="thermoelastic_stress", prior_confidence=0.98),
        RelationSpec("Material", "Structure", "constitutive", "force", equation="thermoelastic_stress", prior_confidence=0.98),
        RelationSpec("Hydraulics", "Structure", "pressure_load", "force", equation="thermoelastic_stress", prior_confidence=0.92),
        RelationSpec("Hydraulics", "Safety", "pressure_to_constraint", "force", constraint="pressure_margin", prior_confidence=0.98),
        RelationSpec("Thermal", "Safety", "temperature_to_constraint", "energy", constraint="temperature_margin", prior_confidence=0.98),
        RelationSpec("Structure", "Safety", "stress_to_constraint", "force", constraint="stress_margin", prior_confidence=0.98),
        RelationSpec("Pump", "Objective", "power_to_objective", "energy", equation="system_objective", prior_confidence=0.95),
        RelationSpec("Thermal", "Objective", "temperature_to_objective", "information", equation="system_objective", prior_confidence=0.90),
        RelationSpec("Structure", "Objective", "mass_stress_to_objective", "information", equation="system_objective", prior_confidence=0.92),
        RelationSpec("Safety", "Objective", "constraint_to_objective", "information", equation="system_objective", prior_confidence=0.98),
    ]
    return GraphSchema("ThermalFluidMechanicalExecutableDesignGraph", nodes, variables, relations, ports, equations, constraints)


def _physics(schema: GraphSchema, x: np.ndarray):
    idx = schema.variable_index
    p = lambda name: x[:, idx[name]]
    velocity = p("mass_flow") / np.maximum(p("fluid_density") * p("flow_area"), 1.0e-9)
    reynolds = p("fluid_density") * velocity * p("hydraulic_diameter") / np.maximum(p("fluid_viscosity"), 1.0e-9)
    relative_roughness = p("roughness") / p("hydraulic_diameter")
    laminar = 64.0 / np.maximum(reynolds, 1.0)
    turbulent = 0.25 / np.maximum(np.log10(relative_roughness / 3.7 + 5.74 / np.maximum(reynolds, 1.0) ** 0.9) ** 2, 1.0e-5)
    blend = 1.0 / (1.0 + np.exp(-(reynolds - 3000.0) / 500.0))
    friction = (1.0 - blend) * laminar + blend * turbulent
    pressure_drop = friction * p("channel_length") / p("hydraulic_diameter") * 0.5 * p("fluid_density") * velocity**2
    pump_power = pressure_drop * p("mass_flow") / np.maximum(p("fluid_density") * p("pump_efficiency"), 1.0e-8)
    outlet_temperature = p("ambient_temperature") + p("heat_load") / np.maximum(p("mass_flow") * p("fluid_cp"), 1.0e-8)
    prandtl = p("fluid_cp") * p("fluid_viscosity") / np.maximum(p("fluid_conductivity"), 1.0e-8)
    nusselt_lam = 3.66 * np.ones_like(reynolds)
    nusselt_turb = 0.023 * np.maximum(reynolds, 1.0) ** 0.8 * np.maximum(prandtl, 0.1) ** 0.4
    nusselt = (1.0 - blend) * nusselt_lam + blend * nusselt_turb
    h = nusselt * p("fluid_conductivity") / p("hydraulic_diameter")
    wetted_area = 4.0 * p("flow_area") / p("hydraulic_diameter") * p("channel_length")
    wall_temperature = outlet_temperature + p("heat_load") / np.maximum(h * wetted_area, 1.0e-6)
    wall_temperature += p("heat_load") * p("wall_thickness") / np.maximum(p("wall_conductivity") * wetted_area, 1.0e-6)
    thermal_stress = 0.62 * p("elastic_modulus") * p("thermal_expansion") * np.maximum(wall_temperature - p("ambient_temperature"), 0.0)
    pressure_stress = pressure_drop * p("hydraulic_diameter") / np.maximum(2.0 * p("wall_thickness"), 1.0e-8)
    combined_stress = np.sqrt(thermal_stress**2 + 3.0 * pressure_stress**2)
    deformation = combined_stress / np.maximum(p("elastic_modulus"), 1.0e-8) * p("channel_length")
    solid_volume = p("channel_length") * p("solid_width") * p("wall_thickness") * 2.0
    solid_mass = p("material_density") * solid_volume
    return {
        "velocity": velocity,
        "reynolds": reynolds,
        "friction": friction,
        "pressure_drop": pressure_drop,
        "pump_power": pump_power,
        "outlet_temperature": outlet_temperature,
        "h": h,
        "wall_temperature": wall_temperature,
        "thermal_stress": thermal_stress,
        "pressure_stress": pressure_stress,
        "combined_stress": combined_stress,
        "deformation": deformation,
        "solid_mass": solid_mass,
    }


class ThermalFluidPhysicsProvider:
    def __init__(self, schema: GraphSchema):
        self.schema = schema

    def evaluate(self, x: np.ndarray) -> PhysicsBatch:
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        idx = self.schema.variable_index
        f = _physics(self.schema, x)
        extras = np.zeros((n, len(self.schema.nodes), 8), dtype=np.float32)
        def put(node, *columns):
            values = np.stack(columns, axis=1).astype(np.float32)
            extras[:, self.schema.node_index[node], : values.shape[1]] = values
        put("Hydraulics", np.log1p(f["reynolds"]), f["friction"], np.log1p(f["pressure_drop"]), np.log1p(f["velocity"]))
        put("Pump", np.log1p(f["pump_power"]), x[:, idx["pump_efficiency"]], np.log1p(f["pressure_drop"]))
        put("Thermal", f["outlet_temperature"] / 500.0, f["wall_temperature"] / 550.0, np.log1p(f["h"]), np.log1p(x[:, idx["heat_load"]]))
        put("Structure", np.log1p(f["combined_stress"]), np.log1p(1.0e6 * f["deformation"]), np.log1p(f["solid_mass"]), np.log1p(f["thermal_stress"]), np.log1p(f["pressure_stress"]))
        pressure_margin = (x[:, idx["pressure_limit"]] - f["pressure_drop"]) / x[:, idx["pressure_limit"]]
        temperature_margin = (x[:, idx["temperature_limit"]] - f["wall_temperature"]) / x[:, idx["temperature_limit"]]
        stress_margin = (x[:, idx["stress_limit"]] - f["combined_stress"]) / x[:, idx["stress_limit"]]
        deformation_margin = (x[:, idx["deformation_limit"]] - f["deformation"]) / x[:, idx["deformation_limit"]]
        margins = np.stack([pressure_margin, temperature_margin, stress_margin, deformation_margin], axis=1).astype(np.float32)
        put("Safety", pressure_margin, temperature_margin, stress_margin, deformation_margin)
        put("Objective", np.log1p(f["pressure_drop"]), np.log1p(f["pump_power"]), f["wall_temperature"] / 550.0, np.log1p(f["combined_stress"]), np.log1p(1.0e6 * f["deformation"]), np.log1p(f["solid_mass"]))
        discrepancy = 0.012 * np.abs(np.sin(np.log1p(f["reynolds"])))
        residuals = np.stack(
            [
                discrepancy,
                0.01 * np.abs(1.0 - x[:, idx["pump_efficiency"]]),
                0.015 * np.abs(np.cos(x[:, idx["hydraulic_diameter"]] * 200.0)),
                0.012 * np.abs(np.sin(x[:, idx["wall_thickness"]] * 300.0)),
                0.008 * np.abs(margins.min(axis=1)),
            ],
            axis=1,
        ).astype(np.float32)
        strength = default_relation_strength(self.schema, n)
        flow_strength = np.tanh(np.log1p(f["reynolds"]) / 10.0)
        thermal_strength = np.tanh(np.maximum(f["wall_temperature"] - x[:, idx["ambient_temperature"]], 0.0) / 100.0)
        structural_strength = np.tanh(f["combined_stress"] / 2.0e8)
        for edge, relation in enumerate(self.schema.relations):
            if relation.transfer == "material":
                strength[:, edge] *= 0.65 + 0.7 * flow_strength
            elif relation.transfer == "energy":
                strength[:, edge] *= 0.65 + 0.7 * thermal_strength
            elif relation.transfer == "force":
                strength[:, edge] *= 0.65 + 0.7 * structural_strength
        fidelity = default_fidelity(self.schema, n, 0.94)
        ranges = np.asarray([v.high - v.low for v in self.schema.variables], dtype=np.float32)
        fractions = np.asarray([0.012 if v.role == "design" else 0.025 if v.role == "condition" else 0.01 for v in self.schema.variables], dtype=np.float32)
        input_std = np.repeat((ranges * fractions).reshape(1, -1), n, axis=0)
        input_std[:, idx["mass_flow"]] += 0.025 * x[:, idx["mass_flow"]]
        input_std[:, idx["heat_load"]] += 0.03 * x[:, idx["heat_load"]]
        return PhysicsBatch(input_std, extras, residuals, margins, strength, fidelity)


def generate_dataset(schema: GraphSchema, n_power: int = 15, seed: int = 2028) -> ExperimentDataset:
    sampler = qmc.Sobol(d=len(schema.variables), scramble=True, seed=seed)
    unit = sampler.random_base2(m=int(n_power)).astype(np.float32)
    low = np.asarray([v.low for v in schema.variables], dtype=np.float32)
    high = np.asarray([v.high for v in schema.variables], dtype=np.float32)
    x = low + unit * (high - low)
    f = _physics(schema, x)
    idx = schema.variable_index
    correction = 1.0 + 0.018 * np.sin(np.log1p(f["reynolds"])) + 0.01 * np.cos(x[:, idx["roughness"]] * 4.0e4)
    pressure_drop = f["pressure_drop"] * correction
    pump_power = pressure_drop * x[:, idx["mass_flow"]] / np.maximum(x[:, idx["fluid_density"]] * x[:, idx["pump_efficiency"]], 1.0e-8)
    outlet_temperature = f["outlet_temperature"] + 0.6 * np.sin(x[:, idx["mass_flow"]] * 7.0)
    wall_temperature = f["wall_temperature"] + 1.2 * np.cos(x[:, idx["wall_conductivity"]] / 30.0)
    combined_stress = f["combined_stress"] * (1.0 + 0.015 * np.sin(x[:, idx["wall_thickness"]] * 350.0))
    deformation = f["deformation"] * (1.0 + 0.012 * np.cos(x[:, idx["elastic_modulus"]] / 2.0e10))
    solid_mass = f["solid_mass"] * (1.0 + 0.006 * np.sin(x[:, idx["solid_width"]] * 40.0))
    y = np.stack([pressure_drop, pump_power, outlet_temperature, wall_temperature, combined_stress, deformation, solid_mass], axis=1).astype(np.float32)
    feasible = (
        (pressure_drop < x[:, idx["pressure_limit"]])
        & (wall_temperature < x[:, idx["temperature_limit"]])
        & (combined_stress < x[:, idx["stress_limit"]])
        & (deformation < x[:, idx["deformation_limit"]])
    ).astype(np.float32)
    train_idx, val_idx, test_idx = fixed_random_split(len(x), seed)
    return ExperimentDataset(
        x,
        y,
        feasible,
        ("pressure_drop", "pump_power", "outlet_temperature", "wall_temperature", "combined_stress", "deformation", "solid_mass"),
        train_idx,
        val_idx,
        test_idx,
        {"generator": "deterministic Darcy-Weisbach/convection/thermoelastic coupled simulator", "sobol_power": n_power, "seed": seed},
    )


def run(output_root: Path, config: TrainingConfig) -> dict:
    schema = build_schema()
    return run_experiment("thermal_fluid_mechanical", schema, ThermalFluidPhysicsProvider(schema), generate_dataset(schema), output_root, config)
