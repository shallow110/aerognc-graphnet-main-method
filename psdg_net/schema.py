from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class UnitDimension:
    """SI base-dimension exponents: mass, length, time, current, temperature, amount, luminosity."""

    mass: float = 0.0
    length: float = 0.0
    time: float = 0.0
    current: float = 0.0
    temperature: float = 0.0
    amount: float = 0.0
    luminosity: float = 0.0

    def vector(self) -> Tuple[float, ...]:
        return (
            self.mass,
            self.length,
            self.time,
            self.current,
            self.temperature,
            self.amount,
            self.luminosity,
        )


DIMENSIONLESS = UnitDimension()
LENGTH = UnitDimension(length=1)
TIME = UnitDimension(time=1)
VELOCITY = UnitDimension(length=1, time=-1)
ACCELERATION = UnitDimension(length=1, time=-2)
MASS = UnitDimension(mass=1)
FORCE = UnitDimension(mass=1, length=1, time=-2)
TORQUE = UnitDimension(mass=1, length=2, time=-2)
PRESSURE = UnitDimension(mass=1, length=-1, time=-2)
POWER = UnitDimension(mass=1, length=2, time=-3)
ENERGY = UnitDimension(mass=1, length=2, time=-2)
TEMPERATURE = UnitDimension(temperature=1)
FLOW_RATE = UnitDimension(length=3, time=-1)


@dataclass(frozen=True)
class VariableSpec:
    name: str
    node: str
    low: float
    high: float
    unit: str = "1"
    dimension: UnitDimension = DIMENSIONLESS
    role: str = "design"
    port: str = ""
    uncertainty_group: str = "independent"

    def validate(self) -> None:
        if not self.name:
            raise ValueError("variable name must be non-empty")
        if not self.node:
            raise ValueError("variable node must be non-empty")
        if not self.high > self.low:
            raise ValueError(f"invalid bounds for {self.name}: [{self.low}, {self.high}]")


@dataclass(frozen=True)
class PortSpec:
    name: str
    node: str
    domain: str
    direction: str
    quantity: str
    dimension: UnitDimension = DIMENSIONLESS

    def validate(self) -> None:
        if self.direction not in {"in", "out", "bidirectional"}:
            raise ValueError(f"invalid port direction {self.direction!r}")


@dataclass(frozen=True)
class NodeSpec:
    name: str
    kind: str
    discipline: str
    variables: Tuple[str, ...] = ()
    ports: Tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class EquationSpec:
    name: str
    node: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    law_type: str
    normalized_scale: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    node: str
    variables: Tuple[str, ...]
    sense: str = "le"
    description: str = ""

    def validate(self) -> None:
        if self.sense not in {"le", "ge", "eq"}:
            raise ValueError(f"invalid constraint sense {self.sense!r}")


@dataclass(frozen=True)
class RelationSpec:
    source: str
    target: str
    relation_type: str
    transfer: str
    source_port: str = ""
    target_port: str = ""
    equation: str = ""
    constraint: str = ""
    prior_confidence: float = 1.0
    causal: bool = True
    reverse_confidence: float = 0.0

    def validate(self) -> None:
        if not 0.0 < self.prior_confidence <= 1.0:
            raise ValueError("prior_confidence must be in (0, 1]")
        if not 0.0 <= self.reverse_confidence <= 1.0:
            raise ValueError("reverse_confidence must be in [0, 1]")


@dataclass
class GraphSchema:
    name: str
    nodes: Sequence[NodeSpec]
    variables: Sequence[VariableSpec]
    relations: Sequence[RelationSpec]
    ports: Sequence[PortSpec] = field(default_factory=tuple)
    equations: Sequence[EquationSpec] = field(default_factory=tuple)
    constraints: Sequence[ConstraintSpec] = field(default_factory=tuple)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nodes = tuple(self.nodes)
        self.variables = tuple(self.variables)
        self.relations = tuple(self.relations)
        self.ports = tuple(self.ports)
        self.equations = tuple(self.equations)
        self.constraints = tuple(self.constraints)
        self.validate()

    @property
    def node_index(self) -> Dict[str, int]:
        return {node.name: i for i, node in enumerate(self.nodes)}

    @property
    def variable_index(self) -> Dict[str, int]:
        return {variable.name: i for i, variable in enumerate(self.variables)}

    @property
    def relation_types(self) -> Tuple[str, ...]:
        return tuple(sorted({relation.relation_type for relation in self.relations} | {"self"}))

    @property
    def node_kinds(self) -> Tuple[str, ...]:
        return tuple(sorted({node.kind for node in self.nodes}))

    @property
    def disciplines(self) -> Tuple[str, ...]:
        return tuple(sorted({node.discipline for node in self.nodes}))

    def validate(self) -> None:
        if not self.name:
            raise ValueError("schema name must be non-empty")
        node_names = [node.name for node in self.nodes]
        if len(node_names) != len(set(node_names)):
            raise ValueError("node names must be unique")
        variable_names = [variable.name for variable in self.variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("variable names must be unique")
        node_set = set(node_names)
        variable_set = set(variable_names)
        for variable in self.variables:
            variable.validate()
            if variable.node not in node_set:
                raise ValueError(f"variable {variable.name} references missing node {variable.node}")
        port_names = {port.name for port in self.ports}
        for port in self.ports:
            port.validate()
            if port.node not in node_set:
                raise ValueError(f"port {port.name} references missing node {port.node}")
        equation_names = {equation.name for equation in self.equations}
        constraint_names = {constraint.name for constraint in self.constraints}
        for equation in self.equations:
            if equation.node not in node_set:
                raise ValueError(f"equation {equation.name} references missing node {equation.node}")
            # Inputs bind to design variables. Outputs may be derived physical
            # quantities produced by an executable provider and therefore need
            # not appear in the external design vector.
            missing = set(equation.inputs) - variable_set
            if missing:
                raise ValueError(f"equation {equation.name} references missing variables {sorted(missing)}")
        for constraint in self.constraints:
            constraint.validate()
            if constraint.node not in node_set:
                raise ValueError(f"constraint {constraint.name} references missing node {constraint.node}")
        for relation in self.relations:
            relation.validate()
            if relation.source not in node_set or relation.target not in node_set:
                raise ValueError(f"relation {relation.source}->{relation.target} references missing node")
            if relation.source_port and relation.source_port not in port_names:
                raise ValueError(f"missing source port {relation.source_port}")
            if relation.target_port and relation.target_port not in port_names:
                raise ValueError(f"missing target port {relation.target_port}")
            if relation.equation and relation.equation not in equation_names:
                raise ValueError(f"missing equation {relation.equation}")
            if relation.constraint and relation.constraint not in constraint_names:
                raise ValueError(f"missing constraint {relation.constraint}")

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "nodes": [asdict(item) for item in self.nodes],
            "variables": [asdict(item) for item in self.variables],
            "ports": [asdict(item) for item in self.ports],
            "equations": [asdict(item) for item in self.equations],
            "constraints": [asdict(item) for item in self.constraints],
            "relations": [asdict(item) for item in self.relations],
            "metadata": dict(self.metadata),
        }


def variables_for_node(schema: GraphSchema, node_name: str) -> List[VariableSpec]:
    return [variable for variable in schema.variables if variable.node == node_name]


def ensure_tuple(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(values)
