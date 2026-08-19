from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .schema import GraphSchema
from .uncertainty import propagate_full_covariance


@dataclass
class CompiledPhysicalGraph:
    node_features: torch.Tensor
    node_kind_ids: torch.Tensor
    discipline_ids: torch.Tensor
    node_dimensions: torch.Tensor
    edge_source: torch.Tensor
    edge_target: torch.Tensor
    edge_type: torch.Tensor
    relation_strength: torch.Tensor
    relation_residual: torch.Tensor
    relation_fidelity: torch.Tensor
    relation_uncertainty: torch.Tensor
    causal_mask: torch.Tensor
    covariance: torch.Tensor
    node_std: torch.Tensor
    correlation: torch.Tensor


class ExecutableGraphCompiler(nn.Module):
    """Compile multidisciplinary variables and physics contracts into tensors."""

    def __init__(
        self,
        schema: GraphSchema,
        extra_dim: int = 8,
        uncertainty_steps: int = 4,
    ):
        super().__init__()
        self.schema = schema
        self.extra_dim = int(extra_dim)
        self.uncertainty_steps = int(uncertainty_steps)
        self.max_variables_per_node = max(
            1,
            max(sum(variable.node == node.name for variable in schema.variables) for node in schema.nodes),
        )
        node_index = schema.node_index
        kind_index = {name: i for i, name in enumerate(schema.node_kinds)}
        discipline_index = {name: i for i, name in enumerate(schema.disciplines)}
        relation_type_index = {name: i for i, name in enumerate(schema.relation_types)}

        lows = torch.tensor([item.low for item in schema.variables], dtype=torch.float32)
        highs = torch.tensor([item.high for item in schema.variables], dtype=torch.float32)
        variable_node = []
        variable_slot = []
        slot_counter = {node.name: 0 for node in schema.nodes}
        for variable in schema.variables:
            variable_node.append(node_index[variable.node])
            variable_slot.append(slot_counter[variable.node])
            slot_counter[variable.node] += 1

        node_dimensions = torch.zeros(len(schema.nodes), 7, dtype=torch.float32)
        node_dimension_count = torch.zeros(len(schema.nodes), 1, dtype=torch.float32)
        for variable in schema.variables:
            idx = node_index[variable.node]
            node_dimensions[idx] += torch.tensor(variable.dimension.vector(), dtype=torch.float32)
            node_dimension_count[idx] += 1.0
        node_dimensions = node_dimensions / node_dimension_count.clamp_min(1.0)

        edge_source = torch.tensor([node_index[item.source] for item in schema.relations], dtype=torch.long)
        edge_target = torch.tensor([node_index[item.target] for item in schema.relations], dtype=torch.long)
        edge_type = torch.tensor([relation_type_index[item.relation_type] for item in schema.relations], dtype=torch.long)
        prior = torch.tensor([item.prior_confidence for item in schema.relations], dtype=torch.float32)

        equation_index = {item.name: i for i, item in enumerate(schema.equations)}
        constraint_index = {item.name: i for i, item in enumerate(schema.constraints)}
        edge_equation = torch.tensor([equation_index.get(item.equation, -1) for item in schema.relations], dtype=torch.long)
        edge_constraint = torch.tensor([constraint_index.get(item.constraint, -1) for item in schema.relations], dtype=torch.long)

        causal_mask = torch.zeros(len(schema.nodes), len(schema.nodes), dtype=torch.bool)
        causal_mask.fill_diagonal_(True)
        for relation in schema.relations:
            src, dst = node_index[relation.source], node_index[relation.target]
            causal_mask[dst, src] = True
            if relation.reverse_confidence > 0.0:
                causal_mask[src, dst] = True

        self.register_buffer("lows", lows)
        self.register_buffer("highs", highs)
        self.register_buffer("variable_node", torch.tensor(variable_node, dtype=torch.long))
        self.register_buffer("variable_slot", torch.tensor(variable_slot, dtype=torch.long))
        self.register_buffer("node_kind_ids", torch.tensor([kind_index[n.kind] for n in schema.nodes], dtype=torch.long))
        self.register_buffer(
            "discipline_ids",
            torch.tensor([discipline_index[n.discipline] for n in schema.nodes], dtype=torch.long),
        )
        self.register_buffer("node_dimensions", node_dimensions)
        self.register_buffer("edge_source", edge_source)
        self.register_buffer("edge_target", edge_target)
        self.register_buffer("edge_type", edge_type)
        self.register_buffer("relation_prior", prior)
        self.register_buffer("edge_equation", edge_equation)
        self.register_buffer("edge_constraint", edge_constraint)
        self.register_buffer("causal_mask", causal_mask)

    @property
    def node_feature_dim(self) -> int:
        # normalized means, local standard deviations, variable-presence mask, physics extras,
        # equation residual, constraint margin, fidelity, propagated node standard deviation.
        return self.max_variables_per_node * 3 + self.extra_dim + 4

    def _node_variable_tensor(self, x: torch.Tensor, input_std: torch.Tensor) -> torch.Tensor:
        denom = (self.highs - self.lows).clamp_min(1.0e-8)
        normalized = (2.0 * (x - self.lows) / denom - 1.0).clamp(-4.0, 4.0)
        normalized_std = (2.0 * input_std.abs() / denom).clamp(0.0, 4.0)
        batch = x.shape[0]
        values = x.new_zeros((batch, len(self.schema.nodes), self.max_variables_per_node))
        stds = x.new_zeros(values.shape)
        mask = x.new_zeros(values.shape)
        for index in range(len(self.schema.variables)):
            node = int(self.variable_node[index])
            slot = int(self.variable_slot[index])
            values[:, node, slot] = normalized[:, index]
            stds[:, node, slot] = normalized_std[:, index]
            mask[:, node, slot] = 1.0
        return torch.cat([values, stds, mask], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        input_std: Optional[torch.Tensor] = None,
        node_extras: Optional[torch.Tensor] = None,
        equation_residuals: Optional[torch.Tensor] = None,
        constraint_margins: Optional[torch.Tensor] = None,
        relation_strength: Optional[torch.Tensor] = None,
        fidelity: Optional[torch.Tensor] = None,
    ) -> CompiledPhysicalGraph:
        if x.ndim != 2 or x.shape[1] != len(self.schema.variables):
            raise ValueError(f"x must have shape [batch, {len(self.schema.variables)}]")
        batch = x.shape[0]
        nodes = len(self.schema.nodes)
        edges = len(self.schema.relations)
        if input_std is None:
            input_std = torch.zeros_like(x)
        if input_std.shape != x.shape:
            raise ValueError("input_std must match x")
        if node_extras is None:
            node_extras = x.new_zeros((batch, nodes, self.extra_dim))
        if node_extras.shape != (batch, nodes, self.extra_dim):
            raise ValueError(f"node_extras must have shape {(batch, nodes, self.extra_dim)}")
        if equation_residuals is None:
            equation_residuals = x.new_zeros((batch, len(self.schema.equations)))
        if constraint_margins is None:
            constraint_margins = x.new_zeros((batch, len(self.schema.constraints)))
        if relation_strength is None:
            relation_strength = self.relation_prior.view(1, -1).expand(batch, -1)
        if relation_strength.shape != (batch, edges):
            raise ValueError(f"relation_strength must have shape {(batch, edges)}")
        if fidelity is None:
            fidelity = x.new_ones((batch, edges))
        if fidelity.shape != (batch, edges):
            raise ValueError(f"fidelity must have shape {(batch, edges)}")

        node_variables = self._node_variable_tensor(x, input_std)
        local_node_variance = x.new_zeros((batch, nodes))
        denom = (self.highs - self.lows).clamp_min(1.0e-8)
        normalized_var = (2.0 * input_std.abs() / denom).pow(2)
        for index in range(len(self.schema.variables)):
            local_node_variance[:, int(self.variable_node[index])] += normalized_var[:, index]

        relation_matrix = x.new_zeros((batch, nodes, nodes))
        for edge in range(edges):
            relation_matrix[:, int(self.edge_source[edge]), int(self.edge_target[edge])] += (
                relation_strength[:, edge].abs() * self.relation_prior[edge] * fidelity[:, edge].clamp(0.0, 1.0)
            )
        covariance, node_std, correlation = propagate_full_covariance(
            local_node_variance,
            relation_matrix,
            steps=self.uncertainty_steps,
        )

        node_equation = x.new_zeros((batch, nodes))
        for index, equation in enumerate(self.schema.equations):
            node = self.schema.node_index[equation.node]
            node_equation[:, node] += equation_residuals[:, index].abs() / max(equation.normalized_scale, 1.0e-8)
        node_constraint = x.new_zeros((batch, nodes))
        for index, constraint in enumerate(self.schema.constraints):
            node = self.schema.node_index[constraint.node]
            node_constraint[:, node] += constraint_margins[:, index]
        node_fidelity = x.new_zeros((batch, nodes))
        node_fidelity_count = x.new_zeros((batch, nodes))
        for edge in range(edges):
            dst = int(self.edge_target[edge])
            node_fidelity[:, dst] += fidelity[:, edge]
            node_fidelity_count[:, dst] += 1.0
        node_fidelity = node_fidelity / node_fidelity_count.clamp_min(1.0)
        node_features = torch.cat(
            [
                node_variables,
                node_extras,
                node_equation.unsqueeze(-1),
                node_constraint.unsqueeze(-1),
                node_fidelity.unsqueeze(-1),
                node_std.unsqueeze(-1),
            ],
            dim=-1,
        )

        edge_residual = x.new_zeros((batch, edges))
        for edge in range(edges):
            eq = int(self.edge_equation[edge])
            con = int(self.edge_constraint[edge])
            if eq >= 0:
                edge_residual[:, edge] += equation_residuals[:, eq].abs()
            if con >= 0:
                edge_residual[:, edge] += torch.relu(-constraint_margins[:, con])
        relation_uncertainty = node_std[:, self.edge_source] * relation_strength.abs()
        relation_uncertainty = relation_uncertainty + covariance[:, self.edge_source, self.edge_target].abs().sqrt()

        return CompiledPhysicalGraph(
            node_features=node_features,
            node_kind_ids=self.node_kind_ids,
            discipline_ids=self.discipline_ids,
            node_dimensions=self.node_dimensions,
            edge_source=self.edge_source,
            edge_target=self.edge_target,
            edge_type=self.edge_type,
            relation_strength=relation_strength,
            relation_residual=edge_residual,
            relation_fidelity=fidelity,
            relation_uncertainty=relation_uncertainty,
            causal_mask=self.causal_mask,
            covariance=covariance,
            node_std=node_std,
            correlation=correlation,
        )
