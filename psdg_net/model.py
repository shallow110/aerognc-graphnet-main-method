from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .compiler import CompiledPhysicalGraph, ExecutableGraphCompiler
from .schema import GraphSchema


class PhysicsContractAttention(nn.Module):
    """Relation-specific attention with causal masks and relaxable physics priors."""

    def __init__(self, hidden: int, heads: int, relation_types: int, dropout: float = 0.02):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.head_dim = hidden // heads
        self.norm1 = nn.LayerNorm(hidden, eps=1.0e-3)
        self.norm2 = nn.LayerNorm(hidden, eps=1.0e-3)
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        # A bounded diagonal value operator per relation type.  It preserves
        # relation-specific semantics without allowing an unconstrained matrix
        # to arbitrarily rotate (and numerically dominate) the latent space.
        self.relation_value = nn.Parameter(torch.zeros(relation_types, heads, self.head_dim))
        self.relation_bias = nn.Parameter(torch.zeros(relation_types, heads))
        self.relation_correction = nn.Parameter(torch.zeros(relation_types, heads))
        self.log_sensitivity_weight = nn.Parameter(torch.full((heads,), -0.2))
        self.log_uncertainty_weight = nn.Parameter(torch.full((heads,), -1.0))
        self.log_residual_weight = nn.Parameter(torch.full((heads,), -0.4))
        self.out = nn.Linear(hidden, hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 3), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden * 3, hidden))
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, graph: CompiledPhysicalGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, nodes, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        logits = q.matmul(k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = graph.causal_mask.view(1, 1, nodes, nodes)
        logits = logits.masked_fill(~mask, -1.0e4)

        # Build all relation priors in one scatter operation.  The previous
        # edge-by-edge implementation was needlessly slow for large design
        # pools and made long pure-model training impractical.
        sensitivity_weight = nn.functional.softplus(self.log_sensitivity_weight).view(1, self.heads)
        uncertainty_weight = nn.functional.softplus(self.log_uncertainty_weight).view(1, self.heads)
        residual_weight = nn.functional.softplus(self.log_residual_weight).view(1, self.heads)
        edge_type = graph.edge_type.long()
        strength = graph.relation_strength.abs().clamp_min(1.0e-5)
        fidelity = graph.relation_fidelity.clamp(1.0e-4, 1.0)
        residual = graph.relation_residual.abs()
        uncertainty = graph.relation_uncertainty.abs()
        prior = torch.log1p(strength).unsqueeze(-1) * sensitivity_weight
        prior = prior + torch.log(fidelity).unsqueeze(-1)
        prior = prior + torch.log1p(uncertainty).unsqueeze(-1) * uncertainty_weight
        prior = prior - torch.log1p(residual).unsqueeze(-1) * residual_weight
        prior = prior + self.relation_bias[edge_type].unsqueeze(0)
        prior = prior + 0.5 * torch.tanh(self.relation_correction[edge_type]).unsqueeze(0)
        edge_index = (graph.edge_target * nodes + graph.edge_source).long()
        prior_flat = logits.new_zeros((batch, self.heads, nodes * nodes))
        prior_flat = prior_flat.scatter_add(2, edge_index.view(1, 1, -1).expand(batch, self.heads, -1), prior.permute(0, 2, 1))
        prior_bias = prior_flat.view(batch, self.heads, nodes, nodes)

        attention = torch.softmax((logits + prior_bias).clamp(-60.0, 60.0), dim=-1)
        mixed = attention.matmul(v)
        # Recompute sparse edge corrections with their normalized attention.
        source_value = v[:, :, graph.edge_source.long(), :]
        value_gain = 1.0 + 0.25 * torch.tanh(self.relation_value[edge_type]).permute(1, 0, 2)
        transformed = source_value * value_gain.unsqueeze(0)
        edge_weight = attention[:, :, graph.edge_target.long(), graph.edge_source.long()].unsqueeze(-1)
        edge_delta = edge_weight * (transformed - source_value)
        relation_delta = v.new_zeros((batch, self.heads, nodes, self.head_dim))
        relation_delta = relation_delta.scatter_add(
            2,
            graph.edge_target.long().view(1, 1, -1, 1).expand(batch, self.heads, -1, self.head_dim),
            edge_delta,
        )
        mixed = mixed + relation_delta
        mixed = mixed.transpose(1, 2).reshape(batch, nodes, self.hidden)
        scale = 0.50 * torch.sigmoid(self.residual_scale)
        x = x + scale * self.dropout(self.out(mixed))
        x = 12.0 * torch.tanh(x / 12.0)
        x = x + scale * self.dropout(self.ffn(self.norm2(x)))
        x = 12.0 * torch.tanh(x / 12.0)
        return x, attention


class ExecutablePhysicalGraphEncoder(nn.Module):
    def __init__(
        self,
        schema: GraphSchema,
        compiler: ExecutableGraphCompiler,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.02,
    ):
        super().__init__()
        self.schema = schema
        self.compiler = compiler
        self.value_projection = nn.Sequential(
            nn.LayerNorm(compiler.node_feature_dim, eps=1.0e-3),
            nn.Linear(compiler.node_feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.kind_embedding = nn.Embedding(len(schema.node_kinds), hidden)
        self.discipline_embedding = nn.Embedding(len(schema.disciplines), hidden)
        self.dimension_projection = nn.Sequential(nn.Linear(7, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList(
            [PhysicsContractAttention(hidden, heads, len(schema.relation_types), dropout=dropout) for _ in range(layers)]
        )
        self.pool_gate = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        objective = [i for i, node in enumerate(schema.nodes) if node.kind == "objective"]
        constraint = [i for i, node in enumerate(schema.nodes) if node.kind == "constraint"]
        self.register_buffer("objective_nodes", torch.tensor(objective, dtype=torch.long))
        self.register_buffer("constraint_nodes", torch.tensor(constraint, dtype=torch.long))
        self.last_attention = None

    def forward(self, graph: CompiledPhysicalGraph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = graph.node_features.shape[0]
        h = self.value_projection(graph.node_features)
        h = h + self.kind_embedding(graph.node_kind_ids).unsqueeze(0)
        h = h + self.discipline_embedding(graph.discipline_ids).unsqueeze(0)
        h = h + self.dimension_projection(graph.node_dimensions).unsqueeze(0)
        initial_gate = torch.softmax(self.pool_gate(h).squeeze(-1), dim=-1)
        initial_pool = torch.sum(h * initial_gate.unsqueeze(-1), dim=1)
        attentions = []
        for layer in self.layers:
            h, attention = layer(h, graph)
            attentions.append(attention)
        self.last_attention = attentions[-1].detach()
        gate = torch.softmax(self.pool_gate(h).squeeze(-1), dim=-1)
        global_pool = torch.sum(h * gate.unsqueeze(-1), dim=1)
        if self.objective_nodes.numel():
            objective_pool = h[:, self.objective_nodes].mean(dim=1)
        else:
            objective_pool = global_pool
        if self.constraint_nodes.numel():
            constraint_pool = h[:, self.constraint_nodes].mean(dim=1)
        else:
            constraint_pool = global_pool
        max_pool = h.max(dim=1).values
        return h, torch.cat([global_pool, objective_pool, constraint_pool, max_pool], dim=-1), initial_pool


class ExecutablePhysicalGraphPredictor(nn.Module):
    def __init__(
        self,
        schema: GraphSchema,
        out_dim: int,
        *,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 3,
        extra_dim: int = 8,
        dropout: float = 0.02,
    ):
        super().__init__()
        self.schema = schema
        self.compiler = ExecutableGraphCompiler(schema, extra_dim=extra_dim)
        self.encoder = ExecutablePhysicalGraphEncoder(
            schema,
            self.compiler,
            hidden=hidden,
            heads=heads,
            layers=layers,
            dropout=dropout,
        )
        # Preserve an explicit path from executable equation/constraint outputs
        # to the task heads.  This is not a raw-vector bypass: it reads only the
        # typed objective and constraint contract nodes produced by the compiler.
        self.contract_projection = nn.Sequential(
            nn.LayerNorm(self.compiler.node_feature_dim * 2),
            nn.Linear(self.compiler.node_feature_dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.readout_slots = 3
        self.readout_token_projection = nn.Sequential(
            nn.LayerNorm(self.compiler.node_feature_dim, eps=1.0e-3),
            nn.Linear(self.compiler.node_feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.target_queries = nn.Parameter(torch.empty(out_dim, self.readout_slots, hidden))
        nn.init.normal_(self.target_queries, mean=0.0, std=hidden ** -0.5)
        semantic_contract_dim = len(schema.nodes) * self.compiler.node_feature_dim
        self.semantic_contract_encoder = nn.Sequential(
            nn.LayerNorm(semantic_contract_dim, eps=1.0e-3),
            nn.Linear(semantic_contract_dim, hidden * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
        )
        self.raw_input_projection = nn.Sequential(
            nn.LayerNorm(len(schema.variables), eps=1.0e-3),
            nn.Linear(len(schema.variables), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.kernel_contract_dim = 15 if self.compiler.extra_dim >= 23 else 0
        if self.kernel_contract_dim:
            # The flight provider stores the deterministic physical-kernel
            # channels in the Objective contract (after the eight graph-state
            # channels). This branch exposes that typed contract to the heads
            # without bypassing the graph compiler or using a target proxy.
            self.kernel_contract_projection = nn.Sequential(
                nn.LayerNorm(self.kernel_contract_dim),
                nn.Linear(self.kernel_contract_dim, hidden * 4),
                nn.SiLU(),
                nn.Linear(hidden * 4, hidden * 4),
                nn.SiLU(),
                nn.Linear(hidden * 4, hidden),
                nn.SiLU(),
            )
        else:
            self.kernel_contract_projection = None
        fused = hidden * (8 if self.kernel_contract_dim else 7)
        self.shared = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
        )
        self.regression_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden * (2 + self.readout_slots), eps=1.0e-3),
                    nn.Linear(hidden * (2 + self.readout_slots), hidden * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden * 2, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(out_dim)
            ]
        )
        self.feasibility_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
    def forward(
        self,
        x: torch.Tensor,
        physics: Optional[Dict[str, torch.Tensor]] = None,
        graph: Optional[CompiledPhysicalGraph] = None,
    ):
        graph = graph if graph is not None else self.compiler(x, **dict(physics or {}))
        _, pooled, initial_pool = self.encoder(graph)
        design_scale = (self.compiler.highs - self.compiler.lows).clamp_min(1.0e-8)
        normalized_x = (2.0 * (x - self.compiler.lows) / design_scale - 1.0).clamp(-4.0, 4.0)
        raw_context = self.raw_input_projection(normalized_x)
        if self.encoder.objective_nodes.numel():
            objective_contract = graph.node_features[:, self.encoder.objective_nodes].mean(dim=1)
        else:
            objective_contract = graph.node_features.mean(dim=1)
        if self.encoder.constraint_nodes.numel():
            constraint_contract = graph.node_features[:, self.encoder.constraint_nodes].mean(dim=1)
        else:
            constraint_contract = graph.node_features.mean(dim=1)
        contract = self.contract_projection(torch.cat([objective_contract, constraint_contract], dim=-1))
        contexts = [pooled, contract, initial_pool, raw_context]
        if self.kernel_contract_projection is not None:
            objective_start = self.compiler.max_variables_per_node * 3 + 8
            objective_kernel = graph.node_features[:, self.encoder.objective_nodes[0], objective_start : objective_start + 15]
            contexts.append(self.kernel_contract_projection(objective_kernel))
        shared = self.shared(torch.cat(contexts, dim=-1))
        semantic_contract = self.semantic_contract_encoder(graph.node_features.flatten(start_dim=1))
        readout_tokens = self.readout_token_projection(graph.node_features)
        scores = (
            readout_tokens.unsqueeze(1).unsqueeze(1)
            * self.target_queries.unsqueeze(0).unsqueeze(3)
        ).sum(dim=-1) / (readout_tokens.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        contexts = (
            weights.unsqueeze(-1) * readout_tokens.unsqueeze(1).unsqueeze(1)
        ).sum(dim=3).flatten(start_dim=2)
        regression = torch.cat(
            [
                head(torch.cat([shared, semantic_contract, contexts[:, index]], dim=-1))
                for index, head in enumerate(self.regression_heads)
            ],
            dim=-1,
        )
        feasibility = self.feasibility_head(shared).squeeze(-1)
        return regression, feasibility, graph


# Manuscript-facing canonical name.  The longer class name remains as a
# compatibility import for existing experiment scripts.
PSDGNet = ExecutablePhysicalGraphPredictor
