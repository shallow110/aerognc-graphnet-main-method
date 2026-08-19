"""Graph-conditioned design generation and predictor-guided refinement.

The generator follows the manuscript protocol: context variables are kept in
the full graph, while only the requested design variables are decoded from a
graph state, a parameter identity embedding and a 12-dimensional latent code.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .compiler import CompiledPhysicalGraph, ExecutableGraphCompiler
from .model import ExecutablePhysicalGraphEncoder
from .schema import GraphSchema


def normalize_variables(x: torch.Tensor, lows: torch.Tensor, highs: torch.Tensor) -> torch.Tensor:
    """Map raw variables to the compiler's bounded [-1, 1] representation."""
    return (2.0 * (x - lows) / (highs - lows).clamp_min(1.0e-8) - 1.0).clamp(-1.0, 1.0)


def denormalize_variables(x_normalized: torch.Tensor, lows: torch.Tensor, highs: torch.Tensor) -> torch.Tensor:
    """Map normalized variables back to raw units and enforce valid bounds."""
    raw = lows + 0.5 * (x_normalized.clamp(-1.0, 1.0) + 1.0) * (highs - lows)
    return raw.clamp(min=lows, max=highs)


class GraphConditionalDesignGenerator(nn.Module):
    """Conditional VAE decoder operating on executable physical-semantic graphs."""

    def __init__(
        self,
        schema: GraphSchema,
        design_indices: Sequence[int],
        *,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 3,
        latent_dim: int = 12,
        extra_dim: int = 8,
        dropout: float = 0.02,
    ):
        super().__init__()
        self.schema = schema
        self.latent_dim = int(latent_dim)
        self.design_indices = tuple(int(index) for index in design_indices)
        if not self.design_indices:
            raise ValueError("design_indices must contain at least one variable")
        if len(set(self.design_indices)) != len(self.design_indices):
            raise ValueError("design_indices must be unique")
        if min(self.design_indices) < 0 or max(self.design_indices) >= len(schema.variables):
            raise IndexError("design_indices contain a variable outside the schema")

        self.compiler = ExecutableGraphCompiler(schema, extra_dim=extra_dim)
        self.encoder = ExecutablePhysicalGraphEncoder(
            schema,
            self.compiler,
            hidden=hidden,
            heads=heads,
            layers=layers,
            dropout=dropout,
        )
        self.design_embed = nn.Sequential(
            nn.Linear(len(self.design_indices), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # The executable graph encoder returns global, objective, constraint
        # and max pools concatenated into a 4*hidden context vector.
        self.posterior = nn.Sequential(nn.LayerNorm(hidden * 5), nn.Linear(hidden * 5, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden)
        self.param_embed = nn.Embedding(len(schema.variables), hidden)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("design_param_ids", torch.tensor(self.design_indices, dtype=torch.long))
        self.register_buffer(
            "design_node_ids",
            torch.tensor(
                [schema.node_index[schema.variables[index].node] for index in self.design_indices],
                dtype=torch.long,
            ),
        )

    def _graph(
        self,
        x_full: torch.Tensor,
        physics: Optional[Dict[str, torch.Tensor]],
        graph: Optional[CompiledPhysicalGraph],
    ) -> CompiledPhysicalGraph:
        return graph if graph is not None else self.compiler(x_full, **dict(physics or {}))

    def encode(
        self,
        x_full: torch.Tensor,
        design_normalized: torch.Tensor,
        physics: Optional[Dict[str, torch.Tensor]] = None,
        graph: Optional[CompiledPhysicalGraph] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        graph = self._graph(x_full, physics, graph)
        _, pooled, _ = self.encoder(graph)
        if design_normalized.shape != (x_full.shape[0], len(self.design_indices)):
            raise ValueError("design_normalized must have shape [batch, len(design_indices)]")
        posterior = self.posterior(torch.cat([pooled, self.design_embed(design_normalized)], dim=-1))
        return self.mu(posterior), self.logvar(posterior).clamp(-8.0, 6.0)

    def decode_normalized(
        self,
        x_full: torch.Tensor,
        z: torch.Tensor,
        physics: Optional[Dict[str, torch.Tensor]] = None,
        graph: Optional[CompiledPhysicalGraph] = None,
    ) -> torch.Tensor:
        graph = self._graph(x_full, physics, graph)
        node_h, _, _ = self.encoder(graph)
        node_h = node_h[:, self.design_node_ids, :]
        param_h = self.param_embed(self.design_param_ids).unsqueeze(0).expand(x_full.shape[0], -1, -1)
        latent_h = self.latent_proj(z).unsqueeze(1)
        return torch.tanh(self.decoder(node_h + param_h + latent_h).squeeze(-1))

    def decode(
        self,
        x_full: torch.Tensor,
        z: torch.Tensor,
        physics: Optional[Dict[str, torch.Tensor]] = None,
        graph: Optional[CompiledPhysicalGraph] = None,
    ) -> torch.Tensor:
        """Decode normalized design variables in [-1, 1]."""
        return self.decode_normalized(x_full, z, physics=physics, graph=graph)

    def forward(
        self,
        x_full: torch.Tensor,
        design_normalized: torch.Tensor,
        physics: Optional[Dict[str, torch.Tensor]] = None,
        graph: Optional[CompiledPhysicalGraph] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph = self._graph(x_full, physics, graph)
        mu, logvar = self.encode(x_full, design_normalized, graph=graph)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode_normalized(x_full, z, graph=graph), mu, logvar


def generator_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0e-3,
) -> torch.Tensor:
    """Reconstruction-plus-KL objective used for the conditional VAE."""
    recon = nn.functional.smooth_l1_loss(reconstruction, target)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon + float(kl_weight) * kl


def project_design_to_bounds(
    design: torch.Tensor,
    design_indices: Sequence[int],
    schema: GraphSchema,
) -> torch.Tensor:
    indices = torch.as_tensor(tuple(int(index) for index in design_indices), dtype=torch.long, device=design.device)
    lows = torch.as_tensor([schema.variables[int(index)].low for index in indices], dtype=design.dtype, device=design.device)
    highs = torch.as_tensor([schema.variables[int(index)].high for index in indices], dtype=design.dtype, device=design.device)
    return design.clamp(min=lows, max=highs)


def insert_design_values(
    x_context: torch.Tensor,
    design: torch.Tensor,
    design_indices: Sequence[int],
) -> torch.Tensor:
    """Insert decoded raw design variables while leaving context untouched."""
    result = x_context.clone()
    indices = torch.as_tensor(tuple(int(index) for index in design_indices), dtype=torch.long, device=result.device)
    result[:, indices] = design.to(device=result.device, dtype=result.dtype)
    return result


def predictor_guided_refinement(
    predictor: nn.Module,
    x_context: torch.Tensor,
    design_initial: torch.Tensor,
    design_indices: Sequence[int],
    schema: GraphSchema,
    *,
    physics_fn: Optional[Callable[[torch.Tensor], Optional[Dict[str, torch.Tensor]]]] = None,
    steps: int = 20,
    step_size: float = 0.025,
    objective_index: int = 0,
) -> Tuple[torch.Tensor, List[float]]:
    """Optimize only design variables with the predictor, keeping context fixed."""
    device = next(predictor.parameters()).device
    context = x_context.to(device)
    design = design_initial.detach().clone().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([design], lr=float(step_size))
    history: List[float] = []

    for _ in range(int(steps)):
        bounded = project_design_to_bounds(design, design_indices, schema)
        x_full = insert_design_values(context, bounded, design_indices)
        physics = physics_fn(x_full) if physics_fn is not None else None
        output = predictor(x_full, physics=physics)
        prediction, feasibility_logit = output[:2]
        objective = prediction[:, int(objective_index)].clamp_min(0.0)
        loss = torch.log1p(objective).mean() + 0.05 * nn.functional.softplus(-feasibility_logit).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([design], 3.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    return project_design_to_bounds(design.detach(), design_indices, schema), history
