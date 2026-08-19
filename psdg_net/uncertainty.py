from __future__ import annotations

from typing import Tuple

import torch


def stable_causal_transport(
    relation_matrix: torch.Tensor,
    attenuation: float = 0.72,
) -> torch.Tensor:
    """Normalize a directed source->target coupling matrix for stable propagation."""

    if relation_matrix.ndim != 3:
        raise ValueError("relation_matrix must have shape [batch, nodes, nodes]")
    coupling = relation_matrix.abs()
    incoming = coupling.sum(dim=1, keepdim=True).clamp_min(1.0)
    return float(attenuation) * coupling / incoming


def propagate_full_covariance(
    local_variance: torch.Tensor,
    relation_matrix: torch.Tensor,
    steps: int = 4,
    attenuation: float = 0.72,
    variance_ceiling: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propagate every source uncertainty through the complete causal graph.

    The fixed-point iteration implements Sigma = D + A^T Sigma A.  A is a
    stable, sample-dependent dimensionless coupling matrix.  Every variable
    contributes to D and therefore to all
    causally reachable nodes.
    """

    if local_variance.ndim != 2:
        raise ValueError("local_variance must have shape [batch, nodes]")
    if relation_matrix.shape[:2] != local_variance.shape or relation_matrix.shape[2] != local_variance.shape[1]:
        raise ValueError("relation_matrix shape is incompatible with local_variance")
    transport = stable_causal_transport(relation_matrix, attenuation=attenuation)
    base = torch.diag_embed(local_variance.clamp_min(0.0))
    covariance = base
    for _ in range(max(1, int(steps))):
        propagated = transport.transpose(1, 2).matmul(covariance).matmul(transport)
        covariance = base + propagated
        diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1).clamp(max=float(variance_ceiling))
        covariance = covariance - torch.diag_embed(torch.diagonal(covariance, dim1=-2, dim2=-1))
        covariance = covariance + torch.diag_embed(diagonal)
    node_std = torch.sqrt(torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(1.0e-12))
    denom = node_std.unsqueeze(-1) * node_std.unsqueeze(-2)
    correlation = (covariance / denom.clamp_min(1.0e-8)).clamp(-1.0, 1.0)
    return covariance, node_std, correlation
