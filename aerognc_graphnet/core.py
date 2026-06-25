from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from .schema import PARAMETER_SPECS, bounds


NODE_NAMES = [
    "InitialNode",
    "GeometryNode",
    "AeroMapNode",
    "NavBeliefNode",
    "GuidanceNode",
    "ControlNode",
    "ActuatorNode",
    "SafetyNode",
    "CostQueryNode",
]


NODE_KIND = {
    "InitialNode": "condition",
    "GeometryNode": "design_geometry",
    "AeroMapNode": "derived_physics",
    "NavBeliefNode": "belief_uncertainty",
    "GuidanceNode": "reference_logic",
    "ControlNode": "control_design",
    "ActuatorNode": "actuator_design",
    "SafetyNode": "constraint",
    "CostQueryNode": "objective_query",
}


DESIGN_GRAPH_EDGES = [
    ("InitialNode", "GuidanceNode", "sets_reference_regime"),
    ("InitialNode", "AeroMapNode", "sets_aero_regime"),
    ("GeometryNode", "AeroMapNode", "geometry_to_aero"),
    ("GeometryNode", "ActuatorNode", "geometry_to_actuator"),
    ("AeroMapNode", "ControlNode", "aero_response_to_control"),
    ("AeroMapNode", "SafetyNode", "aero_load_to_safety"),
    ("AeroMapNode", "CostQueryNode", "trajectory_to_objective"),
    ("NavBeliefNode", "GuidanceNode", "belief_to_guidance"),
    ("NavBeliefNode", "ControlNode", "belief_to_control"),
    ("NavBeliefNode", "SafetyNode", "belief_to_safety"),
    ("NavBeliefNode", "CostQueryNode", "belief_to_objective"),
    ("GuidanceNode", "ControlNode", "reference_to_control"),
    ("ControlNode", "ActuatorNode", "command_to_deflection"),
    ("ControlNode", "SafetyNode", "control_to_safety"),
    ("ControlNode", "CostQueryNode", "effort_to_objective"),
    ("ActuatorNode", "AeroMapNode", "deflection_to_aero"),
    ("SafetyNode", "CostQueryNode", "barrier_to_objective"),
]


PARAMETER_NODE = {spec.name: spec.node for spec in PARAMETER_SPECS}


PARAMETER_SLOT = {
    "InitialNode": ["speed_scale", "pitch_delta_deg"],
    "GeometryNode": ["length_diameter_ratio", "nose_haack_c", "fin_position", "fin_span", "fin_area_scale"],
    "NavBeliefNode": ["pos_sigma", "vel_sigma", "att_sigma_deg"],
    "ControlNode": ["ctrl_gain", "track_weight", "terminal_weight", "control_weight"],
    "ActuatorNode": ["delta_gain", "max_deflection_deg"],
    "SafetyNode": ["q_max", "alpha_max_deg", "p_trace_max"],
}


AERO_DERIVED_FEATURES = [
    "log_drag_scale",
    "log_lift_scale",
    "log_moment_scale",
    "log_control_force_scale",
    "log_control_moment_scale",
    "log_roll_damping_scale",
]


@dataclass(frozen=True)
class DesignGraphNode:
    name: str
    kind: str
    parameters: Tuple[str, ...]


def _node_index(name: str) -> int:
    return NODE_NAMES.index(name)


def _attention_bias_matrix() -> np.ndarray:
    n = len(NODE_NAMES)
    bias = np.zeros((n, n), dtype=np.float32)
    adjacency = np.zeros((n, n), dtype=np.float32)
    for src, dst, _ in DESIGN_GRAPH_EDGES:
        i = _node_index(src)
        j = _node_index(dst)
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
        bias[i, j] += 1.20
        bias[j, i] += 0.75

    strong_pairs = [
        ("GeometryNode", "AeroMapNode", 1.40),
        ("GeometryNode", "ActuatorNode", 0.90),
        ("AeroMapNode", "ControlNode", 1.00),
        ("NavBeliefNode", "GuidanceNode", 1.10),
        ("NavBeliefNode", "ControlNode", 1.00),
        ("ControlNode", "ActuatorNode", 1.20),
        ("SafetyNode", "CostQueryNode", 1.25),
    ]
    for a, b, value in strong_pairs:
        i = _node_index(a)
        j = _node_index(b)
        bias[i, j] += value
        bias[j, i] += value

    dist = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    dist[adjacency > 0] = 1.0
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    reachable = np.isfinite(dist) & (dist > 1.0)
    bias[reachable] += 0.25 / dist[reachable]
    return bias


def _uncertainty_route_matrix() -> np.ndarray:
    route = np.zeros((len(NODE_NAMES), len(NODE_NAMES)), dtype=np.float32)
    nav = _node_index("NavBeliefNode")
    for name in ["GuidanceNode", "ControlNode", "SafetyNode", "CostQueryNode"]:
        route[nav, _node_index(name)] = 1.0
        route[_node_index(name), nav] = 0.5
    return route


def design_graph_metadata() -> Dict[str, object]:
    nodes: List[DesignGraphNode] = []
    for name in NODE_NAMES:
        params = tuple(PARAMETER_SLOT.get(name, ()))
        if name == "AeroMapNode":
            params = tuple(AERO_DERIVED_FEATURES)
        nodes.append(DesignGraphNode(name=name, kind=NODE_KIND[name], parameters=params))
    return {
        "name": "PhysicsTypedHeterogeneousDesignGraph",
        "nodes": [asdict(node) for node in nodes],
        "edges": [{"source": s, "target": t, "kind": k} for s, t, k in DESIGN_GRAPH_EDGES],
        "parameter_binding": dict(PARAMETER_NODE),
        "aero_derived_features": list(AERO_DERIVED_FEATURES),
        "attention_bias": _attention_bias_matrix().tolist(),
        "uncertainty_routes": _uncertainty_route_matrix().tolist(),
    }


class PhysicalDesignGraphBuilder(nn.Module):
    def __init__(self, value_dim: int = 8):
        super().__init__()
        self.value_dim = int(value_dim)
        lows, highs = bounds()
        self.register_buffer("lows", torch.tensor(lows, dtype=torch.float32))
        self.register_buffer("highs", torch.tensor(highs, dtype=torch.float32))
        param_node = []
        param_slot = []
        for spec in PARAMETER_SPECS:
            node = PARAMETER_NODE[spec.name]
            param_node.append(_node_index(node))
            param_slot.append(PARAMETER_SLOT[node].index(spec.name))
        self.register_buffer("param_node", torch.tensor(param_node, dtype=torch.long))
        self.register_buffer("param_slot", torch.tensor(param_slot, dtype=torch.long))
        self.param_lookup = {spec.name: i for i, spec in enumerate(PARAMETER_SPECS)}

    def _p(self, x_raw: torch.Tensor, name: str) -> torch.Tensor:
        return x_raw[:, self.param_lookup[name]]

    def _fill_direct_parameters(self, values: torch.Tensor, x_raw: torch.Tensor) -> None:
        denom = (self.highs - self.lows).clamp_min(1.0e-6)
        x_unit = (2.0 * (x_raw - self.lows) / denom - 1.0).clamp(-3.0, 3.0)
        for i in range(len(PARAMETER_SPECS)):
            values[:, self.param_node[i], self.param_slot[i]] = x_unit[:, i]

    def _fill_aero_mapping(self, values: torch.Tensor, x_raw: torch.Tensor, aero_node_features: torch.Tensor = None) -> None:
        if aero_node_features is None:
            raise ValueError("aero_node_features must be supplied; no real aerodynamic model is bundled")
        if aero_node_features.ndim != 2 or aero_node_features.shape[1] != len(AERO_DERIVED_FEATURES):
            raise ValueError("aero_node_features must have shape [batch, 6]")
        values[:, _node_index("AeroMapNode"), : len(AERO_DERIVED_FEATURES)] = aero_node_features.to(
            device=values.device, dtype=values.dtype
        ).clamp(-3.0, 3.0)

    def forward(self, x_raw: torch.Tensor, aero_node_features: torch.Tensor = None) -> torch.Tensor:
        if x_raw.ndim != 2:
            raise ValueError("x_raw must have shape [batch, num_parameters]")
        values = x_raw.new_zeros((x_raw.shape[0], len(NODE_NAMES), self.value_dim))
        self._fill_direct_parameters(values, x_raw)
        self._fill_aero_mapping(values, x_raw, aero_node_features=aero_node_features)
        return values


class PhysicsBiasedGraphAttention(nn.Module):
    def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.05):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden))
        self.dropout = nn.Dropout(float(dropout))
        self.residual_scale = 0.5
        self.register_buffer("param_kernel_scale", torch.tensor(0.15, dtype=torch.float32))
        self.register_buffer("uncertainty_scale", torch.tensor(0.20, dtype=torch.float32))
        self.register_buffer("physical_bias_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("physical_bias", torch.tensor(_attention_bias_matrix(), dtype=torch.float32))
        self.register_buffer("uncertainty_routes", torch.tensor(_uncertainty_route_matrix(), dtype=torch.float32))
        self._reset_stable_parameters()

    def _reset_stable_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.45)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, node_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        h = self.norm1(x)
        bsz, nodes, _ = h.shape
        qkv = self.qkv(h).view(bsz, nodes, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        structural = self.physical_bias_scale * self.physical_bias
        param_dist = torch.cdist(node_values.detach(), node_values.detach(), p=2.0)
        param_kernel = -self.param_kernel_scale.clamp(0.0, 1.0) * param_dist.clamp(0.0, 20.0)
        nav_idx = _node_index("NavBeliefNode")
        nav_strength = node_values.detach()[:, nav_idx, :].abs().mean(dim=-1).view(bsz, 1, 1)
        uncertainty_bias = self.uncertainty_scale.clamp(0.0, 2.0) * nav_strength.clamp(0.0, 5.0) * self.uncertainty_routes

        logits = logits + structural.view(1, 1, nodes, nodes)
        logits = logits + param_kernel.view(bsz, 1, nodes, nodes)
        logits = logits + uncertainty_bias.view(bsz, 1, nodes, nodes)
        logits = logits.clamp(-60.0, 60.0)
        attn = torch.softmax(logits, dim=-1)
        mixed = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, nodes, self.hidden)
        x = residual + self.residual_scale * self.dropout(self.out(mixed))
        x = x + self.residual_scale * self.dropout(self.ffn(self.norm2(x)))
        return x, attn


class PhysicalGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192, layers: int = 3, value_dim: int = 8):
        super().__init__()
        if int(in_dim) != len(PARAMETER_SPECS):
            raise ValueError("Expected %d parameters, got %d" % (len(PARAMETER_SPECS), int(in_dim)))
        self.builder = PhysicalDesignGraphBuilder(value_dim=value_dim)
        self.value_proj = nn.Linear(value_dim, hidden)
        self.node_embed = nn.Embedding(len(NODE_NAMES), hidden)
        self.kind_embed = nn.Embedding(len(set(NODE_KIND.values())), hidden)
        kind_to_id = {kind: i for i, kind in enumerate(sorted(set(NODE_KIND.values())))}
        self.register_buffer("node_ids", torch.arange(len(NODE_NAMES), dtype=torch.long))
        self.register_buffer("kind_ids", torch.tensor([kind_to_id[NODE_KIND[name]] for name in NODE_NAMES], dtype=torch.long))
        self.blocks = nn.ModuleList([PhysicsBiasedGraphAttention(hidden=hidden, heads=4) for _ in range(int(layers))])
        self.pool_gate = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.last_attention = None

    def forward(self, x_raw: torch.Tensor, aero_node_features: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_values = self.builder(x_raw, aero_node_features=aero_node_features)
        node_ids = self.node_ids.view(1, -1).expand(x_raw.shape[0], -1)
        kind_ids = self.kind_ids.view(1, -1).expand(x_raw.shape[0], -1)
        h = self.value_proj(node_values) + self.node_embed(node_ids) + self.kind_embed(kind_ids)
        attentions = []
        for block in self.blocks:
            h, attn = block(h, node_values)
            attentions.append(attn)
        self.last_attention = attentions[-1].detach()
        gate = torch.softmax(self.pool_gate(h).squeeze(-1), dim=-1)
        pooled = torch.sum(h * gate.unsqueeze(-1), dim=1)
        return h, pooled, node_values


class PhysicalGraphPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192, out_dim: int = 6, layers: int = 3, value_dim: int = 8):
        super().__init__()
        self.encoder = PhysicalGraphEncoder(in_dim=in_dim, hidden=hidden, layers=layers, value_dim=value_dim)
        lows, highs = bounds()
        self.register_buffer("raw_lows", torch.tensor(lows, dtype=torch.float32))
        self.register_buffer("raw_highs", torch.tensor(highs, dtype=torch.float32))
        self.raw_feature_dim = int(in_dim) + 15
        self.raw_skip = nn.Sequential(
            nn.Linear(self.raw_feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        fused_dim = hidden * 2
        self.register_buffer("feature_mean", torch.zeros(1, fused_dim, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(1, fused_dim, dtype=torch.float32))
        self.reg_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(fused_dim),
                    nn.Linear(fused_dim, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(int(out_dim))
            ]
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.last_attention = None

    def set_feature_standardization(self, mean, std) -> None:
        mean_t = torch.as_tensor(mean, dtype=self.feature_mean.dtype, device=self.feature_mean.device)
        std_t = torch.as_tensor(std, dtype=self.feature_std.dtype, device=self.feature_std.device)
        if mean_t.shape != self.feature_mean.shape or std_t.shape != self.feature_std.shape:
            raise ValueError("mean and std must have shape %s" % (tuple(self.feature_mean.shape),))
        self.feature_mean.copy_(mean_t)
        self.feature_std.copy_(std_t.clamp_min(1.0e-6))

    def _raw_kernel_features(self, x_raw: torch.Tensor) -> torch.Tensor:
        denom = (self.raw_highs - self.raw_lows).clamp_min(1.0e-6)
        x_unit01 = ((x_raw - self.raw_lows) / denom).clamp(0.0, 1.0)
        x_unit = (2.0 * x_unit01 - 1.0).clamp(-3.0, 3.0)
        lookup = {spec.name: i for i, spec in enumerate(PARAMETER_SPECS)}

        def p(name: str) -> torch.Tensor:
            return x_raw[:, lookup[name]]

        def pn(name: str) -> torch.Tensor:
            return x_unit01[:, lookup[name]]

        eps = 1.0e-6
        fin_force = p("fin_span") * p("fin_area_scale")
        lever = p("fin_position") * p("length_diameter_ratio")
        control_authority = p("ctrl_gain") * p("delta_gain") * p("max_deflection_deg") * fin_force
        moment_authority = control_authority * p("fin_position")
        slender_drag_proxy = (1.0 / p("length_diameter_ratio").clamp_min(eps)) * (1.0 + 0.08 * p("nose_haack_c"))
        nav_trace_proxy = p("pos_sigma").pow(2) + p("vel_sigma").pow(2) + p("att_sigma_deg").pow(2)
        safety_size_proxy = torch.log1p(p("q_max")) + p("alpha_max_deg") / 30.0 + p("p_trace_max") / 3000.0
        weight_sum = p("track_weight") + p("terminal_weight") + p("control_weight")
        terminal_bias = p("terminal_weight") / weight_sum.clamp_min(eps)
        control_bias = p("control_weight") / weight_sum.clamp_min(eps)
        pitch_rad = torch.deg2rad(p("pitch_delta_deg"))
        extras = torch.stack(
            [
                fin_force,
                lever,
                control_authority,
                moment_authority,
                slender_drag_proxy,
                nav_trace_proxy,
                safety_size_proxy,
                terminal_bias,
                control_bias,
                p("speed_scale") * torch.cos(pitch_rad),
                p("speed_scale") * torch.sin(pitch_rad),
                pn("length_diameter_ratio") * pn("fin_area_scale"),
                pn("fin_position") * pn("fin_span"),
                pn("ctrl_gain") * pn("delta_gain"),
                pn("max_deflection_deg") * pn("fin_area_scale"),
            ],
            dim=-1,
        )
        extras = torch.log1p(extras.abs()) * extras.sign()
        extras = extras.clamp(-8.0, 8.0)
        return torch.cat([x_unit, extras], dim=-1)

    def forward(self, x_raw: torch.Tensor, aero_node_features: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        _, pooled, _ = self.encoder(x_raw, aero_node_features=aero_node_features)
        fused = torch.cat([pooled, self.raw_skip(self._raw_kernel_features(x_raw))], dim=-1)
        fused = (fused - self.feature_mean) / self.feature_std.clamp_min(1.0e-6)
        self.last_attention = self.encoder.last_attention
        pred = torch.cat([head(fused) for head in self.reg_heads], dim=-1)
        return pred, self.cls_head(fused).squeeze(-1)


class GraphConditionalDesignGenerator(nn.Module):
    def __init__(self, in_dim: int, design_indices, hidden: int = 192, latent_dim: int = 12, layers: int = 3, value_dim: int = 8):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.design_indices = [int(i) for i in design_indices]
        self.encoder = PhysicalGraphEncoder(in_dim=in_dim, hidden=hidden, layers=layers, value_dim=value_dim)
        self.design_embed = nn.Sequential(nn.Linear(len(self.design_indices), hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.posterior = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden)
        self.param_embed = nn.Embedding(len(PARAMETER_SPECS), hidden)
        self.decoder = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer("design_param_ids", torch.tensor(self.design_indices, dtype=torch.long))
        self.register_buffer(
            "design_node_ids",
            torch.tensor([_node_index(PARAMETER_NODE[PARAMETER_SPECS[i].name]) for i in self.design_indices], dtype=torch.long),
        )

    def encode(
        self,
        cond_full: torch.Tensor,
        design_n: torch.Tensor,
        aero_node_features: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, pooled, _ = self.encoder(cond_full, aero_node_features=aero_node_features)
        h = self.posterior(torch.cat([pooled, self.design_embed(design_n)], dim=-1))
        return self.mu(h), self.logvar(h).clamp(-8.0, 6.0)

    def decode(self, cond_full: torch.Tensor, z: torch.Tensor, aero_node_features: torch.Tensor = None) -> torch.Tensor:
        node_h, _, _ = self.encoder(cond_full, aero_node_features=aero_node_features)
        node_h = node_h[:, self.design_node_ids, :]
        param_h = self.param_embed(self.design_param_ids).unsqueeze(0).expand(cond_full.shape[0], -1, -1)
        latent_h = self.latent_proj(z).unsqueeze(1)
        return self.decoder(node_h + param_h + latent_h).squeeze(-1)

    def forward(
        self,
        cond_full: torch.Tensor,
        design_n: torch.Tensor,
        aero_node_features: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(cond_full, design_n, aero_node_features=aero_node_features)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        return self.decode(cond_full, z, aero_node_features=aero_node_features), mu, logvar


def predictor_loss(
    pred: torch.Tensor,
    feasible_logit: torch.Tensor,
    target: torch.Tensor,
    feasible: torch.Tensor,
    target_weights: Tuple[float, ...] = (1.8, 2.4, 1.6, 1.2, 0.5, 1.0),
) -> torch.Tensor:
    weights = pred.new_tensor(target_weights).view(1, -1)
    weights = weights / weights.mean()
    elem = 0.75 * (pred - target).pow(2) + 0.25 * nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return (weights * elem).mean() + 0.12 * nn.functional.binary_cross_entropy_with_logits(
        feasible_logit, feasible.to(dtype=pred.dtype)
    )


def generator_loss(recon_design_n: torch.Tensor, design_n: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon = nn.functional.smooth_l1_loss(recon_design_n, design_n)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon + 1.0e-3 * kl


def project_design_to_bounds(design: torch.Tensor, design_indices) -> torch.Tensor:
    lows, highs = bounds()
    idx = torch.tensor([int(i) for i in design_indices], dtype=torch.long, device=design.device)
    low_t = torch.tensor(lows[idx.cpu().numpy()].astype(np.float32), device=design.device, dtype=design.dtype)
    high_t = torch.tensor(highs[idx.cpu().numpy()].astype(np.float32), device=design.device, dtype=design.dtype)
    return torch.minimum(torch.maximum(design, low_t), high_t)


def insert_design_values(cond_full: torch.Tensor, design: torch.Tensor, design_indices) -> torch.Tensor:
    x_full = cond_full.clone()
    idx = torch.tensor([int(i) for i in design_indices], dtype=torch.long, device=cond_full.device)
    x_full[:, idx] = design.to(device=cond_full.device, dtype=cond_full.dtype)
    return x_full


def decode_design_physical(
    generator: GraphConditionalDesignGenerator,
    cond_full: torch.Tensor,
    z: torch.Tensor,
    design_mean: torch.Tensor,
    design_std: torch.Tensor,
    design_indices,
    aero_node_features: torch.Tensor = None,
) -> torch.Tensor:
    design_n = generator.decode(cond_full, z, aero_node_features=aero_node_features)
    design = design_n * design_std.to(device=design_n.device, dtype=design_n.dtype) + design_mean.to(
        device=design_n.device, dtype=design_n.dtype
    )
    return project_design_to_bounds(design, design_indices)


def predictor_guided_refinement(
    predictor: nn.Module,
    cond_full: torch.Tensor,
    design_init: torch.Tensor,
    design_indices,
    steps: int = 20,
    step_size: float = 0.025,
    total_cost_index: int = 5,
    aero_node_feature_fn=None,
) -> Tuple[torch.Tensor, List[float]]:
    """Refine generated design variables while keeping context and constraints fixed."""
    device = next(predictor.parameters()).device
    cond = cond_full.to(device)
    design = design_init.detach().clone().to(device).requires_grad_(True)
    lows, highs = bounds()
    idx = torch.tensor([int(i) for i in design_indices], dtype=torch.long, device=device)
    low_t = torch.tensor(lows[idx.cpu().numpy()].astype(np.float32), device=device)
    high_t = torch.tensor(highs[idx.cpu().numpy()].astype(np.float32), device=device)
    optimizer = torch.optim.Adam([design], lr=float(step_size))
    history: List[float] = []

    for _ in range(int(steps)):
        bounded = torch.minimum(torch.maximum(design, low_t), high_t)
        x_full = cond.clone()
        x_full[:, idx] = bounded
        aero_node_features = None
        if aero_node_feature_fn is not None:
            with torch.no_grad():
                aero_np = aero_node_feature_fn(x_full.detach().cpu().numpy())
                aero_node_features = torch.as_tensor(aero_np, dtype=x_full.dtype, device=x_full.device)
        pred, logit = predictor(x_full, aero_node_features=aero_node_features)
        cost = pred[:, int(total_cost_index)].clamp_min(0.0)
        loss = torch.log1p(cost).mean() + 0.05 * nn.functional.softplus(-logit).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([design], 3.0)
        optimizer.step()
        with torch.no_grad():
            design.copy_(torch.minimum(torch.maximum(design, low_t), high_t))
        history.append(float(loss.detach().cpu()))

    return design.detach(), history
