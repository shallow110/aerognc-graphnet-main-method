from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aerognc_graphnet import (  # noqa: E402
    PARAMETER_SPECS,
    GraphConditionalDesignGenerator,
    PhysicalDesignGraphBuilder,
    PhysicalGraphPredictor,
    bounds,
    condition_full_from_matrix,
    decode_design_physical,
    design_graph_metadata,
    generator_loss,
    index_groups,
    predictor_loss,
    predictor_guided_refinement,
    select_top_k,
    synthetic_aero_node_features,
)
from aerognc_graphnet.learning.physical_design_graph import PhysicalGraphPredictor as RoutedPredictor  # noqa: E402
from aerognc_graphnet.dataset.schema import PARAMETER_SPECS as RoutedParameterSpecs  # noqa: E402


def sample_candidates(n: int = 4) -> np.ndarray:
    lows, highs = bounds()
    rng = np.random.default_rng(7)
    return rng.uniform(lows, highs, size=(n, len(PARAMETER_SPECS))).astype(np.float32)


def main() -> None:
    torch.manual_seed(7)
    x_np = sample_candidates(4)
    x = torch.from_numpy(x_np)
    aero = torch.from_numpy(synthetic_aero_node_features(x_np))

    builder = PhysicalDesignGraphBuilder()
    try:
        builder(x)
        raise AssertionError("builder should require explicit aero_node_features")
    except ValueError:
        pass
    graph = builder(x, aero_node_features=aero)
    assert graph.shape == (4, 9, 8)
    metadata = design_graph_metadata()
    assert metadata["parameter_binding"]["delta_gain"] == "ActuatorNode"
    assert metadata["parameter_binding"]["max_deflection_deg"] == "ActuatorNode"

    predictor = PhysicalGraphPredictor(len(PARAMETER_SPECS), hidden=32, out_dim=6, layers=1)
    routed_predictor = RoutedPredictor(len(RoutedParameterSpecs), hidden=32, out_dim=6, layers=1)
    assert isinstance(routed_predictor, PhysicalGraphPredictor)
    y_hat, feas_logit = predictor(x, aero_node_features=aero)
    assert y_hat.shape == (4, 6)
    assert feas_logit.shape == (4,)
    assert torch.isfinite(y_hat).all()
    assert torch.isfinite(feas_logit).all()
    loss = predictor_loss(y_hat, feas_logit, torch.zeros_like(y_hat), torch.zeros_like(feas_logit))
    assert torch.isfinite(loss)

    cond_idx, design_idx = index_groups()
    cond_full = torch.from_numpy(condition_full_from_matrix(x_np, cond_idx))
    design_n = torch.zeros((4, len(design_idx)), dtype=torch.float32)
    generator = GraphConditionalDesignGenerator(len(PARAMETER_SPECS), design_idx, hidden=32, latent_dim=4, layers=1)
    cond_aero = torch.from_numpy(synthetic_aero_node_features(cond_full.numpy()))
    recon, mu, logvar = generator(cond_full, design_n, aero_node_features=cond_aero)
    assert recon.shape == (4, len(design_idx))
    assert mu.shape == (4, 4)
    assert logvar.shape == (4, 4)
    gen_loss = generator_loss(recon, design_n, mu, logvar)
    assert torch.isfinite(gen_loss)

    generated = decode_design_physical(
        generator,
        cond_full,
        torch.zeros((4, 4), dtype=torch.float32),
        torch.zeros((1, len(design_idx)), dtype=torch.float32),
        torch.ones((1, len(design_idx)), dtype=torch.float32),
        design_idx,
        aero_node_features=cond_aero,
    )
    assert generated.shape == (4, len(design_idx))

    refined, history = predictor_guided_refinement(
        predictor,
        cond_full,
        x[:, design_idx],
        design_idx,
        steps=2,
        step_size=0.001,
        aero_node_feature_fn=synthetic_aero_node_features,
    )
    assert refined.shape == (4, len(design_idx))
    assert len(history) == 2

    selected, scores = select_top_k(
        {
            "epistemic_uncertainty": np.array([0.1, 0.9, 0.2], dtype=np.float32),
            "constraint_boundary": np.array([0.3, 0.4, 0.8], dtype=np.float32),
            "predicted_improvement": np.array([0.5, 0.2, 0.7], dtype=np.float32),
            "diversity_distance": np.array([0.2, 0.6, 0.4], dtype=np.float32),
            "high_gradient_region": np.array([0.1, 0.5, 0.9], dtype=np.float32),
        },
        k=2,
    )
    assert selected.shape == (2,)
    assert scores.shape == (3,)
    print("smoke test passed")


if __name__ == "__main__":
    main()
