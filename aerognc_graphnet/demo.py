from __future__ import annotations

import argparse
import json
from typing import Dict

import numpy as np
import torch

from .active_learning import select_top_k
from .core import (
    GraphConditionalDesignGenerator,
    PhysicalGraphPredictor,
    decode_design_physical,
    generator_loss,
    insert_design_values,
    predictor_guided_refinement,
    predictor_loss,
)
from .schema import LABEL_NAMES, PARAMETER_SPECS, condition_full_from_matrix, index_groups
from .synthetic import synthetic_aero_node_features, synthetic_dataset


def _standardize(a: np.ndarray):
    mean = a.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(a.std(axis=0, keepdims=True), 1.0e-6).astype(np.float32)
    return ((a - mean) / std).astype(np.float32), mean, std


def run_demo(seed: int = 3, n: int = 96, epochs: int = 2) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, aero, y = synthetic_dataset(n=n, seed=seed)
    cond_idx, design_idx = index_groups()
    device = torch.device("cpu")

    x_t = torch.from_numpy(x).to(device)
    aero_t = torch.from_numpy(aero).to(device)
    y_reg_t = torch.from_numpy(y[:, :-1]).to(device)
    y_feas_t = torch.from_numpy(y[:, -1]).to(device)

    predictor = PhysicalGraphPredictor(len(PARAMETER_SPECS), hidden=48, out_dim=6, layers=1).to(device)
    opt = torch.optim.AdamW(predictor.parameters(), lr=8.0e-4, weight_decay=1.0e-4)
    predictor_losses = []
    for _ in range(int(epochs)):
        pred, logit = predictor(x_t, aero_node_features=aero_t)
        loss = predictor_loss(pred, logit, y_reg_t, y_feas_t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 2.0)
        opt.step()
        predictor_losses.append(float(loss.detach().cpu()))

    keep = y[:, LABEL_NAMES.index("total_cost")] <= np.quantile(y[:, LABEL_NAMES.index("total_cost")], 0.35)
    cond_full = condition_full_from_matrix(x[keep], cond_idx)
    design = x[keep][:, design_idx]
    design_n, design_mean, design_std = _standardize(design)
    cond_t = torch.from_numpy(cond_full).to(device)
    design_n_t = torch.from_numpy(design_n).to(device)
    aero_cond_t = torch.from_numpy(synthetic_aero_node_features(cond_full)).to(device)

    generator = GraphConditionalDesignGenerator(len(PARAMETER_SPECS), design_idx, hidden=48, latent_dim=12, layers=1).to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    generator_losses = []
    for _ in range(int(epochs)):
        recon, mu, logvar = generator(cond_t, design_n_t, aero_node_features=aero_cond_t)
        loss_g = generator_loss(recon, design_n_t, mu, logvar)
        opt_g.zero_grad()
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_g.step()
        generator_losses.append(float(loss_g.detach().cpu()))

    sample_n = min(8, cond_t.shape[0])
    z = torch.randn(sample_n, 12, device=device)
    generated_design = decode_design_physical(
        generator,
        cond_t[:sample_n],
        z,
        torch.from_numpy(design_mean).to(device),
        torch.from_numpy(design_std).to(device),
        design_idx,
        aero_node_features=aero_cond_t[:sample_n],
    )
    guided_design, guidance_history = predictor_guided_refinement(
        predictor,
        cond_t[:sample_n],
        generated_design,
        design_idx,
        steps=3,
        step_size=0.005,
        aero_node_feature_fn=synthetic_aero_node_features,
    )
    guided_full = insert_design_values(cond_t[:sample_n], guided_design, design_idx)
    with torch.no_grad():
        guided_aero = torch.from_numpy(synthetic_aero_node_features(guided_full.cpu().numpy())).to(device)
        guided_pred, guided_logit = predictor(guided_full, aero_node_features=guided_aero)

    cost = guided_pred[:, LABEL_NAMES.index("total_cost")].detach().cpu().numpy()
    prob = torch.sigmoid(guided_logit).detach().cpu().numpy()
    metrics = {
        "epistemic_uncertainty": np.zeros(sample_n, dtype=np.float32),
        "constraint_boundary": 1.0 - np.abs(prob - 0.5) * 2.0,
        "predicted_improvement": -cost,
        "diversity_distance": np.linspace(0.0, 1.0, sample_n, dtype=np.float32),
        "high_gradient_region": np.abs(cost - cost.mean()).astype(np.float32),
    }
    selected, scores = select_top_k(metrics, k=min(3, sample_n))
    return {
        "n_parameters": len(PARAMETER_SPECS),
        "dataset_size": int(n),
        "predictor_loss_last": predictor_losses[-1],
        "generator_loss_last": generator_losses[-1],
        "generated_candidates": int(sample_n),
        "guidance_steps": len(guidance_history),
        "selected_indices": selected.tolist(),
        "score_mean": float(np.mean(scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a self-contained AeroGNC-GraphNet main-method demo.")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_demo(seed=args.seed, n=args.n, epochs=args.epochs), indent=2))


if __name__ == "__main__":
    main()
