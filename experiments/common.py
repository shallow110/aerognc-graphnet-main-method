from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from psdg_net.compiler import CompiledPhysicalGraph
from psdg_net.model import ExecutablePhysicalGraphPredictor
from psdg_net.providers import PhysicsBatch, PhysicsFeatureProvider
from psdg_net.schema import GraphSchema


@dataclass
class ExperimentDataset:
    x: np.ndarray
    y: np.ndarray
    feasible: np.ndarray
    target_names: Sequence[str]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    metadata: Dict[str, object]

    def validate(self, variable_count: int) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)
        self.feasible = np.asarray(self.feasible, dtype=np.float32).reshape(-1)
        self.train_idx = np.asarray(self.train_idx, dtype=np.int64)
        self.val_idx = np.asarray(self.val_idx, dtype=np.int64)
        self.test_idx = np.asarray(self.test_idx, dtype=np.int64)
        if self.x.ndim != 2 or self.x.shape[1] != variable_count:
            raise ValueError("dataset variable count does not match schema")
        if self.y.ndim != 2 or self.y.shape[0] != self.x.shape[0]:
            raise ValueError("invalid regression target shape")
        if self.feasible.shape[0] != self.x.shape[0]:
            raise ValueError("invalid feasibility target shape")
        if len(self.target_names) != self.y.shape[1]:
            raise ValueError("target_names length mismatch")
        all_split = np.concatenate([self.train_idx, self.val_idx, self.test_idx])
        if len(np.unique(all_split)) != len(all_split):
            raise ValueError("train/validation/test splits overlap")


@dataclass
class TrainingConfig:
    seed: int = 2026
    hidden: int = 128
    heads: int = 4
    layers: int = 3
    dropout: float = 0.02
    batch_size: int = 1024
    epochs: int = 40
    learning_rate: float = 4.0e-4
    weight_decay: float = 1.0e-4
    patience: int = 10
    feasibility_weight: float = 0.10
    target_transform: str = "log1p"
    device: str = "auto"
    num_workers: int = 0


def deterministic_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fixed_random_split(n: int, seed: int, train_fraction: float = 0.72, val_fraction: float = 0.08):
    rng = np.random.default_rng(seed)
    order = rng.permutation(int(n))
    n_train = int(train_fraction * n)
    n_val = int(val_fraction * n)
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.dtype.str.encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _physics_to_cpu(batch: PhysicsBatch) -> Dict[str, torch.Tensor]:
    return {
        "input_std": torch.as_tensor(batch.input_std, dtype=torch.float32),
        "node_extras": torch.as_tensor(batch.node_extras, dtype=torch.float32),
        "equation_residuals": torch.as_tensor(batch.equation_residuals, dtype=torch.float32),
        "constraint_margins": torch.as_tensor(batch.constraint_margins, dtype=torch.float32),
        "relation_strength": torch.as_tensor(batch.relation_strength, dtype=torch.float32),
        "fidelity": torch.as_tensor(batch.fidelity, dtype=torch.float32),
    }


def _slice_physics(physics: Dict[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: values[indices].to(device, non_blocking=True) for name, values in physics.items()}


def _slice_graph(graph: CompiledPhysicalGraph, indices: torch.Tensor, device: torch.device) -> CompiledPhysicalGraph:
    batched = {
        "node_features": graph.node_features[indices].to(device, non_blocking=True),
        "relation_strength": graph.relation_strength[indices].to(device, non_blocking=True),
        "relation_residual": graph.relation_residual[indices].to(device, non_blocking=True),
        "relation_fidelity": graph.relation_fidelity[indices].to(device, non_blocking=True),
        "relation_uncertainty": graph.relation_uncertainty[indices].to(device, non_blocking=True),
        "covariance": graph.covariance.to(device, non_blocking=True),
        "node_std": graph.node_std.to(device, non_blocking=True),
        "correlation": graph.correlation.to(device, non_blocking=True),
    }
    static = {
        "node_kind_ids": graph.node_kind_ids.to(device, non_blocking=True),
        "discipline_ids": graph.discipline_ids.to(device, non_blocking=True),
        "node_dimensions": graph.node_dimensions.to(device, non_blocking=True),
        "edge_source": graph.edge_source.to(device, non_blocking=True),
        "edge_target": graph.edge_target.to(device, non_blocking=True),
        "edge_type": graph.edge_type.to(device, non_blocking=True),
        "causal_mask": graph.causal_mask.to(device, non_blocking=True),
    }
    return CompiledPhysicalGraph(**batched, **static)


def _compile_graph_cache(
    model: ExecutablePhysicalGraphPredictor,
    x: torch.Tensor,
    physics: Dict[str, torch.Tensor],
    device: torch.device,
    chunk_size: int,
) -> CompiledPhysicalGraph:
    """Compile deterministic graph contracts once, then reuse them every epoch."""
    chunks = []
    model.compiler.eval()
    with torch.no_grad():
        for start in range(0, len(x), chunk_size):
            indices = torch.arange(start, min(start + chunk_size, len(x)), dtype=torch.long)
            xb = x[indices].to(device, non_blocking=True)
            compiled = model.compiler(xb, **_slice_physics(physics, indices, device))
            chunks.append(
                CompiledPhysicalGraph(
                    node_features=compiled.node_features.detach(),
                    node_kind_ids=compiled.node_kind_ids.detach(),
                    discipline_ids=compiled.discipline_ids.detach(),
                    node_dimensions=compiled.node_dimensions.detach(),
                    edge_source=compiled.edge_source.detach(),
                    edge_target=compiled.edge_target.detach(),
                    edge_type=compiled.edge_type.detach(),
                    relation_strength=compiled.relation_strength.detach(),
                    relation_residual=compiled.relation_residual.detach(),
                    relation_fidelity=compiled.relation_fidelity.detach(),
                    relation_uncertainty=compiled.relation_uncertainty.detach(),
                    causal_mask=compiled.causal_mask.detach(),
                    covariance=torch.empty(0),
                    node_std=torch.empty(0),
                    correlation=torch.empty(0),
                )
            )
    first = chunks[0]
    return CompiledPhysicalGraph(
        node_features=torch.cat([item.node_features for item in chunks], dim=0),
        node_kind_ids=first.node_kind_ids,
        discipline_ids=first.discipline_ids,
        node_dimensions=first.node_dimensions,
        edge_source=first.edge_source,
        edge_target=first.edge_target,
        edge_type=first.edge_type,
        relation_strength=torch.cat([item.relation_strength for item in chunks], dim=0),
        relation_residual=torch.cat([item.relation_residual for item in chunks], dim=0),
        relation_fidelity=torch.cat([item.relation_fidelity for item in chunks], dim=0),
        relation_uncertainty=torch.cat([item.relation_uncertainty for item in chunks], dim=0),
        causal_mask=first.causal_mask,
        covariance=torch.empty(0),
        node_std=torch.empty(0),
        correlation=torch.empty(0),
    )


def _transform_targets(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return y.astype(np.float32)
    if transform == "log1p":
        if np.any(y < -1.0e-8):
            raise ValueError("log1p transform requires non-negative targets")
        return np.log1p(np.maximum(y, 0.0)).astype(np.float32)
    raise ValueError(f"unknown target transform {transform}")


def _inverse_targets(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return y
    if transform == "log1p":
        return np.maximum(np.expm1(y), 0.0)
    raise ValueError(transform)


def _evaluate(
    model: ExecutablePhysicalGraphPredictor,
    x: torch.Tensor,
    graph_cache: CompiledPhysicalGraph,
    indices: np.ndarray,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    transform: str,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx_np = indices[start : start + batch_size]
            idx = torch.as_tensor(idx_np, dtype=torch.long)
            xb = x[idx].to(device, non_blocking=True)
            pred, logit, _ = model(xb, graph=_slice_graph(graph_cache, idx, device))
            predictions.append(pred.cpu().numpy())
            probabilities.append(torch.sigmoid(logit).cpu().numpy())
    pred_n = np.concatenate(predictions, axis=0)
    pred_transformed = pred_n * y_std + y_mean
    return _inverse_targets(pred_transformed, transform).astype(np.float32), np.concatenate(probabilities).astype(np.float32)


def regression_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feasible: np.ndarray,
    probability: np.ndarray,
    target_names: Sequence[str],
) -> Dict[str, object]:
    per_target = {}
    r2_values = []
    for index, name in enumerate(target_names):
        r2 = float(r2_score(y_true[:, index], y_pred[:, index]))
        mae = float(np.mean(np.abs(y_true[:, index] - y_pred[:, index])))
        rmse = float(np.sqrt(np.mean((y_true[:, index] - y_pred[:, index]) ** 2)))
        per_target[name] = {"r2": r2, "mae": mae, "rmse": rmse}
        r2_values.append(r2)
    try:
        auc = float(roc_auc_score(feasible, probability))
        ap = float(average_precision_score(feasible, probability))
    except ValueError:
        auc = None
        ap = None
    calibration = float(np.mean((probability - feasible) ** 2))
    # Report both a proper scoring rule and a bin-based calibration error.
    # ECE is computed only from held-out probabilities and binary labels.
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    feasible = np.asarray(feasible, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probability >= lower) & (probability < upper)
        if upper == 1.0:
            mask |= probability == upper
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(probability[mask])) - float(np.mean(feasible[mask])))
    return {
        "per_target": per_target,
        "min_r2": float(np.min(r2_values)),
        "mean_r2": float(np.mean(r2_values)),
        "feasibility_auc": auc,
        "feasibility_ap": ap,
        "feasibility_brier": calibration,
        "feasibility_ece": float(ece),
    }


def run_experiment(
    name: str,
    schema: GraphSchema,
    provider: PhysicsFeatureProvider,
    dataset: ExperimentDataset,
    output_root: Path,
    config: TrainingConfig,
) -> Dict[str, object]:
    dataset.validate(len(schema.variables))
    deterministic_seed(config.seed)
    device = choose_device(config.device)
    output_dir = Path(output_root) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "dataset.npz",
        x=dataset.x,
        y=dataset.y,
        feasible=dataset.feasible,
        train_idx=dataset.train_idx,
        val_idx=dataset.val_idx,
        test_idx=dataset.test_idx,
        target_names=np.asarray(dataset.target_names),
    )
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(dataset.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    started = time.time()
    physics_batch = provider.evaluate(dataset.x)
    x_cpu = torch.as_tensor(dataset.x, dtype=torch.float32)
    y_transformed = _transform_targets(dataset.y, config.target_transform)
    y_mean = y_transformed[dataset.train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    y_std = np.maximum(y_transformed[dataset.train_idx].std(axis=0, keepdims=True), 1.0e-6).astype(np.float32)
    y_normalized = ((y_transformed - y_mean) / y_std).astype(np.float32)
    physics = _physics_to_cpu(physics_batch)

    model = ExecutablePhysicalGraphPredictor(
        schema,
        dataset.y.shape[1],
        hidden=config.hidden,
        heads=config.heads,
        layers=config.layers,
        dropout=config.dropout,
        extra_dim=int(physics_batch.node_extras.shape[-1]),
    ).to(device)
    graph_cache = _compile_graph_cache(
        model,
        x_cpu,
        physics,
        device,
        chunk_size=max(config.batch_size * 8, 8192),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1.0e-6,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
        eta_min=max(config.learning_rate * 0.1, 1.0e-6),
    )
    train_dataset = TensorDataset(
        torch.as_tensor(dataset.train_idx, dtype=torch.long),
        torch.as_tensor(y_normalized[dataset.train_idx], dtype=torch.float32),
        torch.as_tensor(dataset.feasible[dataset.train_idx], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    target_weight = torch.ones((1, dataset.y.shape[1]), dtype=torch.float32, device=device)
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    nonfinite_gradient_batches = 0
    nonfinite_loss_batches = 0
    history = []
    val_subset = dataset.val_idx[: min(len(dataset.val_idx), 12000)]

    print(f"[{name}] device={device} train={len(dataset.train_idx)} val={len(dataset.val_idx)} test={len(dataset.test_idx)}")
    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        count = 0
        valid_train_batches = 0
        stop_training = False
        for idx, target, feasible in train_loader:
            xb = x_cpu[idx].to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            feasible = feasible.to(device, non_blocking=True)
            pred, logit, _ = model(xb, graph=_slice_graph(graph_cache, idx, device))
            squared = (pred - target).pow(2)
            robust = nn.functional.smooth_l1_loss(pred, target, beta=0.5, reduction="none")
            regression_loss = (target_weight * (0.75 * squared + 0.25 * robust)).mean()
            classification_loss = nn.functional.binary_cross_entropy_with_logits(logit, feasible)
            loss = regression_loss + config.feasibility_weight * classification_loss
            if not torch.isfinite(loss):
                nonfinite_loss_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            batch_gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if any(not torch.isfinite(gradient).all() for gradient in batch_gradients):
                nonfinite_gradient_batches += 1
                for gradient in batch_gradients:
                    torch.nan_to_num_(gradient, nan=0.0, posinf=0.0, neginf=0.0)
            nn.utils.clip_grad_value_(model.parameters(), 1.0)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                stop_training = True
                break
            running += float(loss.detach().cpu()) * idx.numel()
            count += idx.numel()
            valid_train_batches += 1

        if stop_training or valid_train_batches == 0:
            break

        model.eval()
        val_sum = 0.0
        val_count = 0
        valid_val_batches = 0
        with torch.no_grad():
            for start in range(0, len(val_subset), config.batch_size):
                idx_np = val_subset[start : start + config.batch_size]
                idx = torch.as_tensor(idx_np, dtype=torch.long)
                xb = x_cpu[idx].to(device, non_blocking=True)
                target = torch.as_tensor(y_normalized[idx_np], dtype=torch.float32, device=device)
                feasible = torch.as_tensor(dataset.feasible[idx_np], dtype=torch.float32, device=device)
                pred, logit, _ = model(xb, graph=_slice_graph(graph_cache, idx, device))
                squared = (pred - target).pow(2)
                robust = nn.functional.smooth_l1_loss(pred, target, beta=0.5, reduction="none")
                val_loss = (target_weight * (0.75 * squared + 0.25 * robust)).mean() + config.feasibility_weight * nn.functional.binary_cross_entropy_with_logits(logit, feasible)
                if not torch.isfinite(val_loss):
                    nonfinite_loss_batches += 1
                    continue
                val_sum += float(val_loss.cpu()) * len(idx_np)
                val_count += len(idx_np)
                valid_val_batches += 1
        if stop_training or valid_val_batches == 0:
            break
        train_loss = running / max(count, 1)
        val_loss_value = val_sum / max(val_count, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss_value})
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"[{name}] epoch={epoch + 1:03d} train={train_loss:.6f} val={val_loss_value:.6f}", flush=True)
        if val_loss_value + 1.0e-6 < best_loss:
            best_loss = val_loss_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    prediction, probability = _evaluate(
        model,
        x_cpu,
        graph_cache,
        dataset.test_idx,
        device,
        y_mean,
        y_std,
        config.target_transform,
        config.batch_size,
    )
    metrics = regression_classification_metrics(
        dataset.y[dataset.test_idx],
        prediction,
        dataset.feasible[dataset.test_idx],
        probability,
        dataset.target_names,
    )
    metrics.update(
        {
            "experiment": name,
            "schema": schema.name,
            "model": "PSDG-Net",
            "seed": config.seed,
            "best_val_loss": best_loss,
            "best_epoch": int(min(history, key=lambda row: row["val_loss"])["epoch"]),
            "nonfinite_gradient_batches": int(nonfinite_gradient_batches),
            "nonfinite_loss_batches": int(nonfinite_loss_batches),
            "elapsed_seconds": time.time() - started,
            "dataset_hash": _array_hash(dataset.x, dataset.y, dataset.feasible),
            "split_hash": _array_hash(dataset.train_idx, dataset.val_idx, dataset.test_idx),
            "n_train": int(len(dataset.train_idx)),
            "n_val": int(len(dataset.val_idx)),
            "n_test": int(len(dataset.test_idx)),
            "target_names": list(dataset.target_names),
            "dataset_metadata": dataset.metadata,
            "training_config": asdict(config),
            "target_transform": config.target_transform,
        }
    )
    torch.save(
        {
            "model_state": best_state,
            "config": asdict(config),
            "schema": schema.as_dict(),
            "y_mean": y_mean,
            "y_std": y_std,
            "target_transform": config.target_transform,
        },
        output_dir / "model.pt",
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        test_idx=dataset.test_idx,
        y_true=dataset.y[dataset.test_idx],
        y_pred=prediction,
        feasible=dataset.feasible[dataset.test_idx],
        feasibility_probability=probability,
    )
    (output_dir / "schema.json").write_text(json.dumps(schema.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] min_r2={metrics['min_r2']:.4f} mean_r2={metrics['mean_r2']:.4f} auc={metrics['feasibility_auc']}")
    return metrics
