"""Train PSDG-Net on either of the two packaged heterogeneous experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from experiments import robot_actuator, thermal_fluid_mechanical
from experiments.common import ExperimentDataset, TrainingConfig, run_experiment


ROOT = Path(__file__).resolve().parents[1]
PACKAGED_DATA = ROOT / "data"


def load_dataset(path: Path) -> ExperimentDataset:
    """Load one of the immutable packaged experiment datasets."""
    with np.load(path / "dataset.npz", allow_pickle=False) as data:
        target_names = tuple(str(value) for value in data["target_names"].tolist())
        dataset = ExperimentDataset(
            x=data["x"],
            y=data["y"],
            feasible=data["feasible"],
            target_names=target_names,
            train_idx=data["train_idx"],
            val_idx=data["val_idx"],
            test_idx=data["test_idx"],
            metadata={},
        )
    metadata_path = path / "dataset_metadata.json"
    if metadata_path.exists():
        dataset.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return dataset


def run_domain(domain: str, data_root: Path, output_root: Path, config: TrainingConfig) -> Dict[str, object]:
    domains: Dict[str, tuple] = {
        "robot": ("robot_electromechanical", robot_actuator.build_schema, robot_actuator.RobotPhysicsProvider),
        "thermal": ("thermal_fluid_mechanical", thermal_fluid_mechanical.build_schema, thermal_fluid_mechanical.ThermalFluidPhysicsProvider),
    }
    name, schema_factory, provider_factory = domains[domain]
    schema = schema_factory()
    dataset = load_dataset(data_root / name)
    return run_experiment(name, schema, provider_factory(schema), dataset, output_root, config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the paper's PSDG-Net predictor on a packaged domain.")
    parser.add_argument("--domain", choices=("robot", "thermal"), required=True)
    parser.add_argument("--data-root", type=Path, default=PACKAGED_DATA)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = TrainingConfig(
        seed=args.seed,
        hidden=128,
        heads=4,
        layers=3,
        dropout=0.02,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=4.0e-4,
        device=args.device,
    )
    run_domain(args.domain, args.data_root, args.output_root, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
