# PSDG-Net Open-Source Method Package

This repository provides the implementation of the method proposed in the paper `PSDG-Net: typed physical-semantic graphs for generative multidisciplinary engineering design`, together with reproducible experiments on two heterogeneous engineering tasks. It contains only the main method, the robot electromechanical actuator experiment, the thermal-fluid-mechanical experiment, their data, and the corresponding tests.

The main aeronautical dataset, baseline models, ablation studies, training checkpoints, prediction outputs, and unrelated implementation branches are intentionally excluded.

## Contents

- `psdg_net/`: executable physical-semantic graph method.
  - `schema.py`: definitions of variables, unit dimensions, nodes, ports, equations, constraints, and relations.
  - `compiler.py`: variable normalization, node-contract construction, causal masks, relational evidence, and four-step full-covariance propagation.
  - `model.py`: relation-specific value operators, physics-contract attention, and the three-layer PSDG-Net multi-task predictor.
  - `generator.py`: graph-conditioned VAE with a 12-dimensional latent variable, reconstruction and KL losses, boundary projection, and 20-step predictor-guided optimization.
  - `active_learning.py`: acquisition interfaces for uncertainty, feasibility boundaries, improvement, diversity, physical residuals, and validation cost.
- `experiments/`: schemas, physics providers, Sobol data generation, and a unified trainer for the robot electromechanical and thermal-fluid-mechanical experiments.
- `data/robot_electromechanical/` and `data/thermal_fluid_mechanical/`: 32,768 candidate samples for each heterogeneous experiment, with fixed train/validation/test splits and schema metadata.
- `tests/`: tests for method forward propagation, covariance propagation, active learning, the generator, and the data contracts of both experiments.

The main method can also be applied to other tasks through any provider that conforms to the `PhysicsFeatureProvider` interface.

## Installation

From the repository root, run:

```powershell
python -m pip install -e .
```

Requirements are Python 3.8+, NumPy, SciPy, scikit-learn, and PyTorch. Both CPU and CUDA execution are supported; CUDA is recommended for large-scale training.

## Testing

```powershell
python -m unittest discover -s tests -v
```

## Running the Two Reproducible Experiments

The commands below use the datasets included in this repository and do not read code or data from outside the repository:

```powershell
python -m experiments.train --domain robot --device cpu
python -m experiments.train --domain thermal --device cpu
```

The default configuration follows the heterogeneous experiment settings used in the paper: hidden width 128, 4 attention heads, 3 relation-aware blocks, dropout 0.02, AdamW, learning rate `4e-4`, batch size 1024, up to 40 epochs, and a `log1p` target transformation. Training outputs are written to `runs/` and do not overwrite `data/`.

To regenerate the Sobol samples from code:

```python
from experiments.robot_actuator import build_schema, generate_dataset

schema = build_schema()
dataset = generate_dataset(schema, n_power=15, seed=2027)
```

## Method Usage Skeleton

```python
import torch
from psdg_net import PSDGNet

model = PSDGNet(schema, out_dim=number_of_targets, hidden=128, heads=4, layers=3)
physics = provider.evaluate(x_numpy).torch(torch.device("cpu"))
prediction, feasibility_logit, graph = model(torch.as_tensor(x_numpy), physics=physics)
```

The provider should return a `PhysicsBatch` containing `input_std`, `node_extras`, `equation_residuals`, `constraint_margins`, `relation_strength`, and `fidelity` for every sample. These fields provide evidence for the graph contracts; they are not target predictions, target-residual shortcuts, or low-fidelity label bypasses.

## Correspondence with the Paper Method

Scalar variables are normalized by their bounds and clipped to `[-4, 4]`. Local uncertainty is propagated through the full causal graph using a four-step fixed-point recurrence. Before attention is computed, the schema-based causal mask removes undeclared information pathways; the model then integrates coupling strength, equation residuals, fidelity, and propagated uncertainty. The regression head directly predicts each target, while a separate logistic head predicts feasibility. The generator keeps context and safety variables fixed, decodes only design variables, and projects the generated output back to the physical bounds.

## License

See [LICENSE](LICENSE).
