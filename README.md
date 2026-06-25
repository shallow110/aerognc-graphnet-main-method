# AeroGNC-GraphNet Main Method Package

This folder contains a self-contained implementation of the AeroGNC-GraphNet main graph-attention method.

Included:

- 19-parameter schema and bounds.
- Nine-node physical-semantic graph builder.
- Physics-biased graph attention encoder.
- Multi-task graph predictor.
- Predictor training loss from the paper.
- Graph-conditioned design generator.
- Generator reconstruction-plus-KL loss from the paper.
- Predictor-guided refinement.
- Active-learning acquisition score.
- A smoke test that checks the extracted code can run.
- A standalone demo with synthetic data fixtures, so the package runs without project datasets or project code.

Package layout:

- `aerognc_graphnet/dataset/`: parameter schema, active-learning acquisition and self-contained synthetic fixtures.
- `aerognc_graphnet/learning/`: physical graph builder, differentiable graph-attention predictor, conditional generator and guidance functions.
- `aerognc_graphnet/examples/`: standalone runnable demo entry points.
- Top-level modules keep the same implementations available through shorter imports.

## Run

```powershell
cd open_source_main_method
python tests/test_smoke.py
python -m aerognc_graphnet.demo
```

## Notes Before Public Release

- Choose and add a real open-source license before publishing.
- For scientific reproduction, replace the synthetic demo aerodynamic fixtures with the intended public aerodynamic coefficient data or solver outputs.
