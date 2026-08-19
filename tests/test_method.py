from __future__ import annotations

import unittest

import numpy as np
import torch

from psdg_net.active_learning import AcquisitionInputs, PhysicsAwareAcquisition
from psdg_net.generator import GraphConditionalDesignGenerator, normalize_variables
from experiments.robot_actuator import RobotPhysicsProvider, build_schema, generate_dataset
from psdg_net.model import ExecutablePhysicalGraphPredictor


class ExecutableGraphTests(unittest.TestCase):
    def setUp(self):
        self.schema = build_schema()
        self.dataset = generate_dataset(self.schema, n_power=6)
        self.provider = RobotPhysicsProvider(self.schema)

    def test_schema_is_executable_and_typed(self):
        self.assertGreater(len(self.schema.ports), 0)
        self.assertGreater(len(self.schema.equations), 0)
        self.assertGreater(len(self.schema.constraints), 0)
        self.assertTrue(all(variable.unit for variable in self.schema.variables))

    def test_full_uncertainty_reaches_distant_nodes(self):
        x = torch.as_tensor(self.dataset.x[:4])
        physics = self.provider.evaluate(self.dataset.x[:4]).torch(torch.device("cpu"))
        model = ExecutablePhysicalGraphPredictor(self.schema, self.dataset.y.shape[1], hidden=64, heads=4, layers=2)
        _, _, graph = model(x, physics)
        self.assertEqual(tuple(graph.covariance.shape), (4, len(self.schema.nodes), len(self.schema.nodes)))
        objective = self.schema.node_index["Objective"]
        self.assertTrue(torch.all(graph.node_std[:, objective] > 0.0))
        off_diagonal = graph.covariance - torch.diag_embed(torch.diagonal(graph.covariance, dim1=-2, dim2=-1))
        self.assertGreater(float(off_diagonal.abs().sum()), 0.0)

    def test_model_forward_shapes(self):
        x_np = self.dataset.x[:8]
        physics = self.provider.evaluate(x_np).torch(torch.device("cpu"))
        model = ExecutablePhysicalGraphPredictor(self.schema, self.dataset.y.shape[1], hidden=64, heads=4, layers=2)
        prediction, logit, graph = model(torch.as_tensor(x_np), physics)
        self.assertEqual(tuple(prediction.shape), (8, self.dataset.y.shape[1]))
        self.assertEqual(tuple(logit.shape), (8,))
        self.assertEqual(graph.relation_strength.shape[1], len(self.schema.relations))

    def test_active_learning_interface(self):
        rng = np.random.default_rng(3)
        inputs = AcquisitionInputs(
            predictive_std=rng.random(32),
            feasibility_probability=rng.random(32),
            predicted_objective=rng.random(32),
            best_observed=0.2,
            embeddings=rng.normal(size=(32, 5)),
            constraint_margin=rng.normal(size=(32, 3)),
            physics_residual=rng.random((32, 4)),
            verification_cost=1.0 + rng.random(32),
        )
        selected, components = PhysicsAwareAcquisition().select(inputs, 5)
        self.assertEqual(selected.shape, (5,))
        self.assertEqual(set(components), {"uncertainty", "boundary", "improvement", "diversity", "physics_residual", "verification_cost"})

    def test_conditional_generator_shapes(self):
        x_np = self.dataset.x[:4]
        x = torch.as_tensor(x_np)
        physics = self.provider.evaluate(x_np).torch(torch.device("cpu"))
        design_idx = [i for i, variable in enumerate(self.schema.variables) if variable.role == "design"]
        generator = GraphConditionalDesignGenerator(self.schema, design_idx, hidden=32, heads=4, layers=1, latent_dim=4)
        lows = generator.compiler.lows[design_idx]
        highs = generator.compiler.highs[design_idx]
        design_n = normalize_variables(x[:, design_idx], lows, highs)
        reconstruction, mu, logvar = generator(x, design_n, physics=physics)
        self.assertEqual(tuple(reconstruction.shape), (4, len(design_idx)))
        self.assertEqual(tuple(mu.shape), (4, 4))
        self.assertEqual(tuple(logvar.shape), (4, 4))


if __name__ == "__main__":
    unittest.main()
