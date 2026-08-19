import unittest
from pathlib import Path

import numpy as np
import torch

from psdg_net.compiler import ExecutableGraphCompiler
from experiments import robot_actuator, thermal_fluid_mechanical


class DomainContractTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_both_domains_declare_executable_contracts(self):
        for schema in (robot_actuator.build_schema(), thermal_fluid_mechanical.build_schema()):
            self.assertTrue(schema.ports)
            self.assertTrue(schema.equations)
            self.assertTrue(schema.constraints)
            self.assertTrue(schema.relations)
            self.assertTrue(all(variable.unit and len(variable.dimension.vector()) == 7 for variable in schema.variables))
            self.assertTrue(all(relation.source != relation.target for relation in schema.relations))

    def test_new_domain_uncertainty_reaches_objectives(self):
        cases = (
            (robot_actuator, robot_actuator.RobotPhysicsProvider),
            (thermal_fluid_mechanical, thermal_fluid_mechanical.ThermalFluidPhysicsProvider),
        )
        for module, provider_type in cases:
            schema = module.build_schema()
            dataset = module.generate_dataset(schema, n_power=6)
            physics = provider_type(schema).evaluate(dataset.x[:16]).torch(torch.device("cpu"))
            graph = ExecutableGraphCompiler(schema)(torch.as_tensor(dataset.x[:16]), **physics)
            objective = [index for index, node in enumerate(schema.nodes) if node.kind == "objective"]
            self.assertTrue(objective)
            self.assertGreater(float(graph.node_std[:, objective].mean()), 0.0)
            diagonal = torch.diag_embed(torch.diagonal(graph.covariance, dim1=-2, dim2=-1))
            self.assertGreater(float((graph.covariance - diagonal).abs().max()), 0.0)

    def test_packaged_datasets_match_domain_schemas(self):
        for name, module in (
            ("robot_electromechanical", robot_actuator),
            ("thermal_fluid_mechanical", thermal_fluid_mechanical),
        ):
            dataset_path = self.ROOT / "data" / name / "dataset.npz"
            self.assertTrue(dataset_path.exists())
            with np.load(dataset_path, allow_pickle=False) as data:
                self.assertEqual(data["x"].shape[1], len(module.build_schema().variables))
                self.assertEqual(data["y"].shape[0], data["x"].shape[0])
                self.assertEqual(data["feasible"].shape[0], data["x"].shape[0])

if __name__ == "__main__":
    unittest.main()
