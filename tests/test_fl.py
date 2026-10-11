"""Unit tests for Federated Learning with Differential Privacy."""

import unittest
import torch
from core.federated_learning import FederatedClient, FederatedServer
from simulation.iot_devices import get_default_smart_home_devices
from simulation.traffic_generator import TrafficGenerator


class TestFederatedLearning(unittest.TestCase):
    def setUp(self):
        self.devices = get_default_smart_home_devices()
        self.generator = TrafficGenerator(self.devices)
        
        # Create 2 clients with small datasets
        self.client1 = FederatedClient("smart_gateway_01", "Gateway", "gateway")
        x1, y1 = self.generator.generate_dataset_for_device(self.devices["smart_gateway_01"], n_samples=40)
        self.client1.set_data(x1, y1)

        self.client2 = FederatedClient("smart_cam_01", "Camera", "camera")
        x2, y2 = self.generator.generate_dataset_for_device(self.devices["smart_cam_01"], n_samples=40)
        self.client2.set_data(x2, y2)

        self.server = FederatedServer()
        val_x, val_y = self.generator.generate_validation_set(n_samples=50)
        self.server.set_validation_data(val_x, val_y)

    def test_federated_round(self):
        summary = self.server.run_round(
            clients=[self.client1, self.client2],
            clipping_norm=1.0,
            noise_multiplier=0.3,
            local_epochs=1,
            lr=0.01
        )
        self.assertEqual(summary["round"], 1)
        self.assertEqual(summary["clients_participated"], 2)
        self.assertIn("val_metrics", summary)
        self.assertIn("accuracy", summary["val_metrics"])
        self.assertTrue(summary["epsilon"] > 0)


if __name__ == "__main__":
    unittest.main()
