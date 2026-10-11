"""Unit tests for DDoSClassifier model."""

import unittest
import torch
from core.model import DDoSClassifier, NUM_FEATURES


class TestDDoSClassifier(unittest.TestCase):
    def setUp(self):
        self.model = DDoSClassifier()

    def test_forward_shape(self):
        batch_size = 8
        dummy_input = torch.randn(batch_size, NUM_FEATURES)
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (batch_size, 2))

    def test_predict_proba(self):
        dummy_input = torch.randn(4, NUM_FEATURES)
        probs = self.model.predict_proba(dummy_input)
        self.assertEqual(probs.shape, (4, 2))
        # Sum of probabilities per row should be ~1.0
        sums = torch.sum(probs, dim=-1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=4)

    def test_get_and_set_weights(self):
        weights = self.model.get_weights()
        self.assertIn("fc1.weight", weights)
        self.assertIn("ln1.weight", weights)
        
        # Modify and restore
        new_model = DDoSClassifier()
        new_model.set_weights(weights)
        for k, v in weights.items():
            self.assertTrue(torch.equal(v, new_model.state_dict()[k].cpu()))

    def test_evaluate(self):
        x = torch.randn(20, NUM_FEATURES)
        y = torch.randint(0, 2, (20,))
        metrics = self.model.evaluate(x, y)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("loss", metrics)
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
