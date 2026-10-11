"""Unit tests for Differential Privacy components."""

import unittest
import torch
import math
from core.differential_privacy import (
    compute_l2_norm,
    clip_weight_deltas,
    add_gaussian_noise,
    apply_differential_privacy,
    PrivacyAccountant
)


class TestDifferentialPrivacy(unittest.TestCase):
    def test_l2_norm_computation(self):
        tensor_dict = {
            "w1": torch.tensor([3.0, 4.0]),  # sqrt(9 + 16) = 5.0
        }
        norm = compute_l2_norm(tensor_dict)
        self.assertAlmostEqual(norm, 5.0, places=4)

    def test_clipping(self):
        tensor_dict = {
            "w1": torch.tensor([6.0, 8.0]),  # norm = 10.0
        }
        max_norm = 5.0
        clipped, orig_norm, was_clipped = clip_weight_deltas(tensor_dict, max_norm)
        self.assertAlmostEqual(orig_norm, 10.0, places=4)
        self.assertTrue(was_clipped)
        clipped_norm = compute_l2_norm(clipped)
        self.assertAlmostEqual(clipped_norm, 5.0, places=4)

    def test_clipping_no_op_when_below_threshold(self):
        tensor_dict = {
            "w1": torch.tensor([1.0, 1.0]),  # norm = sqrt(2) ~ 1.414
        }
        max_norm = 5.0
        clipped, orig_norm, was_clipped = clip_weight_deltas(tensor_dict, max_norm)
        self.assertFalse(was_clipped)
        self.assertAlmostEqual(compute_l2_norm(clipped), orig_norm, places=4)

    def test_gaussian_noise(self):
        tensor_dict = {
            "w1": torch.zeros((100, 100))
        }
        noisy = add_gaussian_noise(tensor_dict, clipping_norm=1.0, noise_multiplier=0.5)
        std_sample = noisy["w1"].std().item()
        # Sample standard deviation should be close to 0.5
        self.assertAlmostEqual(std_sample, 0.5, delta=0.08)

    def test_privacy_accountant_stepping(self):
        accountant = PrivacyAccountant(target_delta=1e-5)
        eps1 = accountant.step(noise_multiplier=0.5, sample_ratio=1.0)
        self.assertTrue(eps1 > 0)
        eps2 = accountant.step(noise_multiplier=0.5, sample_ratio=1.0)
        self.assertTrue(eps2 > eps1)  # Budget accumulates over rounds
        summary = accountant.get_summary()
        self.assertEqual(summary["rounds_completed"], 2)
        self.assertTrue(summary["is_private"])


if __name__ == "__main__":
    unittest.main()
