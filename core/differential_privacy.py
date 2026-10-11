"""Differential Privacy (DP) Engine for Federated Smart Home IDS.

Provides:
1. L2 Norm gradient / weight delta clipping.
2. Calibrated Gaussian mechanism noise injection.
3. (Epsilon, Delta)-DP Privacy Budget Accountant.
"""

from typing import Dict, Tuple, Optional
import math
import torch


def compute_l2_norm(tensor_dict: Dict[str, torch.Tensor]) -> float:
    """Computes the total L2 norm across all parameter tensors in the dictionary."""
    total_sq_norm = 0.0
    for tensor in tensor_dict.values():
        total_sq_norm += torch.sum(tensor ** 2).item()
    return math.sqrt(total_sq_norm)


def clip_weight_deltas(
    delta_dict: Dict[str, torch.Tensor],
    max_norm: float
) -> Tuple[Dict[str, torch.Tensor], float, bool]:
    """Clips the total L2 norm of weight deltas to max_norm (C).
    
    Returns:
        clipped_deltas: Scaled deltas with norm <= max_norm.
        original_norm: The computed L2 norm before clipping.
        was_clipped: Boolean indicating whether clipping was triggered.
    """
    original_norm = compute_l2_norm(delta_dict)
    if original_norm <= 0:
        return delta_dict, 0.0, False

    clip_factor = min(1.0, max_norm / (original_norm + 1e-12))
    was_clipped = clip_factor < 1.0

    clipped_deltas = {}
    for name, param in delta_dict.items():
        clipped_deltas[name] = param * clip_factor

    return clipped_deltas, original_norm, was_clipped


def add_gaussian_noise(
    clipped_delta_dict: Dict[str, torch.Tensor],
    clipping_norm: float,
    noise_multiplier: float
) -> Dict[str, torch.Tensor]:
    """Injects calibrated Gaussian noise: N(0, (noise_multiplier * clipping_norm)^2).
    
    If noise_multiplier == 0, returns the unperturbed deltas (non-private baseline).
    """
    if noise_multiplier <= 0.0:
        return clipped_delta_dict

    std_dev = noise_multiplier * clipping_norm
    noisy_deltas = {}

    for name, param in clipped_delta_dict.items():
        noise = torch.randn_like(param) * std_dev
        noisy_deltas[name] = param + noise

    return noisy_deltas


def apply_differential_privacy(
    local_weights: Dict[str, torch.Tensor],
    global_weights: Dict[str, torch.Tensor],
    clipping_norm: float = 1.0,
    noise_multiplier: float = 0.5
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Computes weight update delta, applies L2 clipping and Gaussian noise,
    and reconstructs the privatized local weights for federated aggregation.
    
    delta = local_weights - global_weights
    delta_clipped = clip(delta, C)
    delta_noisy = delta_clipped + N(0, (sigma * C)^2)
    privatized_weights = global_weights + delta_noisy
    """
    # 1. Compute weight deltas
    deltas = {}
    for name, param in local_weights.items():
        deltas[name] = param - global_weights[name]

    # 2. Clip deltas to bounded L2 sensitivity C
    clipped_deltas, orig_norm, was_clipped = clip_weight_deltas(deltas, clipping_norm)

    # 3. Add calibrated Gaussian perturbation
    noisy_deltas = add_gaussian_noise(clipped_deltas, clipping_norm, noise_multiplier)
    post_noise_norm = compute_l2_norm(noisy_deltas)

    # 4. Reconstruct privatized weights
    privatized_weights = {}
    for name in global_weights.keys():
        privatized_weights[name] = global_weights[name] + noisy_deltas[name]

    privacy_meta = {
        "pre_clip_l2_norm": float(orig_norm),
        "post_noise_l2_norm": float(post_noise_norm),
        "clipping_norm": float(clipping_norm),
        "noise_multiplier": float(noise_multiplier),
        "was_clipped": bool(was_clipped)
    }

    return privatized_weights, privacy_meta


class PrivacyAccountant:
    """Tracks cumulative (epsilon, delta)-DP privacy budget over federated training rounds.
    
    Uses Advanced Composition / Moments Accountant approximation for Gaussian mechanisms.
    """
    def __init__(self, target_delta: float = 1e-5):
        self.target_delta = target_delta
        self.rounds_completed = 0
        self.cumulative_epsilon = 0.0
        self.history = []

    def step(self, noise_multiplier: float, sample_ratio: float = 1.0) -> float:
        """Records one federated learning round and updates cumulative privacy loss epsilon.
        
        Using the standard Gaussian composition bound:
        For noise_multiplier sigma, sensitivity 1, sampling ratio q in round t:
        eps_step approx (q * sqrt(2 * ln(1.25 / delta))) / sigma
        With T rounds advanced composition:
        eps_total approx q * sqrt(2 * T * ln(1 / delta)) / sigma + T * q^2 / (2 * sigma^2)
        """
        self.rounds_completed += 1

        if noise_multiplier <= 0.0:
            # No noise means no formal DP guarantee (infinite epsilon)
            self.cumulative_epsilon = float("inf")
            eps_status = "No Privacy Guarantee (Noise = 0)"
        else:
            t = self.rounds_completed
            q = max(min(sample_ratio, 1.0), 0.01)
            sigma = noise_multiplier
            delta = self.target_delta
            
            # Analytical composition estimate
            term1 = (q * math.sqrt(2.0 * t * math.log(1.0 / delta))) / sigma
            term2 = (t * (q ** 2)) / (2.0 * (sigma ** 2))
            self.cumulative_epsilon = round(term1 + term2, 4)

            if self.cumulative_epsilon < 1.0:
                eps_status = "Very Strong Privacy (ε < 1.0)"
            elif self.cumulative_epsilon < 4.0:
                eps_status = "Strong / Commercial Grade Privacy (ε < 4.0)"
            elif self.cumulative_epsilon < 10.0:
                eps_status = "Moderate Privacy (ε < 10.0)"
            else:
                eps_status = "Relaxed Privacy (ε ≥ 10.0)"

        record = {
            "round": self.rounds_completed,
            "noise_multiplier": noise_multiplier,
            "epsilon": self.cumulative_epsilon if math.isfinite(self.cumulative_epsilon) else 9999.0,
            "delta": self.target_delta,
            "status": eps_status
        }
        self.history.append(record)
        return self.cumulative_epsilon

    def get_summary(self) -> Dict:
        """Returns the current privacy accounting status."""
        current_eps = self.cumulative_epsilon if math.isfinite(self.cumulative_epsilon) else "Infinity (No DP)"
        if self.cumulative_epsilon == 0.0:
            status = "Initial (No rounds executed)"
        elif not math.isfinite(self.cumulative_epsilon):
            status = "No Privacy Guarantee"
        elif self.cumulative_epsilon < 1.0:
            status = "Very Strong Privacy"
        elif self.cumulative_epsilon < 4.0:
            status = "Strong Privacy"
        elif self.cumulative_epsilon < 10.0:
            status = "Moderate Privacy"
        else:
            status = "Relaxed Privacy"

        return {
            "rounds_completed": self.rounds_completed,
            "cumulative_epsilon": current_eps,
            "target_delta": self.target_delta,
            "privacy_tier": status,
            "is_private": math.isfinite(self.cumulative_epsilon) and self.cumulative_epsilon < 9999
        }

    def reset(self):
        """Resets the privacy accountant."""
        self.rounds_completed = 0
        self.cumulative_epsilon = 0.0
        self.history = []
