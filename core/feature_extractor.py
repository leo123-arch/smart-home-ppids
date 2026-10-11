"""Network Flow Feature Extractor and Normalizer.

Transforms raw network flow statistics into normalized 10-dimensional tensors
suitable for edge inference and federated training.
"""

from typing import Dict, Any, List, Union
import numpy as np
import torch
from core.model import FEATURE_NAMES, NUM_FEATURES


# Baseline normalization statistics derived from smart home IoT traffic profiles
FEATURE_NORMALIZATION = {
    # feature: (min_val, max_val) for min-max scaling to [0, 1]
    "packet_rate": (0.0, 15000.0),       # pps
    "byte_rate": (0.0, 15000000.0),      # bytes/sec (up to 15 MB/s)
    "syn_ratio": (0.0, 1.0),             # proportion
    "ack_ratio": (0.0, 1.0),             # proportion
    "udp_ratio": (0.0, 1.0),             # proportion
    "avg_packet_size": (40.0, 1500.0),   # bytes
    "packet_size_std": (0.0, 600.0),     # bytes std
    "flow_duration": (0.1, 60.0),        # seconds
    "dst_port_entropy": (0.0, 1.0),      # entropy [0, 1]
    "src_ip_entropy": (0.0, 1.0)         # entropy [0, 1]
}


class FlowFeatureExtractor:
    """Extracts and normalizes features from raw network flow summaries."""
    
    @staticmethod
    def normalize_value(val: float, min_val: float, max_val: float) -> float:
        """Clamps and normalizes a scalar to [0, 1]."""
        clamped = max(min_val, min(float(val), max_val))
        return (clamped - min_val) / max(max_val - min_val, 1e-6)

    @classmethod
    def extract_features(cls, flow_dict: Dict[str, Any]) -> np.ndarray:
        """Extracts and normalizes the 10-dimensional feature vector from a flow dictionary."""
        vector = []
        for feat in FEATURE_NAMES:
            raw_val = flow_dict.get(feat, 0.0)
            min_val, max_val = FEATURE_NORMALIZATION[feat]
            norm_val = cls.normalize_value(raw_val, min_val, max_val)
            vector.append(norm_val)
        return np.array(vector, dtype=np.float32)

    @classmethod
    def to_tensor(cls, flow_dict: Dict[str, Any]) -> torch.Tensor:
        """Returns a torch Tensor with shape [1, 10]."""
        feat_vec = cls.extract_features(flow_dict)
        return torch.tensor(feat_vec, dtype=torch.float32).unsqueeze(0)

    @classmethod
    def batch_to_tensor(cls, flows: List[Dict[str, Any]]) -> torch.Tensor:
        """Converts a list of flow dictionaries into a batch tensor [N, 10]."""
        vectors = [cls.extract_features(f) for f in flows]
        return torch.tensor(np.array(vectors), dtype=torch.float32)
