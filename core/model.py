"""DDoS Detection Neural Network for Smart Home Edge Nodes.

Uses a lightweight Multi-Layer Perceptron (MLP) with Layer Normalization,
specifically designed for edge IoT execution and differential privacy compatibility.
"""

from typing import Dict, Tuple, Optional
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


FEATURE_NAMES = [
    "packet_rate",       # packets per second
    "byte_rate",         # bytes per second
    "syn_ratio",         # proportion of SYN packets [0, 1]
    "ack_ratio",         # proportion of ACK packets [0, 1]
    "udp_ratio",         # proportion of UDP packets [0, 1]
    "avg_packet_size",   # mean packet size in bytes
    "packet_size_std",   # std dev of packet sizes
    "flow_duration",     # duration of network flow in seconds
    "dst_port_entropy",  # entropy of destination ports [0, 1]
    "src_ip_entropy",    # entropy of source IPs [0, 1]
]

NUM_FEATURES = len(FEATURE_NAMES)


class DDoSClassifier(nn.Module):
    """Edge-friendly Neural Network for Smart Home DDoS Detection.
    
    LayerNorm is used instead of BatchNorm because BatchNorm introduces
    inter-sample dependencies that violate sample-level Differential Privacy.
    """
    def __init__(self, input_dim: int = NUM_FEATURES, hidden_dim1: int = 32, hidden_dim2: int = 16):
        super().__init__()
        self.input_dim = input_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        self.dropout1 = nn.Dropout(0.05)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.ln2 = nn.LayerNorm(hidden_dim2)
        
        self.fc3 = nn.Linear(hidden_dim2, 2)  # Class 0: Normal, Class 1: DDoS
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits [batch_size, 2]."""
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        logits = self.fc3(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Softmax probabilities [batch_size, 2]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def predict_ddos_prob(self, x: torch.Tensor) -> float:
        """Predict DDoS probability (class 1) for a single sample or batch."""
        probs = self.predict_proba(x)
        if probs.dim() == 1 or probs.size(0) == 1:
            return float(probs.view(-1, 2)[0, 1].item())
        return probs[:, 1].tolist()

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Extract a clone of model state dict."""
        return {k: v.clone().detach().cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """Load state dict into model."""
        self.load_state_dict({k: v.to(next(self.parameters()).device) for k, v in weights.items()})

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics (accuracy, precision, recall, f1, loss)."""
        self.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            logits = self.forward(x)
            loss = criterion(logits, y).item()
            preds = torch.argmax(logits, dim=-1)
            
            total = y.size(0)
            correct = (preds == y).sum().item()
            accuracy = correct / max(total, 1)
            
            # Binary classification metrics for DDoS (class 1)
            tp = ((preds == 1) & (y == 1)).sum().item()
            fp = ((preds == 1) & (y == 0)).sum().item()
            fn = ((preds == 0) & (y == 1)).sum().item()
            tn = ((preds == 0) & (y == 0)).sum().item()
            
            precision = tp / max(tp + fp, 1e-7)
            recall = tp / max(tp + fn, 1e-7)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-7)
            
            return {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn)
            }
