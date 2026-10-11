"""Federated Learning Core: Local Edge Clients and Central FedAvg Aggregator with Differential Privacy.

Enables collaborative training across smart home IoT edge devices without
centralizing raw network telemetry.
"""

from typing import Dict, List, Tuple, Any, Optional
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from core.model import DDoSClassifier
from core.differential_privacy import apply_differential_privacy, PrivacyAccountant


class FederatedClient:
    """Simulates a smart home IoT edge device (e.g., Camera, Gateway, Hub).
    
    The device trains exclusively on its local, private telemetry.
    Only differentially private perturbed model weights are shared.
    """
    def __init__(self, client_id: str, name: str, device_type: str):
        self.client_id = client_id
        self.name = name
        self.device_type = device_type
        self.local_model = DDoSClassifier()
        self.x_data: Optional[torch.Tensor] = None
        self.y_data: Optional[torch.Tensor] = None

    def set_data(self, x: torch.Tensor, y: torch.Tensor):
        """Sets the private local dataset for this edge device."""
        self.x_data = x.clone().detach()
        self.y_data = y.clone().detach()

    @property
    def num_samples(self) -> int:
        return self.x_data.size(0) if self.x_data is not None else 0

    def local_train(
        self,
        global_weights: Dict[str, torch.Tensor],
        epochs: int = 2,
        lr: float = 0.01,
        batch_size: int = 16,
        clipping_norm: float = 1.0,
        noise_multiplier: float = 0.5
    ) -> Dict[str, Any]:
        """Performs local training on private data, applies DP, and returns privatized weights."""
        if self.x_data is None or self.y_data is None or len(self.x_data) == 0:
            raise ValueError(f"Client {self.client_id} has no training data.")

        # 1. Synchronize with latest global model
        self.local_model.set_weights(global_weights)
        self.local_model.train()

        dataset = TensorDataset(self.x_data, self.y_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.local_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        steps = 0

        for _ in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.local_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1

        avg_loss = total_loss / max(steps, 1)

        # 2. Extract trained weights
        raw_local_weights = self.local_model.get_weights()

        # 3. Apply Local Differential Privacy (Clipping + Calibrated Gaussian Noise)
        privatized_weights, privacy_meta = apply_differential_privacy(
            local_weights=raw_local_weights,
            global_weights=global_weights,
            clipping_norm=clipping_norm,
            noise_multiplier=noise_multiplier
        )

        # Evaluate local performance before sharing
        local_eval = self.local_model.evaluate(self.x_data, self.y_data)

        return {
            "client_id": self.client_id,
            "name": self.name,
            "weights": privatized_weights,
            "num_samples": self.num_samples,
            "train_loss": float(avg_loss),
            "eval_metrics": local_eval,
            "privacy_meta": privacy_meta
        }


class FederatedServer:
    """Central Orchestrator for Federated Averaging (FedAvg) and DP Accounting."""
    
    def __init__(self, target_delta: float = 1e-5):
        self.global_model = DDoSClassifier()
        self.accountant = PrivacyAccountant(target_delta=target_delta)
        self.history: List[Dict[str, Any]] = []
        self.val_x: Optional[torch.Tensor] = None
        self.val_y: Optional[torch.Tensor] = None

    def set_validation_data(self, val_x: torch.Tensor, val_y: torch.Tensor):
        """Sets the global validation test set for evaluating rounds."""
        self.val_x = val_x
        self.val_y = val_y

    def aggregate_fedavg(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Computes weighted average of privatized weights across clients:
        
        W_global = sum( (n_k / N_total) * W_k_privatized )
        """
        total_samples = sum(u["num_samples"] for u in client_updates)
        if total_samples == 0:
            return self.global_model.get_weights()

        # Initialize accumulator with zeros matching model structure
        aggregated_weights: Dict[str, torch.Tensor] = {}
        first_weights = client_updates[0]["weights"]
        for key in first_weights.keys():
            aggregated_weights[key] = torch.zeros_like(first_weights[key], dtype=torch.float32)

        # Weighted summation
        for update in client_updates:
            weight_factor = update["num_samples"] / total_samples
            for key, param in update["weights"].items():
                aggregated_weights[key] += param.float() * weight_factor

        self.global_model.set_weights(aggregated_weights)
        return aggregated_weights

    def run_round(
        self,
        clients: List[FederatedClient],
        clipping_norm: float = 1.0,
        noise_multiplier: float = 0.5,
        local_epochs: int = 2,
        lr: float = 0.01,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """Executes a complete Federated Learning round with Differential Privacy."""
        current_global_weights = self.global_model.get_weights()

        # 1. Edge devices train locally and privatize updates
        client_updates = []
        for client in clients:
            if client.num_samples > 0:
                update = client.local_train(
                    global_weights=current_global_weights,
                    epochs=local_epochs,
                    lr=lr,
                    batch_size=batch_size,
                    clipping_norm=clipping_norm,
                    noise_multiplier=noise_multiplier
                )
                client_updates.append(update)

        if not client_updates:
            return {"error": "No participating clients with training data."}

        # 2. Server aggregates privatized updates via FedAvg
        self.aggregate_fedavg(client_updates)

        # 3. Privacy Accountant step (subsampling ratio q = batch_size / local_samples)
        client_sample_count = max(client_updates[0]["num_samples"], 1)
        subsample_q = min(1.0, max(0.01, batch_size / client_sample_count))
        eps = self.accountant.step(noise_multiplier=noise_multiplier, sample_ratio=subsample_q)
        privacy_status = self.accountant.get_summary()

        # 4. Evaluate global model on validation set
        val_metrics = {}
        if self.val_x is not None and self.val_y is not None:
            val_metrics = self.global_model.evaluate(self.val_x, self.val_y)

        round_number = len(self.history) + 1
        round_summary = {
            "round": round_number,
            "clients_participated": len(client_updates),
            "clipping_norm": clipping_norm,
            "noise_multiplier": noise_multiplier,
            "epsilon": eps,
            "delta": self.accountant.target_delta,
            "privacy_tier": privacy_status["privacy_tier"],
            "val_metrics": val_metrics,
            "client_summaries": [
                {
                    "client_id": u["client_id"],
                    "name": u["name"],
                    "num_samples": u["num_samples"],
                    "train_loss": u["train_loss"],
                    "local_accuracy": u["eval_metrics"]["accuracy"],
                    "privacy_meta": u["privacy_meta"]
                }
                for u in client_updates
            ]
        }

        self.history.append(round_summary)
        return round_summary
