"""Smart Home IoT Traffic Generator and DDoS Attack Simulator.

Simulates normal multi-device IoT telemetry and realistic volumetric DDoS attack variants:
- TCP SYN Flood
- UDP Amplification (DNS/NTP)
- Layer-7 HTTP Flood
- Mirai Botnet Burst
"""

from typing import Dict, List, Any, Tuple, Optional
import random
import numpy as np
import torch

from simulation.iot_devices import SmartDevice, get_default_smart_home_devices
from core.feature_extractor import FlowFeatureExtractor


class TrafficGenerator:
    """Simulates real-time IoT network flows and generates dataset splits for Federated Learning."""

    def __init__(self, devices: Optional[Dict[str, SmartDevice]] = None):
        self.devices = devices or get_default_smart_home_devices()
        self.active_attack: Optional[Dict[str, Any]] = None

    def start_attack(
        self,
        attack_type: str,
        target_device_id: str,
        intensity: float = 1.0
    ) -> Dict[str, Any]:
        """Initiates an active DDoS attack against a target device or the gateway."""
        if target_device_id not in self.devices:
            raise ValueError(f"Unknown device: {target_device_id}")

        valid_types = ["syn_flood", "udp_flood", "http_flood", "mirai_burst"]
        if attack_type not in valid_types:
            raise ValueError(f"Unknown attack type: {attack_type}. Valid: {valid_types}")

        self.active_attack = {
            "type": attack_type,
            "target_device_id": target_device_id,
            "intensity": max(0.2, min(float(intensity), 5.0)),
            "start_time": None
        }

        # Update device status
        self.devices[target_device_id].status = "under_attack"

        return self.active_attack

    def stop_attack(self) -> Optional[Dict[str, Any]]:
        """Halts the active DDoS attack and resets device states."""
        stopped = self.active_attack
        if stopped and stopped["target_device_id"] in self.devices:
            dev = self.devices[stopped["target_device_id"]]
            if dev.status == "under_attack":
                dev.status = "normal"

        self.active_attack = None
        return stopped

    def generate_normal_flow(self, device: SmartDevice) -> Dict[str, Any]:
        """Generates a realistic benign IoT network flow record based on device baseline profile."""
        p = device.baseline_profile
        
        packet_rate = random.uniform(p["packet_rate"][0], p["packet_rate"][1])
        byte_rate = random.uniform(p["byte_rate"][0], p["byte_rate"][1])
        syn_ratio = random.uniform(p["syn_ratio"][0], p["syn_ratio"][1])
        ack_ratio = random.uniform(p["ack_ratio"][0], p["ack_ratio"][1])
        udp_ratio = random.uniform(p["udp_ratio"][0], p["udp_ratio"][1])
        avg_packet_size = random.uniform(p["avg_packet_size"][0], p["avg_packet_size"][1])
        packet_size_std = random.uniform(p["packet_size_std"][0], p["packet_size_std"][1])
        flow_duration = random.uniform(p["flow_duration"][0], p["flow_duration"][1])
        dst_port_entropy = random.uniform(p["dst_port_entropy"][0], p["dst_port_entropy"][1])
        src_ip_entropy = random.uniform(p["src_ip_entropy"][0], p["src_ip_entropy"][1])

        return {
            "device_id": device.device_id,
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "syn_ratio": syn_ratio,
            "ack_ratio": ack_ratio,
            "udp_ratio": udp_ratio,
            "avg_packet_size": avg_packet_size,
            "packet_size_std": packet_size_std,
            "flow_duration": flow_duration,
            "dst_port_entropy": dst_port_entropy,
            "src_ip_entropy": src_ip_entropy,
            "label": 0  # Normal
        }

    def generate_attack_flow(
        self,
        device: SmartDevice,
        attack_type: str,
        intensity: float = 1.0
    ) -> Dict[str, Any]:
        """Generates a realistic DDoS attack flow record based on attack signature."""
        mult = max(0.5, intensity)

        if attack_type == "syn_flood":
            # High packet rate, small packets, pure SYN flags, randomized spoofed source IPs
            packet_rate = random.uniform(3500.0, 9500.0) * mult
            avg_packet_size = random.uniform(54.0, 74.0)
            packet_size_std = random.uniform(0.0, 8.0)
            byte_rate = packet_rate * avg_packet_size
            syn_ratio = random.uniform(0.92, 1.0)
            ack_ratio = random.uniform(0.0, 0.05)
            udp_ratio = 0.0
            dst_port_entropy = random.uniform(0.05, 0.2)  # Focused on target port (e.g., 80, 443, 554)
            src_ip_entropy = random.uniform(0.85, 0.99)    # High spoofed source entropy
            flow_duration = random.uniform(10.0, 60.0)

        elif attack_type == "udp_flood":
            # Volumetric UDP amplification (DNS/NTP/SSDP reflection), large packet payloads
            packet_rate = random.uniform(2800.0, 8000.0) * mult
            avg_packet_size = random.uniform(1100.0, 1480.0)
            packet_size_std = random.uniform(20.0, 70.0)
            byte_rate = packet_rate * avg_packet_size
            syn_ratio = 0.0
            ack_ratio = 0.0
            udp_ratio = random.uniform(0.95, 1.0)
            dst_port_entropy = random.uniform(0.02, 0.15)
            src_ip_entropy = random.uniform(0.70, 0.95)
            flow_duration = random.uniform(15.0, 60.0)

        elif attack_type == "http_flood":
            # Layer-7 application flood: Rapid HTTP GET/POST requests
            packet_rate = random.uniform(1200.0, 4500.0) * mult
            avg_packet_size = random.uniform(380.0, 650.0)
            packet_size_std = random.uniform(40.0, 120.0)
            byte_rate = packet_rate * avg_packet_size
            syn_ratio = random.uniform(0.12, 0.28)
            ack_ratio = random.uniform(0.70, 0.88)
            udp_ratio = 0.0
            dst_port_entropy = random.uniform(0.01, 0.1)   # HTTP/HTTPS port
            src_ip_entropy = random.uniform(0.75, 0.96)
            flow_duration = random.uniform(8.0, 40.0)

        elif attack_type == "mirai_burst":
            # Mirai-style IoT botnet volumetric attack + command propagation
            packet_rate = random.uniform(4000.0, 11000.0) * mult
            avg_packet_size = random.uniform(200.0, 900.0)
            packet_size_std = random.uniform(10.0, 50.0)
            byte_rate = packet_rate * avg_packet_size
            syn_ratio = random.uniform(0.40, 0.70)
            ack_ratio = random.uniform(0.10, 0.30)
            udp_ratio = random.uniform(0.30, 0.60)
            dst_port_entropy = random.uniform(0.35, 0.75)
            src_ip_entropy = random.uniform(0.85, 0.99)
            flow_duration = random.uniform(20.0, 60.0)

        else:
            # Generic fallback volumetric flood
            packet_rate = random.uniform(2000.0, 6000.0) * mult
            avg_packet_size = random.uniform(300.0, 800.0)
            byte_rate = packet_rate * avg_packet_size
            syn_ratio = random.uniform(0.5, 0.8)
            ack_ratio = random.uniform(0.1, 0.3)
            udp_ratio = random.uniform(0.2, 0.5)
            packet_size_std = 50.0
            dst_port_entropy = 0.3
            src_ip_entropy = 0.8
            flow_duration = 30.0

        return {
            "device_id": device.device_id,
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "syn_ratio": syn_ratio,
            "ack_ratio": ack_ratio,
            "udp_ratio": udp_ratio,
            "avg_packet_size": avg_packet_size,
            "packet_size_std": packet_size_std,
            "flow_duration": flow_duration,
            "dst_port_entropy": dst_port_entropy,
            "src_ip_entropy": src_ip_entropy,
            "label": 1  # DDoS
        }

    def generate_dataset_for_device(
        self,
        device: SmartDevice,
        n_samples: int = 250,
        attack_ratio: float = 0.35
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a private local training split (features, labels) for an edge device.
        
        Contains both normal device traffic and diverse DDoS attack patterns.
        """
        n_attack = int(n_samples * attack_ratio)
        n_normal = n_samples - n_attack

        flows = []
        # Normal flows
        for _ in range(n_normal):
            flows.append(self.generate_normal_flow(device))

        # Attack flows
        attack_types = ["syn_flood", "udp_flood", "http_flood", "mirai_burst"]
        for _ in range(n_attack):
            atk_type = random.choice(attack_types)
            intensity = random.uniform(0.8, 2.0)
            flows.append(self.generate_attack_flow(device, atk_type, intensity))

        random.shuffle(flows)

        x_tensor = FlowFeatureExtractor.batch_to_tensor(flows)
        y_tensor = torch.tensor([f["label"] for f in flows], dtype=torch.long)
        return x_tensor, y_tensor

    def generate_validation_set(self, n_samples: int = 500) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a comprehensive balanced validation set across all devices and attacks."""
        flows = []
        devices_list = list(self.devices.values())
        attack_types = ["syn_flood", "udp_flood", "http_flood", "mirai_burst"]

        for _ in range(n_samples // 2):
            dev = random.choice(devices_list)
            flows.append(self.generate_normal_flow(dev))

        for _ in range(n_samples // 2):
            dev = random.choice(devices_list)
            atk_type = random.choice(attack_types)
            flows.append(self.generate_attack_flow(dev, atk_type, random.uniform(0.6, 2.5)))

        random.shuffle(flows)
        x_tensor = FlowFeatureExtractor.batch_to_tensor(flows)
        y_tensor = torch.tensor([f["label"] for f in flows], dtype=torch.long)
        return x_tensor, y_tensor

    def step_live_traffic(self) -> Dict[str, Dict[str, Any]]:
        """Generates one instant snapshot of live network traffic for each smart home device.
        
        Reflects ongoing attacks if active.
        """
        snapshots = {}
        for dev_id, device in self.devices.items():
            if self.active_attack and self.active_attack["target_device_id"] == dev_id:
                # Target is receiving DDoS flood
                flow = self.generate_attack_flow(
                    device,
                    self.active_attack["type"],
                    self.active_attack["intensity"]
                )
            elif (
                self.active_attack
                and self.active_attack["target_device_id"] != dev_id
                and self.active_attack["type"] == "mirai_burst"
                and dev_id == "smart_cam_01"
            ):
                # In Mirai attacks, the camera might be the compromised bot sending traffic
                flow = self.generate_attack_flow(device, "mirai_burst", 0.8)
            else:
                # Normal baseline traffic
                flow = self.generate_normal_flow(device)

            snapshots[dev_id] = flow

        return snapshots
