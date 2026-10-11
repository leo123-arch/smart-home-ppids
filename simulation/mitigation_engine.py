"""Automated Edge Mitigation Engine for Smart Home IoT Networks.

Handles real-time anomaly alerts and triggers edge defense policies:
- Dynamic Firewall Drop Rules
- Adaptive Bandwidth Rate-Limiting
- Device Isolation / VLAN Quarantine
"""

from typing import Dict, List, Any, Optional
import time
import datetime
from simulation.iot_devices import SmartDevice


class MitigationEngine:
    """Evaluates PP-IDS model confidence and applies automated edge defenses."""

    def __init__(self, ddos_threshold: float = 0.82):
        self.ddos_threshold = ddos_threshold
        self.alerts: List[Dict[str, Any]] = []
        self.active_firewall_rules: List[Dict[str, Any]] = []
        self.mitigation_log: List[Dict[str, Any]] = []

    def evaluate_and_mitigate(
        self,
        device: SmartDevice,
        flow_features: Dict[str, Any],
        ddos_prob: float
    ) -> Optional[Dict[str, Any]]:
        """Checks if flow exceeds threat threshold, raises alert, and executes mitigation."""
        device.last_ddos_prob = ddos_prob

        if ddos_prob < self.ddos_threshold:
            # Benign traffic or below threshold
            if device.status == "under_attack":
                # Restored to normal
                device.status = "normal"
            return None

        # Threat detected! Determine signature
        syn_ratio = flow_features.get("syn_ratio", 0.0)
        udp_ratio = flow_features.get("udp_ratio", 0.0)
        pkt_rate = flow_features.get("packet_rate", 0.0)
        avg_pkt_size = flow_features.get("avg_packet_size", 0.0)

        if syn_ratio > 0.7:
            attack_type_guess = "TCP SYN Flood"
            recommended_action = "Dynamic TCP SYN Cookies & Rate-Limiting"
        elif udp_ratio > 0.7:
            attack_type_guess = "UDP Amplification / Flood"
            recommended_action = "Port Filtering & UDP Ingress Drop"
        elif avg_pkt_size > 300 and syn_ratio < 0.4 and udp_ratio < 0.2:
            attack_type_guess = "Layer-7 HTTP / API Flood"
            recommended_action = "Connection Throttling & Captcha/Token Challenge"
        else:
            attack_type_guess = "Mirai Botnet / Volumetric Flood"
            recommended_action = "VLAN Quarantine & Ingress Drop"

        alert_id = f"ALT-{int(time.time() * 1000) % 1000000:06d}"
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # Apply mitigation to device
        actions_applied = []
        
        # 1. Firewall Drop Rule
        fw_rule = {
            "rule_id": f"FW-{len(self.active_firewall_rules) + 1}",
            "target_ip": device.ip_address,
            "action": "DROP_BURST_INGRESS",
            "timestamp": timestamp
        }
        self.active_firewall_rules.append(fw_rule)
        actions_applied.append(f"Firewall: DROP_BURST on {device.ip_address}")

        # 2. Rate limiting & Device State
        if "Quarantine" in recommended_action:
            device.status = "quarantined"
            actions_applied.append(f"Quarantined {device.name} to Isolated VLAN")
        else:
            device.status = "mitigated"
            actions_applied.append(f"Applied 10% Bandwidth Rate-Limit on {device.name}")

        device.active_mitigations = actions_applied

        alert = {
            "alert_id": alert_id,
            "timestamp": timestamp,
            "device_id": device.device_id,
            "device_name": device.name,
            "device_ip": device.ip_address,
            "ddos_probability": round(ddos_prob, 4),
            "threat_type": attack_type_guess,
            "packet_rate": round(pkt_rate, 1),
            "severity": "CRITICAL" if ddos_prob > 0.92 else "HIGH",
            "actions_applied": actions_applied
        }

        # Keep alerts list bounded
        self.alerts.insert(0, alert)
        if len(self.alerts) > 100:
            self.alerts.pop()

        self.mitigation_log.insert(0, {
            "timestamp": timestamp,
            "device_id": device.device_id,
            "action": ", ".join(actions_applied),
            "threat": attack_type_guess
        })
        if len(self.mitigation_log) > 100:
            self.mitigation_log.pop()

        return alert

    def reset_all_mitigations(self, devices: Dict[str, SmartDevice]):
        """Resets all firewall rules, rate limits, and returns devices to normal state."""
        self.active_firewall_rules.clear()
        for device in devices.values():
            device.status = "normal"
            device.active_mitigations.clear()
            device.last_ddos_prob = 0.0

    def get_summary(self) -> Dict[str, Any]:
        return {
            "active_rules_count": len(self.active_firewall_rules),
            "total_alerts": len(self.alerts),
            "recent_alerts": self.alerts[:10],
            "recent_mitigations": self.mitigation_log[:10]
        }
