"""Smart Home IoT Devices Model and Registry.

Defines virtual smart home IoT edge devices with distinct baseline traffic profiles
and security lifecycle states.
"""

from typing import Dict, List, Any


class SmartDevice:
    """Represents a connected smart home IoT device."""
    def __init__(
        self,
        device_id: str,
        name: str,
        ip_address: str,
        mac_address: str,
        device_type: str,
        icon: str,
        baseline_profile: Dict[str, Any]
    ):
        self.device_id = device_id
        self.name = name
        self.ip_address = ip_address
        self.mac_address = mac_address
        self.device_type = device_type
        self.icon = icon
        self.baseline_profile = baseline_profile
        self.status = "normal"  # normal, under_attack, mitigated, quarantined
        self.active_mitigations: List[str] = []
        self.total_packets_seen = 0
        self.last_ddos_prob = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "device_type": self.device_type,
            "icon": self.icon,
            "status": self.status,
            "active_mitigations": self.active_mitigations,
            "last_ddos_prob": round(self.last_ddos_prob, 4),
            "baseline": self.baseline_profile
        }


def get_default_smart_home_devices() -> Dict[str, SmartDevice]:
    """Instantiates standard smart home IoT edge devices."""
    devices = {
        "smart_gateway_01": SmartDevice(
            device_id="smart_gateway_01",
            name="Central Home Gateway",
            ip_address="192.168.1.1",
            mac_address="A4:C3:F0:8B:12:01",
            device_type="gateway",
            icon="router",
            baseline_profile={
                "packet_rate": (40.0, 120.0),
                "byte_rate": (30000.0, 120000.0),
                "syn_ratio": (0.05, 0.15),
                "ack_ratio": (0.70, 0.90),
                "udp_ratio": (0.20, 0.40),
                "avg_packet_size": (450.0, 850.0),
                "packet_size_std": (200.0, 450.0),
                "flow_duration": (5.0, 30.0),
                "dst_port_entropy": (0.6, 0.85),
                "src_ip_entropy": (0.5, 0.75)
            }
        ),
        "smart_cam_01": SmartDevice(
            device_id="smart_cam_01",
            name="Living Room 4K Security Cam",
            ip_address="192.168.1.101",
            mac_address="3C:52:A1:77:4D:11",
            device_type="camera",
            icon="videocam",
            baseline_profile={
                "packet_rate": (80.0, 250.0),
                "byte_rate": (150000.0, 800000.0),
                "syn_ratio": (0.01, 0.05),
                "ack_ratio": (0.85, 0.98),
                "udp_ratio": (0.05, 0.20),
                "avg_packet_size": (800.0, 1300.0),
                "packet_size_std": (150.0, 350.0),
                "flow_duration": (15.0, 50.0),
                "dst_port_entropy": (0.1, 0.3),
                "src_ip_entropy": (0.1, 0.3)
            }
        ),
        "smart_thermostat_01": SmartDevice(
            device_id="smart_thermostat_01",
            name="Climate Thermostat & HVAC",
            ip_address="192.168.1.105",
            mac_address="58:24:29:E3:6C:9B",
            device_type="thermostat",
            icon="thermostat",
            baseline_profile={
                "packet_rate": (2.0, 15.0),
                "byte_rate": (500.0, 4000.0),
                "syn_ratio": (0.02, 0.08),
                "ack_ratio": (0.80, 0.95),
                "udp_ratio": (0.10, 0.30),
                "avg_packet_size": (120.0, 350.0),
                "packet_size_std": (50.0, 120.0),
                "flow_duration": (2.0, 20.0),
                "dst_port_entropy": (0.05, 0.2),
                "src_ip_entropy": (0.05, 0.2)
            }
        ),
        "smart_lock_01": SmartDevice(
            device_id="smart_lock_01",
            name="Front Door Smart Lock",
            ip_address="192.168.1.110",
            mac_address="AC:67:84:90:5E:2A",
            device_type="lock",
            icon="lock",
            baseline_profile={
                "packet_rate": (1.0, 8.0),
                "byte_rate": (300.0, 2500.0),
                "syn_ratio": (0.05, 0.12),
                "ack_ratio": (0.75, 0.92),
                "udp_ratio": (0.0, 0.10),
                "avg_packet_size": (150.0, 400.0),
                "packet_size_std": (60.0, 150.0),
                "flow_duration": (1.0, 10.0),
                "dst_port_entropy": (0.05, 0.15),
                "src_ip_entropy": (0.05, 0.15)
            }
        ),
        "smart_speaker_01": SmartDevice(
            device_id="smart_speaker_01",
            name="Voice Assistant Speaker",
            ip_address="192.168.1.115",
            mac_address="70:EE:50:33:1B:FF",
            device_type="speaker",
            icon="speaker",
            baseline_profile={
                "packet_rate": (10.0, 60.0),
                "byte_rate": (15000.0, 90000.0),
                "syn_ratio": (0.03, 0.08),
                "ack_ratio": (0.80, 0.95),
                "udp_ratio": (0.15, 0.35),
                "avg_packet_size": (350.0, 750.0),
                "packet_size_std": (120.0, 280.0),
                "flow_duration": (5.0, 25.0),
                "dst_port_entropy": (0.2, 0.45),
                "src_ip_entropy": (0.2, 0.4)
            }
        )
    }
    return devices
