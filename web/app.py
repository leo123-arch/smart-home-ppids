"""Flask Backend Server and REST API for Privacy-Preserving Smart Home IDS.

Orchestrates live IoT traffic simulation, federated learning rounds with differential privacy,
real-time inference, and automated edge mitigations.
"""

from typing import Dict, Any, List
import time
import threading
import copy
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import torch

from core.model import DDoSClassifier
from core.feature_extractor import FlowFeatureExtractor
from core.federated_learning import FederatedClient, FederatedServer
from simulation.iot_devices import get_default_smart_home_devices, SmartDevice
from simulation.traffic_generator import TrafficGenerator
from simulation.mitigation_engine import MitigationEngine


app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Global State Container
class SystemState:
    def __init__(self):
        self.lock = threading.Lock()
        self.devices = get_default_smart_home_devices()
        self.traffic_gen = TrafficGenerator(self.devices)
        self.mitigation = MitigationEngine(ddos_threshold=0.80)
        self.server = FederatedServer(target_delta=1e-5)
        
        # Prepare Federated Clients with private local data
        self.clients: List[FederatedClient] = []
        for dev_id, dev in self.devices.items():
            client = FederatedClient(client_id=dev_id, name=dev.name, device_type=dev.device_type)
            x_data, y_data = self.traffic_gen.generate_dataset_for_device(dev, n_samples=200, attack_ratio=0.35)
            client.set_data(x_data, y_data)
            self.clients.append(client)

        # Global validation set
        val_x, val_y = self.traffic_gen.generate_validation_set(n_samples=400)
        self.server.set_validation_data(val_x, val_y)

        # Telemetry history buffer (last 30 snapshots)
        self.telemetry_history: List[Dict[str, Any]] = []
        self.latest_live_flows: Dict[str, Any] = {}
        
        # Auto-training flag
        self.auto_fl_enabled = False
        self.running = True

state = SystemState()


def background_traffic_worker():
    """Continuously steps network traffic and runs edge inference."""
    while state.running:
        try:
            with state.lock:
                timestamp_str = time.strftime("%H:%M:%S")
                snapshots = state.traffic_gen.step_live_traffic()
                state.latest_live_flows = snapshots

                # Real-time inference on each device flow using the global model
                device_telemetry_entry = {
                    "timestamp": timestamp_str,
                    "devices": {}
                }

                for dev_id, flow in snapshots.items():
                    device = state.devices[dev_id]
                    flow_tensor = FlowFeatureExtractor.to_tensor(flow)
                    ddos_prob = state.server.global_model.predict_ddos_prob(flow_tensor)
                    device.last_ddos_prob = ddos_prob

                    # Automated mitigation evaluation
                    alert = state.mitigation.evaluate_and_mitigate(device, flow, ddos_prob)

                    device_telemetry_entry["devices"][dev_id] = {
                        "packet_rate": round(flow["packet_rate"], 1),
                        "byte_rate": round(flow["byte_rate"], 1),
                        "syn_ratio": round(flow["syn_ratio"], 2),
                        "udp_ratio": round(flow["udp_ratio"], 2),
                        "ddos_prob": round(ddos_prob, 4),
                        "status": device.status
                    }

                # Maintain history
                state.telemetry_history.append(device_telemetry_entry)
                if len(state.telemetry_history) > 30:
                    state.telemetry_history.pop(0)

                # If auto-training is toggled and we have few rounds, run an occasional round
                if state.auto_fl_enabled and (len(state.server.history) < 25):
                    # Step FL round every ~10 iterations
                    if len(state.telemetry_history) % 8 == 0:
                        state.server.run_round(
                            clients=state.clients,
                            clipping_norm=1.0,
                            noise_multiplier=0.4,
                            local_epochs=1,
                            lr=0.01
                        )
        except Exception as e:
            print(f"Error in background worker: {e}")

        time.sleep(1.2)


# Start background simulation thread
sim_thread = threading.Thread(target=background_traffic_worker, daemon=True)
sim_thread.start()


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    with state.lock:
        val_acc = 0.0
        val_loss = 0.0
        val_f1 = 0.0
        if state.server.history:
            latest_metrics = state.server.history[-1].get("val_metrics", {})
            val_acc = latest_metrics.get("accuracy", 0.0)
            val_loss = latest_metrics.get("loss", 0.0)
            val_f1 = latest_metrics.get("f1", 0.0)
        else:
            # Baseline zero-round evaluation
            metrics = state.server.global_model.evaluate(state.server.val_x, state.server.val_y)
            val_acc = metrics.get("accuracy", 0.0)
            val_loss = metrics.get("loss", 0.0)
            val_f1 = metrics.get("f1", 0.0)

        privacy_info = state.server.accountant.get_summary()

        return jsonify({
            "active_attack": state.traffic_gen.active_attack,
            "total_fl_rounds": len(state.server.history),
            "global_model": {
                "val_accuracy": round(val_acc * 100, 2),
                "val_loss": round(val_loss, 4),
                "val_f1": round(val_f1 * 100, 2)
            },
            "privacy": privacy_info,
            "devices": [dev.to_dict() for dev in state.devices.values()],
            "mitigation_summary": state.mitigation.get_summary(),
            "auto_fl_enabled": state.auto_fl_enabled
        })


@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    with state.lock:
        return jsonify({
            "history": state.telemetry_history,
            "latest_flows": state.latest_live_flows
        })


@app.route("/api/attack/start", methods=["POST"])
def start_attack():
    data = request.json or {}
    attack_type = data.get("attack_type", "syn_flood")
    target_device = data.get("target_device_id", "smart_gateway_01")
    intensity = float(data.get("intensity", 1.0))

    with state.lock:
        try:
            attack_meta = state.traffic_gen.start_attack(attack_type, target_device, intensity)
            return jsonify({"status": "success", "attack": attack_meta})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/attack/stop", methods=["POST"])
def stop_attack():
    with state.lock:
        stopped = state.traffic_gen.stop_attack()
        return jsonify({"status": "success", "stopped": stopped})


@app.route("/api/fl/round", methods=["POST"])
def run_fl_round():
    data = request.json or {}
    clipping_norm = float(data.get("clipping_norm", 1.0))
    noise_multiplier = float(data.get("noise_multiplier", 0.5))
    local_epochs = int(data.get("local_epochs", 2))
    lr = float(data.get("lr", 0.01))

    with state.lock:
        try:
            summary = state.server.run_round(
                clients=state.clients,
                clipping_norm=clipping_norm,
                noise_multiplier=noise_multiplier,
                local_epochs=local_epochs,
                lr=lr
            )
            return jsonify({"status": "success", "round_summary": summary})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/fl/train_batch", methods=["POST"])
def run_fl_batch():
    data = request.json or {}
    rounds_count = int(data.get("rounds", 3))
    clipping_norm = float(data.get("clipping_norm", 1.0))
    noise_multiplier = float(data.get("noise_multiplier", 0.25))

    summaries = []
    with state.lock:
        for _ in range(min(rounds_count, 10)):
            s = state.server.run_round(
                clients=state.clients,
                clipping_norm=clipping_norm,
                noise_multiplier=noise_multiplier,
                local_epochs=1,
                lr=0.01
            )
            summaries.append(s)

    return jsonify({"status": "success", "rounds_completed": summaries})


@app.route("/api/fl/toggle_auto", methods=["POST"])
def toggle_auto_fl():
    with state.lock:
        state.auto_fl_enabled = not state.auto_fl_enabled
        return jsonify({"status": "success", "auto_fl_enabled": state.auto_fl_enabled})


@app.route("/api/fl/history", methods=["GET"])
def get_fl_history():
    with state.lock:
        return jsonify({
            "history": state.server.history,
            "privacy_history": state.server.accountant.history
        })


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    with state.lock:
        return jsonify(state.mitigation.get_summary())


@app.route("/api/mitigation/reset", methods=["POST"])
def reset_mitigation():
    with state.lock:
        state.mitigation.reset_all_mitigations(state.devices)
        return jsonify({"status": "success", "message": "Mitigations and firewall rules reset."})


if __name__ == "__main__":
    # Pre-train initial 2 rounds so the model has basic detection capability out-of-the-box
    print("[*] Pre-training initial Federated Learning rounds with Differential Privacy...")
    for r in range(2):
        state.server.run_round(
            clients=state.clients,
            clipping_norm=1.0,
            noise_multiplier=0.4,
            local_epochs=2,
            lr=0.01
        )
    print(f"[+] Initialized! Validation Accuracy: {state.server.history[-1]['val_metrics']['accuracy']*100:.1f}%")
    print("[*] Starting Web Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
