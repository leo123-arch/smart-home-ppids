"""Single-Click Demo Launcher for Smart Home PP-IDS.

Starts the simulation and security dashboard, then opens your web browser.
"""

import sys
import os
import time
import webbrowser
import threading

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web.app import app, state


def open_browser():
    """Waits for server initialization then opens browser."""
    time.sleep(1.8)
    url = "http://127.0.0.1:5000"
    print(f"\n[+] Opening PP-IDS Dashboard in your browser: {url}")
    webbrowser.open(url)


def main():
    print("=" * 70)
    print("   SMART HOME PRIVACY-PRESERVING INTRUSION DETECTION SYSTEM (PP-IDS)")
    print("       Federated Learning (FedAvg) + Differential Privacy (DP-SGD)")
    print("                      DDoS Threat Mitigation Engine")
    print("=" * 70)
    print("\n[*] Initializing Smart Home IoT Edge Nodes...")
    for dev_id, dev in state.devices.items():
        print(f"    - {dev.name} ({dev.ip_address}) [{dev.device_type.upper()}]")

    print("\n[*] Pre-training initial baseline model via Privacy-Preserving FL...")
    for r in range(2):
        res = state.server.run_round(
            clients=state.clients,
            clipping_norm=1.0,
            noise_multiplier=0.25,
            local_epochs=2,
            lr=0.01
        )
        acc = res["val_metrics"]["accuracy"] * 100
        eps = res["epsilon"]
        print(f"    Round #{res['round']} Complete -> Test Acc: {acc:.1f}% | Privacy Budget: \u03B5 = {eps:.2f}")

    # Launch browser thread
    threading.Thread(target=open_browser, daemon=True).start()

    print("\n[*] Starting Dashboard Web Server on http://127.0.0.1:5000")
    print("[*] Press Ctrl+C in terminal to stop.")
    print("=" * 70 + "\n")

    try:
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[!] Shutting down Smart Home PP-IDS...")
        state.running = False


if __name__ == "__main__":
    main()
