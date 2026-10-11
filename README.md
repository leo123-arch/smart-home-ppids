# Smart Home Privacy-Preserving Intrusion Detection System (PP-IDS)
### Distributed DDoS Detection via Federated Learning and Differential Privacy

---

## 📌 Executive Summary & Motivation

Smart home IoT devices (smart cameras, central gateways, smart locks, connected thermostats) are increasingly targeted by volumetric and stealthy **Distributed Denial of Service (DDoS)** attacks and botnets (e.g., Mirai, Bashlite). 

Traditional Intrusion Detection Systems (IDS) rely on centralizing network packet captures (PCAP) and flow telemetry in a remote cloud server. In a smart home, this creates severe privacy violations:
- **Home Occupancy Leakage**: Intermittent device activity reveals when residents wake up, leave, or sleep.
- **Surveillance Eavesdropping**: Unencrypted or metadata-rich camera flows expose video streaming habits.
- **Credential & Behavioral Exposure**: Centralized servers become high-value targets for data breaches.

**Smart Home PP-IDS** solves this privacy dilemma:
1. **Raw Telemetry Stays Local**: IoT edge nodes train neural networks exclusively on their private local network telemetry.
2. **Differential Privacy Guarantee**: Local weight updates are clipped to bounded $L_2$ norm ($C$) and perturbed with calibrated Gaussian noise ($\sigma$) before transmission, mathematically bounding privacy loss $(\varepsilon, \delta)$ and preventing model inversion attacks.
3. **Federated Averaging (FedAvg)**: The central home gateway/server aggregates only privatized weight updates to create a robust global DDoS detection model.
4. **Automated Edge Mitigation**: Real-time inference classifies flows and triggers automated defenses (dynamic firewall drop rules, adaptive bandwidth rate-limiting, and IoT VLAN quarantine).

---

## 🏗️ Architecture & Privacy Pipeline

```
  +-------------------------------------------------------------+
  |                   Smart Home IoT Edge Fleet                 |
  |                                                             |
  | [4K Security Cam]   [Smart Gateway]   [Climate Thermostat]  |
  |   (Private Flows)     (Private Flows)    (Private Flows)    |
  |         |                    |                  |           |
  |         v                    v                  v           |
  |  +--------------+    +--------------+    +--------------+   |
  |  | Local Train  |    | Local Train  |    | Local Train  |   |
  |  |  Loss & Grad |    |  Loss & Grad |    |  Loss & Grad |   |
  |  +--------------+    +--------------+    +--------------+   |
  |         |                    |                  |           |
  |         v                    v                  v           |
  |  +--------------+    +--------------+    +--------------+   |
  |  | L2 Clipping  |    | L2 Clipping  |    | L2 Clipping  |   |
  |  | (Norm <= C)  |    | (Norm <= C)  |    | (Norm <= C)  |   |
  |  +--------------+    +--------------+    +--------------+   |
  |         |                    |                  |           |
  |         v                    v                  v           |
  |  +--------------+    +--------------+    +--------------+   |
  |  | Gaussian DP  |    | Gaussian DP  |    | Gaussian DP  |   |
  |  | Noise (+ N)  |    | Noise (+ N)  |    | Noise (+ N)  |   |
  |  +--------------+    +--------------+    +--------------+   |
  +---------|--------------------|------------------|-----------+
            |                    |                  |
            +-------------->     |     <------------+
                                 v
                 +-------------------------------+
                 |  Central Federated Aggregator |
                 |       FedAvg: W_t+1 = Σ w_i   |
                 |   Privacy Accountant (ε, δ)   |
                 +-------------------------------+
                                 |
                                 v  Broadcast Global Model
                 +-------------------------------+
                 |   Edge Real-Time Inference    |
                 |    & Dynamic Threat Defense   |
                 |  - Dynamic Firewall IP Drops  |
                 |  - Bandwidth Rate Limiting    |
                 |  - Device VLAN Quarantine     |
                 +-------------------------------+
```

---

## 🔬 Mathematical Foundations of Differential Privacy

Each edge node computes local weight deltas $\Delta W = W_{local} - W_{global}$.

### 1. $L_2$ Norm Sensitivity Bounding
$$\Delta W_{clipped} = \frac{\Delta W}{\max\left(1, \frac{\|\Delta W\|_2}{C}\right)}$$
where $C$ is the clipping threshold, ensuring bounded sensitivity.

### 2. Calibrated Gaussian Noise Perturbation
$$\tilde{\Delta W} = \Delta W_{clipped} + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right)$$
where $\sigma$ is the noise multiplier.

### 3. $(\varepsilon, \delta)$-Privacy Accounting
Under advanced composition over $T$ communication rounds:
$$\varepsilon \approx \frac{q \sqrt{2 T \ln(1/\delta)}}{\sigma} + \frac{T q^2}{2 \sigma^2}$$
Our real-time accountant computes exact accumulated privacy loss $\varepsilon$ with a target failure probability $\delta = 10^{-5}$.

---

## ⚡ Simulated DDoS Attack Vectors

1. **TCP SYN Flood**: Volumetric attack overwhelming transmission buffers with spoofed SYN packets and zero ACKs.
2. **UDP Amplification**: Exploiting reflection protocols (DNS/NTP) with high bandwidth payloads.
3. **Layer-7 HTTP Flood**: Application-layer resource exhaustion targeting web and API ports.
4. **Mirai Botnet Burst**: Multi-vector infection burst scanning random ports followed by coordinated volumetric flooding.

---

## 🚀 Quickstart & Execution

### 1. Requirements
Python 3.10+ with `torch`, `flask`, `numpy`, `scikit-learn` installed.

### 2. Run the Full Interactive System
From the project directory:
```bash
python run_demo.py
```
This script will:
- Initialize the smart home edge nodes and synthetic datasets.
- Pre-train initial baseline federated learning rounds with differential privacy.
- Start the Flask backend server on `http://127.0.0.1:5000`.
- Automatically open the interactive cybersecurity dashboard in your default browser.

### 3. Run Unit Tests
```bash
python -m unittest discover tests
```
Runs tests verifying the neural network, LayerNorm invariance, $L_2$ clipping, Gaussian mechanism calibration, and FedAvg convergence.

---

## 🖥️ Interactive Dashboard Features

- **Live Device Fleet Monitor**: Real-time traffic metrics (pps, byte rate, SYN/UDP ratio, anomaly confidence) for all 5 smart devices.
- **Adversarial Attack Simulator**: Inject SYN floods, UDP amplification, HTTP floods, or Mirai botnet bursts on target devices with adjustable intensity.
- **Privacy-Preserving FL Studio**: Adjust noise multiplier ($\sigma$) and clipping norm ($C$), run single or multi-round training, or toggle continuous auto-training.
- **Telemetry & Convergence Charts**: Real-time streaming charts for packet rates, DDoS anomaly confidence, and model accuracy vs. privacy budget ($\varepsilon$).
- **Automated Incident Mitigation Feed**: Live feed of detected threats and applied countermeasures with dynamic firewall drop rules.
