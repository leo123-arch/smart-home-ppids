// Dashboard Client Controller for Smart Home PP-IDS

let trafficChart = null;
let ddosProbChart = null;
let flConvergenceChart = null;

const DEVICE_COLORS = {
  "smart_gateway_01": "#3b82f6",     // Blue
  "smart_cam_01": "#06b6d4",         // Cyan
  "smart_thermostat_01": "#10b981",  // Emerald
  "smart_lock_01": "#f59e0b",        // Amber
  "smart_speaker_01": "#8b5cf6"      // Violet
};

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  initCharts();
  fetchInitialData();
  
  // Continuous polling loop every 1.5 seconds
  setInterval(refreshDashboard, 1500);
});

function setupEventListeners() {
  const intensityRange = document.getElementById("attackIntensityRange");
  const intensityVal = document.getElementById("intensityVal");
  if (intensityRange && intensityVal) {
    intensityRange.addEventListener("input", (e) => {
      intensityVal.innerText = `${e.target.value}x`;
    });
  }

  // Attack buttons
  document.getElementById("btnStartAttack").addEventListener("click", triggerAttack);
  document.getElementById("btnStopAttack").addEventListener("click", stopAttack);

  // FL Training buttons
  document.getElementById("btnTrainOneRound").addEventListener("click", () => trainFLRound(1));
  document.getElementById("btnTrainFiveRounds").addEventListener("click", () => trainFLRound(5));
  document.getElementById("btnToggleAutoTrain").addEventListener("click", toggleAutoTrain);

  // Mitigation reset button
  document.getElementById("btnResetMitigations").addEventListener("click", resetMitigations);
}

// Chart Initializations
function initCharts() {
  if (typeof Chart === "undefined") {
    console.warn("Chart.js not loaded. Visual charts will be unavailable.");
    return;
  }

  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: {
      legend: { labels: { color: "#9ca3af", font: { size: 11 } } }
    },
    scales: {
      x: {
        ticks: { color: "#6b7280", font: { size: 10 } },
        grid: { color: "rgba(75, 85, 99, 0.15)" }
      },
      y: {
        ticks: { color: "#6b7280", font: { size: 10 } },
        grid: { color: "rgba(75, 85, 99, 0.15)" }
      }
    }
  };

  // 1. Ingress Traffic Chart
  const ctxTraffic = document.getElementById("trafficChart").getContext("2d");
  trafficChart = new Chart(ctxTraffic, {
    type: "line",
    data: {
      labels: [],
      datasets: []
    },
    options: {
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        y: {
          ...commonOptions.scales.y,
          title: { display: true, text: "Packets / sec", color: "#9ca3af" },
          beginAtZero: true
        }
      }
    }
  });

  // 2. DDoS Probability Chart
  const ctxDDoS = document.getElementById("ddosProbChart").getContext("2d");
  ddosProbChart = new Chart(ctxDDoS, {
    type: "line",
    data: {
      labels: [],
      datasets: []
    },
    options: {
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        y: {
          ...commonOptions.scales.y,
          min: 0,
          max: 100,
          title: { display: true, text: "DDoS Prob (%)", color: "#9ca3af" }
        }
      }
    }
  });

  // 3. FL Convergence Chart
  const ctxFL = document.getElementById("flConvergenceChart").getContext("2d");
  flConvergenceChart = new Chart(ctxFL, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Global Test Accuracy (%)",
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.15)",
          borderWidth: 2,
          yAxisID: "yAcc",
          data: [],
          tension: 0.3
        },
        {
          label: "Privacy Loss \u03B5 (Epsilon)",
          borderColor: "#a78bfa",
          backgroundColor: "rgba(167, 139, 250, 0.15)",
          borderWidth: 2,
          borderDash: [5, 5],
          yAxisID: "yEps",
          data: [],
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#9ca3af", font: { size: 11 } } }
      },
      scales: {
        x: {
          title: { display: true, text: "Communication Round", color: "#9ca3af" },
          ticks: { color: "#6b7280" },
          grid: { color: "rgba(75, 85, 99, 0.15)" }
        },
        yAcc: {
          position: "left",
          min: 0,
          max: 100,
          title: { display: true, text: "Accuracy %", color: "#10b981" },
          ticks: { color: "#10b981" },
          grid: { color: "rgba(75, 85, 99, 0.15)" }
        },
        yEps: {
          position: "right",
          beginAtZero: true,
          title: { display: true, text: "\u03B5 Privacy Loss", color: "#a78bfa" },
          ticks: { color: "#a78bfa" },
          grid: { drawOnChartArea: false }
        }
      }
    }
  });
}

// Fetch Initial Data
async function fetchInitialData() {
  await refreshDashboard();
  await refreshFLHistory();
}

// Main Polling Refresh
async function refreshDashboard() {
  try {
    const [statusRes, telemetryRes, alertsRes] = await Promise.all([
      fetch("/api/status"),
      fetch("/api/telemetry"),
      fetch("/api/alerts")
    ]);

    const statusData = await statusRes.json();
    const telemetryData = await telemetryRes.json();
    const alertsData = await alertsRes.json();

    updateStatusUI(statusData);
    updateDevicesUI(statusData.devices, telemetryData.latest_flows);
    updateTelemetryCharts(telemetryData.history);
    updateAlertsUI(alertsData);
  } catch (err) {
    console.error("Dashboard refresh error:", err);
  }
}

// Update Top Bar & Metrics
function updateStatusUI(data) {
  // Global Model Metrics
  const acc = data.global_model.val_accuracy;
  const f1 = data.global_model.val_f1;
  document.getElementById("metricAccuracy").innerText = `${acc}%`;
  document.getElementById("metricF1").innerText = `F1-Score: ${f1}%`;

  // Privacy Budget
  const eps = data.privacy.cumulative_epsilon;
  const tier = data.privacy.privacy_tier;
  const epsStr = typeof eps === "number" ? `\u03B5 = ${eps.toFixed(2)}` : eps;
  document.getElementById("metricEpsilon").innerText = epsStr;
  document.getElementById("metricPrivacyTier").innerText = `Guarantee: ${tier}`;
  document.getElementById("headerEpsilonText").innerText = epsStr;
  document.getElementById("privacyTierBadge").innerText = tier;

  // FL Rounds
  document.getElementById("metricRounds").innerText = `#${data.total_fl_rounds}`;
  document.getElementById("headerRoundText").innerText = `#${data.total_fl_rounds}`;

  // Mitigations
  document.getElementById("metricMitigations").innerText = data.mitigation_summary.active_rules_count;
  document.getElementById("metricAlertCount").innerText = `${data.mitigation_summary.total_alerts} Total Incidents`;

  // Attack Status Pill
  const attackPill = document.getElementById("attackStatusPill");
  const attackText = document.getElementById("attackStatusText");

  if (data.active_attack) {
    attackPill.className = "status-pill alert";
    const atkName = data.active_attack.type.toUpperCase().replace("_", " ");
    attackText.innerText = `ATTACK ACTIVE: ${atkName}`;
  } else {
    attackPill.className = "status-pill safe";
    attackText.innerText = "NETWORK SECURE";
  }

  // Auto-training button text
  const autoText = document.getElementById("autoTrainText");
  const autoIcon = document.getElementById("autoTrainIcon");
  if (data.auto_fl_enabled) {
    autoText.innerText = "Disable Auto-Training (Active)";
    autoIcon.innerText = "\u23F8";
  } else {
    autoText.innerText = "Enable Continuous Auto-Training";
    autoIcon.innerText = "\u25B6";
  }
}

// Update Smart Device Cards
function updateDevicesUI(devices, latestFlows) {
  const container = document.getElementById("devicesContainer");
  if (!container || !devices) return;

  container.innerHTML = devices.map(dev => {
    const flow = latestFlows && latestFlows[dev.device_id] ? latestFlows[dev.device_id] : null;
    const pktRate = flow ? Math.round(flow.packet_rate) : 0;
    const byteRate = flow ? (flow.byte_rate / 1024).toFixed(1) : "0.0";
    const ddosProbPct = Math.round(dev.last_ddos_prob * 100);

    // Color fill based on probability
    let probColor = "#10b981"; // green
    if (ddosProbPct >= 80) probColor = "#f43f5e"; // red
    else if (ddosProbPct >= 40) probColor = "#f59e0b"; // amber

    // Status class
    let cardClass = "device-card";
    let badgeClass = "device-badge normal";
    let statusText = "NORMAL";

    if (dev.status === "under_attack") {
      cardClass += " attacked";
      badgeClass = "device-badge attacked";
      statusText = "UNDER ATTACK";
    } else if (dev.status === "mitigated") {
      cardClass += " mitigated";
      badgeClass = "device-badge mitigated";
      statusText = "RATE LIMITED";
    } else if (dev.status === "quarantined") {
      cardClass += " quarantined";
      badgeClass = "device-badge quarantined";
      statusText = "QUARANTINED";
    }

    return `
      <div class="${cardClass}">
        <div class="device-top">
          <div class="device-info">
            <h3>${dev.name}</h3>
            <div class="device-ip">${dev.ip_address}</div>
          </div>
          <span class="${badgeClass}">${statusText}</span>
        </div>

        <div class="device-stat-row">
          <span>Traffic Rate:</span>
          <span class="stat-val">${pktRate} pps (${byteRate} KB/s)</span>
        </div>

        <div class="device-stat-row">
          <span>SYN / UDP Ratio:</span>
          <span class="stat-val">${flow ? (flow.syn_ratio * 100).toFixed(0) : 0}% / ${flow ? (flow.udp_ratio * 100).toFixed(0) : 0}%</span>
        </div>

        <div>
          <div class="prob-meter-label">
            <span>DDoS Anomaly Confidence</span>
            <span style="font-weight:600; color:${probColor};">${ddosProbPct}%</span>
          </div>
          <div class="prob-meter-bg">
            <div class="prob-meter-fill" style="width: ${ddosProbPct}%; background-color: ${probColor};"></div>
          </div>
        </div>
      </div>
    `;
  }).join("");
}

// Update Real-time Time-series Charts
function updateTelemetryCharts(history) {
  if (!history || history.length === 0 || !trafficChart || !ddosProbChart) return;

  const timestamps = history.map(h => h.timestamp);
  const sampleEntry = history[0].devices;
  const deviceIds = Object.keys(sampleEntry);

  // 1. Packets / Sec Chart
  trafficChart.data.labels = timestamps;
  trafficChart.data.datasets = deviceIds.map(devId => ({
    label: devId.replace("smart_", "").replace("_01", "").toUpperCase(),
    borderColor: DEVICE_COLORS[devId] || "#9ca3af",
    backgroundColor: "transparent",
    borderWidth: 2,
    pointRadius: 1,
    tension: 0.2,
    data: history.map(h => h.devices[devId]?.packet_rate || 0)
  }));
  trafficChart.update("none");

  // 2. DDoS Probability Chart
  ddosProbChart.data.labels = timestamps;
  ddosProbChart.data.datasets = deviceIds.map(devId => ({
    label: devId.replace("smart_", "").replace("_01", "").toUpperCase(),
    borderColor: DEVICE_COLORS[devId] || "#9ca3af",
    backgroundColor: "transparent",
    borderWidth: 2,
    pointRadius: 1,
    tension: 0.2,
    data: history.map(h => ((h.devices[devId]?.ddos_prob || 0) * 100))
  }));
  ddosProbChart.update("none");
}

// Update FL Convergence Chart
async function refreshFLHistory() {
  try {
    const res = await fetch("/api/fl/history");
    const data = await res.json();
    if (!flConvergenceChart || !data.history) return;

    const labels = data.history.map(h => `R${h.round}`);
    const accuracies = data.history.map(h => (h.val_metrics?.accuracy || 0) * 100);
    const epsilons = data.history.map(h => h.epsilon);

    flConvergenceChart.data.labels = labels;
    flConvergenceChart.data.datasets[0].data = accuracies;
    flConvergenceChart.data.datasets[1].data = epsilons;
    flConvergenceChart.update();
  } catch (err) {
    console.error("FL History error:", err);
  }
}

// Update Security Alert Feed
function updateAlertsUI(alertsData) {
  const container = document.getElementById("alertsList");
  if (!container) return;

  if (!alertsData.recent_alerts || alertsData.recent_alerts.length === 0) {
    container.innerHTML = '<div class="empty-state">No active DDoS anomalies detected. Network is healthy.</div>';
    return;
  }

  container.innerHTML = alertsData.recent_alerts.map(a => {
    const isQuarantine = (a.actions_applied || []).some(act => act.includes("Quarantined"));
    const itemClass = isQuarantine ? "alert-item quarantine" : "alert-item";

    return `
      <div class="${itemClass}">
        <div>
          <div style="font-weight:600; color:#fff;">
            [${a.severity}] ${a.threat_type} on ${a.device_name}
          </div>
          <div style="color:var(--text-secondary); font-size:0.75rem;">
            Prob: ${(a.ddos_probability * 100).toFixed(1)}% &bull; Action: ${(a.actions_applied || []).join("; ")}
          </div>
        </div>
        <div class="alert-time">${a.timestamp}</div>
      </div>
    `;
  }).join("");
}

// User Actions
async function triggerAttack() {
  const attackType = document.getElementById("attackTypeSelect").value;
  const targetDevice = document.getElementById("targetDeviceSelect").value;
  const intensity = parseFloat(document.getElementById("attackIntensityRange").value);

  try {
    await fetch("/api/attack/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        attack_type: attackType,
        target_device_id: targetDevice,
        intensity: intensity
      })
    });
    refreshDashboard();
  } catch (err) {
    console.error("Failed to start attack:", err);
  }
}

async function stopAttack() {
  try {
    await fetch("/api/attack/stop", { method: "POST" });
    refreshDashboard();
  } catch (err) {
    console.error("Failed to stop attack:", err);
  }
}

async function trainFLRound(rounds = 1) {
  const noise = parseFloat(document.getElementById("dpNoiseInput").value) || 0.25;
  const clip = parseFloat(document.getElementById("dpClipInput").value) || 1.0;

  try {
    if (rounds === 1) {
      await fetch("/api/fl/round", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clipping_norm: clip,
          noise_multiplier: noise,
          local_epochs: 1,
          lr: 0.01
        })
      });
    } else {
      await fetch("/api/fl/train_batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rounds: rounds,
          clipping_norm: clip,
          noise_multiplier: noise
        })
      });
    }

    await refreshDashboard();
    await refreshFLHistory();
  } catch (err) {
    console.error("FL training error:", err);
  }
}

async function toggleAutoTrain() {
  try {
    await fetch("/api/fl/toggle_auto", { method: "POST" });
    refreshDashboard();
  } catch (err) {
    console.error("Auto train toggle error:", err);
  }
}

async function resetMitigations() {
  try {
    await fetch("/api/mitigation/reset", { method: "POST" });
    refreshDashboard();
  } catch (err) {
    console.error("Reset mitigation error:", err);
  }
}
