---
date:
  created: 2025-12-07
readtime: 5
authors:
    - salmanabulatif
---

# SNR Blog

Welcome to the blog for the **Wi-Fi Proximity Sensor** project, where we transform a standard Wi-Fi adapter into a real-time proximity detector using machine learning.

<!-- more -->

# Wi-Fi Proximity Sensor with Machine Learning: A Hacker's Guide

**Hey hackers and pentesters!**  
Want to turn a cheap Wi-Fi adapter into a stealthy proximity sensor for physical security assessments or red team ops? This Python-based project leverages wireless signal metrics—signal strength, noise level, and Signal-to-Noise Ratio (SNR)—to estimate distances to nearby objects or people. Whether you're detecting movement in a target environment, building covert presence detection, or integrating with IoT for sneaky automation, this tool has you covered. It combines raw signal processing with a machine learning pipeline for precision and adaptability.

*The full source code will drop soon on [GitHub](https://github.com/salmanabulatif/SNRv0) (link coming soon).*

---

## Project Overview

This Wi-Fi Proximity Sensor runs on Linux, sniffing metrics from a Wi-Fi interface to estimate proximity. It supports two modes: static threshold-based detection for quick setups, and a machine learning mode that trains on custom data for environment-specific accuracy. Built with Python 3, `scikit-learn`, and `joblib`, it uses system tools (`iw`, `/proc/net/wireless`, `iwconfig`) to pull raw signal data.

For pentesters, this is a lightweight, hackable tool for:
- Physical security testing
- Rogue device detection
- Social engineering ops where proximity matters

---

## Key Features

- **Metric Sniffing:** Pulls signal strength and noise using multiple Linux tools for reliability across setups.
- **Proximity Estimation:** Maps SNR to proximity zones ("Very Close," "Normal Range," "Moving Away," "Far Away") or exact distances via ML.
- **Signal Smoothing:** Uses exponential moving average (EMA) to stabilize noisy Wi-Fi signals.
- **Machine Learning:** Collects data and trains regression (`LinearRegression`) or classification (`SVC`) models for custom environments.
- **Real-Time Monitoring:** Low-latency updates for dynamic scenarios like tailgating detection.
- **Pentesting Applications:** Think intrusion detection, physical access monitoring, or IoT exploitation.

---

## Technical Breakdown

The system is built for hackers who love to tinker.  
It operates in **four modes**:

1. **Monitoring Mode:** Real-time tracking of Wi-Fi metrics with proximity estimates using static thresholds or a trained model.
2. **Data Collection Mode:** Logs signal data with ground-truth distances or labels for training custom models.
3. **Model Training Mode:** Builds a `scikit-learn` pipeline to predict distances or proximity zones.
4. **Inference Mode:** Deploys the trained model for live, environment-tuned proximity detection.

---

### Signal Processing

The system scrapes Wi-Fi metrics using a tiered approach to handle diverse hardware:

- **`iw dev <interface> link`:** Modern, clean way to get signal strength (dBm).
- **`/proc/net/wireless`:** Kernel-level stats for signal and noise, great for older systems.
- **`iwconfig`:** Fallback for legacy setups, parsing signal/noise from text output.

**SNR Calculation:**

```
SNR = Signal Strength (dBm) - Noise Level (dBm)
```

If noise is unavailable (common in some adapters), it approximates:

```
SNR ≈ Signal Strength + 90
```

**Signal Smoothing (EMA, α = 0.7):**

```
SNR_smoothed_t = α · SNR_smoothed_{t-1} + (1-α) · SNR_raw_t
```

This keeps readings stable for reliable proximity estimates during pentests.

---

### Machine Learning Pipeline

For hackers, the ML component is where things get juicy. Feature vector includes:

- Signal strength (dBm, default -100 if missing)
- Noise level (dBm, default -90 if missing)
- Smoothed SNR (dB, default 0 if missing)

The system auto-detects label type:

- **Regression** (`LinearRegression`): Predicts distance (cm), mapped to zones:
  - <50 cm: "VERY CLOSE"
  - 50–200 cm: "NORMAL RANGE"
  - 200–400 cm: "MOVING AWAY"
  - >400 cm: "FAR AWAY"
- **Classification** (`SVC` with probability): Directly predicts proximity zones.

A `StandardScaler` normalizes features to handle varying signal ranges. Models are serialized with `joblib` for quick deployment.

Pentesters can train models in specific environments (e.g., target office) to account for walls, interference, or hardware quirks.

---

### Pentesting Applications

- **Physical Intrusion Detection:** Detect tailgating or unauthorized access by monitoring SNR changes near doors or secure areas.
- **Rogue Device Tracking:** Identify movement of Wi-Fi-enabled devices (e.g., laptops, phones) in restricted zones.
- **IoT Exploitation:** Integrate with MQTT or Home Assistant to trigger payloads when someone approaches.
- **Social Engineering:** Use proximity data to time physical access attempts or device interactions.

---

## Core Code (Preview)

Here’s a stripped-down version of the core logic, focusing on metric extraction and monitoring.  
The full code, with data collection and training, will be released soon.

```python
#!/usr/bin/env python3
import subprocess
import re
import time
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

INTERFACE = "wlan0"
SAMPLING_RATE = 0.5
SMOOTHING_FACTOR = 0.7
MODEL_FILE = "wifi_proximity_model.pkl"

def get_wireless_metrics(interface):
    metrics = {'signal': None, 'noise': None}
    try:
        output = subprocess.check_output(["iw", "dev", interface, "link"], text=True, stderr=subprocess.DEVNULL)
        sig_match = re.search(r"signal:\s*(-?\d+)\s*dBm", output)
        if sig_match:
            metrics['signal'] = int(sig_match.group(1))
    except Exception:
        try:
            with open("/proc/net/wireless", "r") as f:
                for line in f.readlines()[2:]:
                    if interface in line:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            metrics['signal'] = int(float(parts[3]))
                            metrics['noise'] = int(float(parts[4]))
                            break
        except Exception:
            pass
    return metrics

def calculate_snr(metrics):
    if metrics['signal'] is not None and metrics['noise'] is not None:
        return metrics['signal'] - metrics['noise']
    elif metrics['signal'] is not None:
        return metrics['signal'] + 90
    return None

def monitor(interface, model=None):
    smoothed_snr = None
    print(f"{'Time':<8} | {'Signal':>7} | {'SNR':>6} | {'Status':<25}")
    print("-" * 50)
    try:
        while True:
            timestamp = time.strftime("%H:%M:%S")
            metrics = get_wireless_metrics(interface)
            snr = calculate_snr(metrics)
            if snr is not None:
                smoothed_snr = snr if smoothed_snr is None else SMOOTHING_FACTOR * smoothed_snr + (1 - SMOOTHING_FACTOR) * snr
            signal_str = f"{metrics['signal']} dBm" if metrics['signal'] is not None else "N/A"
            snr_str = f"{smoothed_snr:.1f} dB" if smoothed_snr is not None else "N/A"
            status = "NO SIGNAL"
            if snr is not None:
                if model:
                    features = [[metrics['signal'] or -100, metrics['noise'] or -90, smoothed_snr or 0]]
                    pred = model.predict(features)[0]
                    status = f"{pred:.1f} cm" if isinstance(pred, float) else pred
                else:
                    status = ("VERY CLOSE" if smoothed_snr < 26 else
                              "NORMAL RANGE" if smoothed_snr < 33 else
                              "MOVING AWAY" if smoothed_snr < 40 else
                              "FAR AWAY")
            print(f"{timestamp:<8} | {signal_str:>7} | {snr_str:>6} | {status:<25}", end="\r")
            time.sleep(SAMPLING_RATE)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
```

This code is robust, with fallbacks for metric extraction and support for both static and ML-based detection.  
Hackers can extend it to log data to a remote server or trigger scripts on proximity events.

---

## Setup and Usage

### Requirements

- **OS:** Linux (Kali, Ubuntu, etc.) with a Wi-Fi adapter (monitor mode not required but useful).
- **Software:** Python 3.6+, `scikit-learn`, `joblib`, `numpy`.
- **Tools:** `iw`, `iwconfig`, and read access to `/proc/net/wireless`.

Install dependencies:

```bash
pip install scikit-learn joblib numpy
```

Check your Wi-Fi interface:

```bash
ip link show
```

### Commands

- **Monitor Mode:**
  ```bash
  python3 wifi_proximity.py --interface wlan0
  ```
  Uses `wifi_proximity_model.pkl` if present; falls back to static thresholds.

- **Data Collection Mode:**
  ```bash
  python3 wifi_proximity.py --collect --interface wlan0
  ```
  Input distances (cm) or labels (e.g., "VERY CLOSE"). Saves to `wifi_prox_data.csv`.

- **Model Training Mode:**
  ```bash
  python3 wifi_proximity.py --train
  ```
  Trains and saves the model to `wifi_proximity_model.pkl`.

### Sample Output

Monitoring output looks like:

```
Time     | Signal  | SNR    | Status
------------------------------------------------
12:34:56 | -50 dBm | 40.0 dB | FAR AWAY
12:34:57 | -48 dBm | 42.0 dB | FAR AWAY
```

With an ML model, you might see distances (e.g., "150.2 cm") for regression.

---

## Hacking Tips and Tricks

- **Spoofing Defense:** Wi-Fi signals can be manipulated (e.g., via signal amplifiers). Train models in the target environment to detect anomalies in SNR patterns.
- **Stealth Mode:** Run in a Docker container or Raspberry Pi for covert deployment. Redirect output to a log file or C2 server with `tee` or `nc`.
- **Model Evasion:** Test model robustness by introducing noise (e.g., using `aircrack-ng` to simulate interference). Retrain with adversarial data to harden the model.
- **Integration:** Pipe proximity events to `metasploit` or `nmap` scripts for automated pentesting workflows.
- **Hardware Mods:** Use high-gain antennas to extend range or directional antennas for precise tracking.

---

## Challenges and Limitations

- **Signal Noise:** Multipath effects, walls, and interference can skew SNR. Train models with diverse data to compensate.
- **Hardware Variability:** Some adapters don’t report noise, forcing reliance on the fallback SNR formula. Test on target hardware.
- **ML Dependence:** Model accuracy hinges on training data quality. Collect data in realistic conditions (e.g., during a recon phase).
- **Latency:** The 0.5s sampling rate may miss rapid movements. Tune `SAMPLING_RATE` for faster response at the cost of stability.

---

## Future Enhancements for Pentesters

- **Multi-Adapter Support:** Combine multiple Wi-Fi interfaces for triangulation or redundancy.
- **Packet Injection:** Integrate with `scapy` to correlate proximity with specific devices (MAC addresses).
- **C2 Integration:** Stream data to a command-and-control server for remote monitoring.
- **ML Upgrades:** Swap `LinearRegression`/`SVC` for XGBoost or neural nets for better accuracy.
- **Web Dashboard:** Build a Flask-based UI for real-time visualization during engagements.

---

## Conclusion

This Wi-Fi Proximity Sensor is a hacker’s dream for physical security testing.  
It’s lightweight, customizable, and perfect for detecting movement, triggering payloads, or enhancing red team ops. Whether you’re sneaking through a facility or building a stealthy IoT trap, this tool delivers.

*The full code will hit [GitHub](#) (link coming soon), so stay tuned. Grab a Wi-Fi adapter, fire up Kali, and start hacking proximity like a pro!*
