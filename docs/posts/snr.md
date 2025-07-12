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

# Wi-Fi Proximity Sensing with Machine Learning

Welcome to the blog for the **Wi-Fi Proximity Sensor** project, where we transform a standard Wi-Fi adapter into a real-time proximity detector using machine learning. In this post, you'll learn how wireless signal metrics (like SNR) can be used for presence detection, and how adding data-driven intelligence unlocks robust, adaptive performance for smart environments.

---

## Table of Contents

- [Overview](#overview)
- [Motivation & Applications](#motivation-applications)
- [How It Works](#how-it-works)
- [Adding Machine Learning](#adding-machine-learning)
- [Step-by-Step Guide](#step-by-step-guide)
- [Sample Code](#sample-code)
- [Results & Improvements](#results--improvements)
- [Next Steps](#next-steps)
- [References](#references)

---

## Overview

This project uses a standard Wi-Fi adaptor as a proximity sensor by monitoring wireless signal strength, noise levels, and Signal-to-Noise Ratio (SNR). The sensor estimates how close or far objects are by analyzing how wireless metrics change over time.

While traditional approaches use static thresholds (e.g., "SNR less than X means close"), we leverage **machine learning** to adapt to different environments and hardware, improving accuracy and robustness.

---

## Motivation & Applications

Why turn a Wi-Fi adaptor into a proximity sensor?

- **Presence detection**: Know when someone enters or leaves a room.
- **Smart automation**: Trigger IoT devices based on proximity.
- **Security**: Alert when unexpected movement is detected.
- **Robotics**: Enable robots to sense nearby obstacles using wireless signals.

Traditional sensors (infrared, ultrasonic) require dedicated hardware. Wi-Fi metrics are available on most devices and can be repurposed for sensing, often with zero extra hardware cost.

---

## How It Works

1. **Read Wi-Fi Metrics**:  
   The script monitors signal (`dBm`), noise (`dBm`), and calculates SNR.
2. **Estimate Distance**:  
   A simple formula estimates distance from SNR (higher SNR = closer).
3. **Status Classification**:  
   Static logic classifies readings into zones: "Very Close", "Normal Range", "Far Away", etc.
4. **Live Monitoring**:  
   The script prints proximity status in real-time.

### Limitations of Static Thresholds

- Every environment (walls, furniture, interference) changes the relationship between SNR and actual distance.
- Hardware differences affect readings.
- Static logic struggles in noisy or dynamic settings.

---

## Adding Machine Learning

To solve these limitations, we introduce a **machine learning model**:

1. **Data Collection Mode**:  
   Collect signal, noise, SNR, and your true distance (or label) at each reading.
2. **Model Training Mode**:  
   Train a regression or classification model using `scikit-learn` for your specific environment/hardware.
3. **Inference Mode**:  
   The script uses your trained model for smarter, adaptive proximity prediction.

### Benefits

- Adapts to your unique hardware and environment.
- Learns from real, labeled data.
- Can predict actual distances or custom proximity zones.

---

## Step-by-Step Guide

### 1. Install Requirements

```sh
pip install scikit-learn joblib mkdocs
```

### 2. Collect Training Data

Run the script in data collection mode:

```sh
python3 codev0_ml.py --collect
```

For each sample, enter the true distance (in cm) or a label (e.g., "VERY CLOSE") at the prompt. Try different locations and distances for variety!

### 3. Train the Machine Learning Model

```sh
python3 codev0_ml.py --train
```

This creates a model file (`wifi_proximity_model.pkl`).

### 4. Run in Monitoring Mode

```sh
python3 codev0_ml.py
```

If a trained model is found, it will use ML for proximity prediction. Otherwise, it falls back to static logic.

---

## Sample Code

Here’s a simplified snippet for the ML-enhanced proximity sensor:

```python
def get_features(metrics, snr):
    return [
        metrics['signal'] if metrics['signal'] is not None else -100,
        metrics['noise'] if metrics['noise'] is not None else -90,
        snr if snr is not None else 0
    ]

def predict_proximity(model, metrics, snr):
    features = [get_features(metrics, snr)]
    pred = model.predict(features)[0]
    if isinstance(pred, float):
        # Regression: map distance to status
        if pred < 50:
            status = "VERY CLOSE (<50 cm)"
        elif pred < 200:
            status = "NORMAL RANGE (0.5-2 m)"
        elif pred < 400:
            status = "MOVING AWAY (2-4 m)"
        else:
            status = "FAR AWAY (>4 m)"
        return status, f"{pred:.1f} cm"
    else:
        return pred, None
```

See the full code in an upcoming version on github.

---

## Results & Improvements

### What You Get

- **Real-time proximity estimates** via Wi-Fi metrics.
- **Customizable accuracy** for your environment.
- **Adaptability** to any Wi-Fi hardware.

### How to Improve Further

- Collect more data for better accuracy.
- Try advanced ML models (Random Forest, Neural Network).
- Add more features (channel frequency, historical data).
- Integrate with smart home automations or robotics.

---

## Next Steps

- Package as a [Python module](https://packaging.python.org/tutorials/packaging-projects/).
- Integrate with [Home Assistant](https://www.home-assistant.io/).
- Visualize data with [Grafana](https://grafana.com/) or [Plotly](https://plotly.com/).
- Explore other wireless metrics (Bluetooth, Zigbee).

---

## References

- [Scikit-learn documentation](https://scikit-learn.org/)
- [joblib documentation](https://joblib.readthedocs.io/)
- [Wireless metrics on Linux](https://wireless.wiki.kernel.org/)
- [Presence detection with Wi-Fi](https://www.sciencedirect.com/science/article/pii/S1570870516307057)

---

## Conclusion

With just a Wi-Fi adaptor and some data, you can build a smart proximity sensor powered by machine learning. This approach is flexible, affordable, and adaptable to countless smart environments. Happy hacking!
