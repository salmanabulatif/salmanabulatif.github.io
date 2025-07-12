#!/usr/bin/env python3
import os
import re
import time
import subprocess
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration
INTERFACE = "wlan0"  # Set to your Wi-Fi interface
SAMPLING_RATE = 0.5
SMOOTHING_FACTOR = 0.7
MODEL_FILE = "wifi_proximity_model.pkl"
DATA_FILE = "wifi_prox_data.csv"
MIN_SAMPLES_PER_LABEL = 10  # Minimum samples per label for meaningful training
WINDOW_SIZE = 5  # Moving average window size

# Proximity labels - adjust based on your environment
PROXIMITY_LABELS = {
    "VERY_CLOSE": "VERY CLOSE (<1m)",
    "NEAR": "NEAR (1-3m)",
    "MID_RANGE": "MID-RANGE (3-5m)",
    "FAR_AWAY": "FAR AWAY (>5m)"
}

def get_wireless_metrics(interface):
    """Robustly fetch wireless metrics from multiple sources"""
    metrics = {'signal': None, 'noise': None}
    
    # Try iw dev <interface> link
    try:
        output = subprocess.check_output(
            ["iw", "dev", interface, "link"],
            stderr=subprocess.DEVNULL,
            text=True
        )
        sig_match = re.search(r"signal:\s*(-?\d+)\s*dBm", output)
        noise_match = re.search(r"noise:\s*(-?\d+)\s*dBm", output)
        if sig_match:
            metrics['signal'] = int(sig_match.group(1))
        if noise_match:
            metrics['noise'] = int(noise_match.group(1))
    except Exception:
        pass

    # Fallback to /proc/net/wireless
    if metrics['signal'] is None:
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

    # Final fallback to iwconfig
    if metrics['signal'] is None:
        try:
            output = subprocess.check_output(
                ["iwconfig", interface],
                stderr=subprocess.DEVNULL,
                text=True
            )
            sig_match = re.search(r"Signal [^=]+=(-?\d+) dBm", output)
            noise_match = re.search(r"Noise [^=]+=(-?\d+) dBm", output)
            if sig_match:
                metrics['signal'] = int(sig_match.group(1))
            if noise_match:
                metrics['noise'] = int(noise_match.group(1))
        except Exception:
            pass
            
    return metrics

def calculate_snr(metrics):
    """Calculate Signal-to-Noise Ratio with fallbacks"""
    if metrics['signal'] is not None and metrics['noise'] is not None:
        return metrics['signal'] - metrics['noise']
    elif metrics['signal'] is not None:
        # Estimate noise floor
        return metrics['signal'] + 90
    else:
        return None

def collect_data(interface, out_file=DATA_FILE):
    """Enhanced data collection with time-based averaging"""
    print("=== ADVANCED DATA COLLECTION MODE ===")
    print("Position yourself at specific distances and enter the corresponding label")
    print("Available labels:")
    for label in PROXIMITY_LABELS.values():
        print(f"  - {label}")
    print("\nPress Ctrl+C to stop collection\n")
    
    # Initialize CSV with header
    if not os.path.exists(out_file):
        with open(out_file, "w") as f:
            f.write("timestamp,signal,noise,snr,label\n")

    print(f"{'Time':<8} | {'Signal':>7} | {'Noise':>7} | {'SNR':>6} | {'Samples':>7} | {'Label':<20}")
    print("-" * 70)
    
    current_label = None
    sample_buffer = []
    
    try:
        while True:
            # Get new metrics
            metrics = get_wireless_metrics(interface)
            snr = calculate_snr(metrics)
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Display current reading
            signal = metrics['signal'] if metrics['signal'] is not None else "N/A"
            noise = metrics['noise'] if metrics['noise'] is not None else "N/A"
            snr_str = f"{snr:.1f}" if snr is not None else "N/A"
            sample_count = len(sample_buffer)
            
            print(f"{timestamp:<8} | {signal:>7} | {noise:>7} | {snr_str:>6} | {sample_count:>7} | ", end="")
            
            # Get user input
            user_input = input().strip().upper()
            
            # Process commands
            if user_input in ["Q", "QUIT", "EXIT"]:
                break
                
            # Process label input
            if user_input:
                # Validate label
                valid_label = False
                for key, value in PROXIMITY_LABELS.items():
                    if user_input in [key, value.upper()]:
                        current_label = key
                        valid_label = True
                        break
                
                if not valid_label:
                    print(f"Invalid label! Valid options: {', '.join(PROXIMITY_LABELS.keys())}")
                    continue
                    
                # Save buffer when changing label
                if sample_buffer:
                    save_samples(sample_buffer, current_label, out_file)
                    sample_buffer = []
            
            # Add to buffer
            if metrics['signal'] is not None:
                sample_buffer.append({
                    'timestamp': datetime.now().isoformat(),
                    'signal': metrics['signal'],
                    'noise': metrics['noise'] if metrics['noise'] is not None else -90,
                    'snr': snr if snr is not None else 0
                })
                
    except KeyboardInterrupt:
        if sample_buffer and current_label:
            save_samples(sample_buffer, current_label, out_file)
        print("\nData collection stopped.")

def save_samples(samples, label, out_file):
    """Save collected samples to file"""
    with open(out_file, "a") as f:
        for sample in samples:
            f.write(f"{sample['timestamp']},{sample['signal']},{sample['noise']},{sample['snr']},{label}\n")
    print(f"Saved {len(samples)} samples for label: {label}")

def train_model(data_file=DATA_FILE, model_file=MODEL_FILE):
    """Enhanced model training with validation and feature engineering"""
    print("=== ADVANCED MODEL TRAINING ===")
    
    # Load and validate data
    if not os.path.exists(data_file):
        print("Data file not found!")
        return
        
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    # Check data quality
    if df.empty:
        print("No data found in file!")
        return
        
    # Check for required columns
    required_cols = ['signal', 'snr', 'label']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Missing columns: {', '.join(missing)}")
        return
        
    # Check sample distribution
    label_counts = df['label'].value_counts()
    print("\nLabel distribution:")
    print(label_counts)
    
    if any(label_counts < MIN_SAMPLES_PER_LABEL):
        print(f"\nWarning: Some labels have fewer than {MIN_SAMPLES_PER_LABEL} samples!")
        print("Consider collecting more data for better model performance.")
    
    # Prepare features and labels
    X = df[['signal', 'snr']]
    y = df['label']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Create pipeline
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=42
        )
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\nTraining accuracy: {train_acc:.2f}")
    print(f"Validation accuracy: {test_acc:.2f}")
    
    # Save model and encoder
    joblib.dump({
        'model': model,
        'encoder': le,
        'timestamp': datetime.now().isoformat(),
        'accuracy': test_acc
    }, model_file)
    
    print(f"Model saved to {model_file}")
    return model, le

def monitor(interface, model=None, encoder=None):
    """Enhanced monitoring with prediction smoothing"""
    if not os.path.exists(f"/sys/class/net/{interface}"):
        print(f"ERROR: Interface {interface} not found!", file=sys.stderr)
        os.system("ip -o link | awk '!/loopback/ {print $2}' | cut -d':' -f1")
        sys.exit(1)

    # Initialize state
    signal_window = deque(maxlen=WINDOW_SIZE)
    snr_window = deque(maxlen=WINDOW_SIZE)
    prediction_buffer = deque(maxlen=5)
    smoothed_signal = None
    smoothed_snr = None
    
    print("=" * 70)
    print("AI-POWERED PROXIMITY DETECTION SYSTEM")
    print("=" * 70)
    print(f"{'Time':<8} | {'Signal':>7} | {'Noise':>7} | {'SNR':>6} | {'Prediction':<25} | {'Confidence':>9}")
    print("-" * 85)

    try:
        while True:
            timestamp = time.strftime("%H:%M:%S")
            metrics = get_wireless_metrics(interface)
            raw_snr = calculate_snr(metrics)
            
            # Handle missing data
            if metrics['signal'] is None:
                signal_str = noise_str = snr_str = "N/A"
                print(f"{timestamp:<8} | {signal_str:>7} | {noise_str:>7} | {snr_str:>6} | {'NO SIGNAL':<25} | ", end="\r")
                time.sleep(SAMPLING_RATE)
                continue
                
            # Update windows
            signal_window.append(metrics['signal'])
            snr_window.append(raw_snr if raw_snr is not None else 0)
            
            # Calculate smoothed values
            smoothed_signal = sum(signal_window) / len(signal_window)
            smoothed_snr = sum(snr_window) / len(snr_window)
            
            # Prepare display values
            signal_str = f"{smoothed_signal:.1f} dBm"
            noise_str = f"{metrics['noise']} dBm" if metrics['noise'] is not None else "N/A"
            snr_str = f"{smoothed_snr:.1f} dB"
            
            # Make prediction if model exists
            if model and encoder:
                features = [[smoothed_signal, smoothed_snr]]
                
                # Get prediction and probabilities
                pred = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                confidence = proba[pred]
                
                # Decode prediction
                label = encoder.inverse_transform([pred])[0]
                human_label = PROXIMITY_LABELS.get(label, label)
                
                # Update prediction buffer
                prediction_buffer.append(pred)
                
                # Get most common recent prediction
                if prediction_buffer:
                    final_pred = max(set(prediction_buffer), key=prediction_buffer.count)
                    final_label = encoder.inverse_transform([final_pred])[0]
                    human_label = PROXIMITY_LABELS.get(final_label, final_label)
                
                status = f"{human_label} ({confidence:.1%})"
                print(f"{timestamp:<8} | {signal_str:>7} | {noise_str:>7} | {snr_str:>6} | {status:<25} | {confidence:>8.1%}", end="\r")
            else:
                # Fallback static logic
                if smoothed_snr < 25:
                    status = PROXIMITY_LABELS["VERY_CLOSE"]
                elif smoothed_snr < 29:
                    status = PROXIMITY_LABELS["NEAR"]
                elif smoothed_snr < 31:
                    status = PROXIMITY_LABELS["MID_RANGE"]
                else:
                    status = PROXIMITY_LABELS["FAR_AWAY"]
                print(f"{timestamp:<8} | {signal_str:>7} | {noise_str:>7} | {snr_str:>6} | {status:<25} | {'N/A':>9}", end="\r")
            
            sys.stdout.flush()
            time.sleep(SAMPLING_RATE)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Wi-Fi Proximity Sensor")
    parser.add_argument("--collect", action="store_true", help="Data collection mode")
    parser.add_argument("--train", action="store_true", help="Train ML model")
    parser.add_argument("--model", type=str, default=MODEL_FILE, help="Model file to load/save")
    parser.add_argument("--interface", type=str, default=INTERFACE, help="Wi-Fi interface")
    args = parser.parse_args()

    if args.collect:
        collect_data(args.interface)
    elif args.train:
        train_model(model_file=args.model)
    else:
        # Try to load model
        model_data = None
        if os.path.exists(args.model):
            try:
                model_data = joblib.load(args.model)
                print(f"Loaded model trained at {model_data['timestamp']}")
                print(f"Validation accuracy: {model_data['accuracy']:.2f}")
            except Exception as e:
                print(f"Error loading model: {e}")
                model_data = None
                
        monitor(
            args.interface,
            model=model_data['model'] if model_data else None,
            encoder=model_data['encoder'] if model_data else None
        )

if __name__ == "__main__":
    main()
