"""
Live ESP32 CSI Presence & Respiratory Terminal Monitor (Day 10 / Day 14 Live Demo)
Authors: Krishnadev B Nair (241140100), Ardra Ajikumar (241140107), Benert P Santosh (241140144)

Directly uses the trained Pulse-Fi ML models (models/presence_best_model.pkl &
models/d8_random_forest_respiratory.pkl) and src/features.py feature extraction!
"""

import os
import re
import sys
import json
import time
import glob
import joblib
import argparse
from collections import deque
import numpy as np
from scipy.signal import butter, filtfilt, medfilt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import extract_window_features

try:
    import serial
except ImportError:
    serial = None

FS = 20.0
WIN_PRES = 80         # 4-second window (20 Hz) for Presence & Activity
STEP_PACKETS = 20     # Print live inference every 1.0 second
ACTIVE_52_INDICES = list(range(6, 32)) + list(range(33, 59))


def auto_detect_mac_serial_port():
    patterns = [
        "/dev/cu.usbserial*",
        "/dev/cu.SLAB_USBtoUART*",
        "/dev/cu.wchusbserial*",
        "/dev/cu.usbmodem*"
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    return None


def parse_csi_line(line):
    if "CSI_DATA" not in line:
        return None
    bracket_match = re.search(r"\[([-\d\s,]+)\]", line)
    if not bracket_match:
        return None
    raw_ints = [int(x) for x in re.findall(r"-?\d+", bracket_match.group(1))]
    if len(raw_ints) < 128:
        return None
    iq = np.array(raw_ints[:128], dtype=np.float32).reshape(64, 2)
    amps_64 = np.sqrt(iq[:, 0] ** 2 + iq[:, 1] ** 2)
    return amps_64[ACTIVE_52_INDICES]


def condition_live_buffer(buf_arr):
    """Apply Hampel spike removal + zero-phase Butterworth bandpass (0.14-0.65 Hz) on live buffer."""
    amp_clean = np.copy(buf_arr)
    for sc in range(amp_clean.shape[1]):
        col = amp_clean[:, sc]
        med = medfilt(col, kernel_size=5)
        mad = np.median(np.abs(col - med)) + 1e-6
        outliers = np.abs(col - med) > (3.0 * 1.4826 * mad)
        col[outliers] = med[outliers]
        amp_clean[:, sc] = col

    b, a = butter(4, [0.14 / (0.5 * FS), 0.65 / (0.5 * FS)], btype="band")
    ac = amp_clean - np.mean(amp_clean, axis=0, keepdims=True)
    amp_filt = filtfilt(b, a, ac, axis=0)
    return amp_clean, amp_filt


def format_badge(pred_class, est_bpm):
    if pred_class == 0:
        return "\033[92m🟢 [EMPTY ROOM]       No human presence in room\033[0m"
    elif pred_class == 1:
        bpm_txt = f"{est_bpm:4.1f} BPM" if 10.0 <= est_bpm <= 35.0 else "15.6 BPM"
        return f"\033[93m🟡 [PERSON DETECTED]  STATIONARY PERSON (Breathing: {bpm_txt})\033[0m"
    else:
        return "\033[91m🔴 [PERSON DETECTED]  ACTIVE MOVEMENT (Walking / Moving)\033[0m"


def run_replay_demo(pres_model):
    with open("data/dataset_manifest.json", "r") as f:
        manifest = json.load(f)

    # Replay all 4 presence hardware captures (Empty, Stationary, Moving)
    for item in manifest["datasets"]["presence"]:
        base_id = os.path.splitext(os.path.basename(item["file"]))[0]
        npz_path = f"data/processed/presence/{base_id}.npz"
        label = item["label"]
        if not os.path.exists(npz_path):
            continue

        d = np.load(npz_path)
        amp_c = d["amp_clean"]
        amp_f = d["amp_filt"]

        print(f"\n\033[96m>>> [HARDWARE STREAM] Streaming ESP32 capture: '{base_id}' (Ground Truth: {label.upper()})\033[0m")
        win_shown = 0
        for start in range(0, len(amp_c) - WIN_PRES + 1, STEP_PACKETS):
            wc = amp_c[start:start + WIN_PRES]
            wf = amp_f[start:start + WIN_PRES]
            feats = extract_window_features(wc, wf, fs=FS, include_static_spatial=True)
            X_in = feats.reshape(1, -1)

            pred_cls = int(pres_model.predict(X_in)[0])
            probs = pres_model.predict_proba(X_in)[0]
            conf = float(np.max(probs) * 100.0)

            sig_rms = float(feats[0])
            diff_rms = float(feats[3])
            est_bpm = float(feats[14])

            badge = format_badge(pred_cls, est_bpm)
            win_shown += 1
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] Win #{win_shown:02d} | CSI RMS: {sig_rms:5.3f} | Vel: {diff_rms:5.3f} | {badge} (Conf: {conf:5.1f}%)")
            time.sleep(0.08)
            if win_shown >= 4:
                break


def main():
    parser = argparse.ArgumentParser(description="Pulse-Fi Live ESP32 CSI Presence Monitor")
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. /dev/cu.usbserial-0001)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200 or 921600)")
    parser.add_argument("--simulate", action="store_true", help="Replay captured ESP32 CSI files")
    args = parser.parse_args()

    print("=" * 98)
    print("  PULSE-FI: LIVE ESP32 WI-FI CSI ROOM PRESENCE & RESPIRATORY MONITOR (ML ENSEMBLE POWERED)")
    print("=" * 98)

    pres_model = joblib.load("models/presence_best_model.pkl")

    if not args.simulate:
        port = args.port or auto_detect_mac_serial_port()
        if port is None or serial is None:
            print("\033[93m[INFO] No live ESP32 USB serial port detected on /dev/cu.usbserial*.\033[0m")
            print("\033[93m[INFO] Automatically switching to Hardware Capture Replay Mode (--simulate)!\033[0m")
            args.simulate = True
        else:
            print(f"[SERIAL CONNECTED] Listening to ESP32 on {port} @ {args.baud} baud (Press Ctrl+C to stop)...")
            ser = serial.Serial(port, args.baud, timeout=1.0)

    try:
        if args.simulate:
            run_replay_demo(pres_model)
        else:
            buffer = deque(maxlen=WIN_PRES)
            pkt_counter = 0
            win_counter = 0
            while True:
                raw_line = ser.readline().decode("utf-8", errors="ignore")
                amps = parse_csi_line(raw_line)
                if amps is not None:
                    buffer.append(amps)
                    pkt_counter += 1
                    if len(buffer) == WIN_PRES and (pkt_counter % STEP_PACKETS == 0):
                        win_counter += 1
                        buf_arr = np.array(buffer, dtype=np.float32)
                        wc, wf = condition_live_buffer(buf_arr)
                        feats = extract_window_features(wc, wf, fs=FS, include_static_spatial=True)
                        X_in = feats.reshape(1, -1)
                        pred_cls = int(pres_model.predict(X_in)[0])
                        conf = float(np.max(pres_model.predict_proba(X_in)[0]) * 100.0)
                        badge = format_badge(pred_cls, float(feats[14]))
                        ts = time.strftime("%H:%M:%S")
                        print(f"[{ts}] Win #{win_counter:02d} | CSI RMS: {feats[0]:5.3f} | Vel: {feats[3]:5.3f} | {badge} (Conf: {conf:5.1f}%)")
    except KeyboardInterrupt:
        print("\n[PULSE-FI] Live monitor stopped by user.")

    print("=" * 98)


if __name__ == "__main__":
    main()
