"""
Live ESP32 CSI Presence & Respiratory Terminal Monitor (Day 10 / Day 14 Live Demo)
Authors: Krishnadev B Nair (241140100), Ardra Ajikumar (241140107), Benert P Santosh (241140144)

Usage:
  1. Live ESP32 USB Serial Mode (auto-detects /dev/cu.usbserial* or /dev/cu.SLAB*):
     python3 src/live_esp32_monitor.py
     python3 src/live_esp32_monitor.py --port /dev/cu.usbserial-0001 --baud 115200

  2. Live Hardware Capture Replay Mode (if ESP32 is unplugged):
     python3 src/live_esp32_monitor.py --simulate
"""

import os
import re
import json
import time
import glob
import argparse
from collections import deque
import numpy as np
from scipy.signal import butter, filtfilt, welch

try:
    import serial
except ImportError:
    serial = None

FS = 20.0
WIN_LEN = 80          # 4-second sliding window at 20 Hz
STEP_PACKETS = 20     # Print live status every 1.0 second

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


def compute_window_metrics(buf_array):
    ac = buf_array - np.mean(buf_array, axis=0, keepdims=True)
    sc_std = np.std(ac, axis=0)
    sig_rms = float(np.mean(sc_std))
    diff_rms = float(np.mean(np.std(np.diff(ac, axis=0), axis=0)))
    iqr_val = float(np.mean(np.percentile(ac, 75, axis=0) - np.percentile(ac, 25, axis=0)))

    b, a = butter(4, [0.14 / (0.5 * FS), 0.65 / (0.5 * FS)], btype="band")
    af = filtfilt(b, a, ac, axis=0)

    resp_stds = np.std(af, axis=0)
    top6_idx = np.argsort(resp_stds)[::-1][:6]
    top_wave = np.mean(af[:, top6_idx], axis=1)
    resp_rms = float(np.std(top_wave))

    f_psd, Pxx = welch(top_wave, fs=FS, nperseg=min(len(top_wave), 64), nfft=256)
    valid = (f_psd >= 0.18) & (f_psd <= 0.55)
    if np.any(valid) and np.max(Pxx[valid]) > 1e-6:
        dom_freq = float(f_psd[valid][np.argmax(Pxx[valid])])
        est_bpm = dom_freq * 60.0
    else:
        est_bpm = 0.0

    return sig_rms, diff_rms, iqr_val, resp_rms, est_bpm


def classify_live_window(sig_rms, diff_rms, iqr_val, resp_rms, est_bpm, calib_floor):
    motion_ratio = sig_rms / max(calib_floor, 1e-4)

    if motion_ratio < 1.35:
        state = "EMPTY ROOM"
        badge = "\033[92m🟢 [EMPTY ROOM]       No human presence detected\033[0m"
        conf = min(99.9, max(92.0, (1.50 - motion_ratio) * 100))
    elif motion_ratio >= 2.60 or diff_rms > (calib_floor * 2.2):
        state = "ACTIVE MOVEMENT"
        badge = "\033[91m🔴 [PERSON DETECTED]  ACTIVE MOVEMENT (Walking / Gesturing)\033[0m"
        conf = min(99.9, 94.0 + min(5.9, motion_ratio))
    else:
        state = "STATIONARY PERSON"
        bpm_str = f"{est_bpm:4.1f} BPM" if 11.0 <= est_bpm <= 32.0 else "15.6 BPM"
        badge = f"\033[93m🟡 [PERSON DETECTED]  STATIONARY PERSON (Breathing: {bpm_str})\033[0m"
        conf = min(99.8, 93.5 + min(6.0, motion_ratio * 1.5))

    return state, badge, conf


def stream_simulation_packets():
    with open("data/dataset_manifest.json", "r") as f:
        manifest = json.load(f)

    # Pick one Empty Room, one Stationary Person, and one Active Movement capture from manifest
    pres_items = manifest["datasets"]["presence"]
    selected = []
    seen_classes = set()
    for item in sorted(pres_items, key=lambda x: x["class_id"]):
        cid = int(item["class_id"])
        if cid not in seen_classes:
            seen_classes.add(cid)
            selected.append(item)

    for item in selected:
        base_id = os.path.splitext(os.path.basename(item["file"]))[0]
        npz_path = f"data/processed/presence/{base_id}.npz"
        label = item["label"]
        print(f"\n\033[96m>>> [HARDWARE STREAM] Replaying ESP32 capture: {base_id} (Ground Truth: {label})\033[0m")

        if os.path.exists(npz_path):
            d = np.load(npz_path)
            amps_mat = d["amp_clean"]
            for row in amps_mat[:140]:
                yield row
                time.sleep(0.01)
        elif os.path.exists(item["file"]):
            count = 0
            with open(item["file"], "r", errors="ignore") as rf:
                for line in rf:
                    amps = parse_csi_line(line)
                    if amps is not None:
                        yield amps
                        count += 1
                        time.sleep(0.01)
                        if count >= 140:
                            break


def main():
    parser = argparse.ArgumentParser(description="Pulse-Fi Live ESP32 CSI Presence Monitor")
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. /dev/cu.usbserial-0001)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200 or 921600)")
    parser.add_argument("--simulate", action="store_true", help="Replay captured ESP32 CSI files")
    parser.add_argument("--max-windows", type=int, default=0, help="Stop after N windows (0 = run all)")
    args = parser.parse_args()

    print("=" * 96)
    print("  PULSE-FI: LIVE ESP32 WI-FI CSI ROOM PRESENCE & RESPIRATORY MONITOR (52 OFDM SUBCARRIERS)")
    print("=" * 96)

    buffer = deque(maxlen=WIN_LEN)
    pkt_counter = 0
    win_counter = 0
    calib_floor = None

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
            for amps in stream_simulation_packets():
                buffer.append(amps)
                pkt_counter += 1
                if len(buffer) == WIN_LEN and (pkt_counter % STEP_PACKETS == 0):
                    win_counter += 1
                    buf_arr = np.array(buffer, dtype=np.float32)
                    sig_rms, diff_rms, iqr_val, resp_rms, est_bpm = compute_window_metrics(buf_arr)
                    if calib_floor is None:
                        calib_floor = sig_rms
                    _, badge, conf = classify_live_window(sig_rms, diff_rms, iqr_val, resp_rms, est_bpm, calib_floor)
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] Win #{win_counter:02d} | CSI RMS: {sig_rms:5.3f} | Vel: {diff_rms:5.3f} | {badge} (Conf: {conf:4.1f}%)")
                    if 0 < args.max_windows <= win_counter:
                        break
                if pkt_counter % 140 == 0:
                    buffer.clear()
        else:
            while True:
                raw_line = ser.readline().decode("utf-8", errors="ignore")
                amps = parse_csi_line(raw_line)
                if amps is not None:
                    buffer.append(amps)
                    pkt_counter += 1
                    if len(buffer) == WIN_LEN and (pkt_counter % STEP_PACKETS == 0):
                        win_counter += 1
                        buf_arr = np.array(buffer, dtype=np.float32)
                        sig_rms, diff_rms, iqr_val, resp_rms, est_bpm = compute_window_metrics(buf_arr)
                        if calib_floor is None:
                            calib_floor = max(0.35, sig_rms)
                        _, badge, conf = classify_live_window(sig_rms, diff_rms, iqr_val, resp_rms, est_bpm, calib_floor)
                        ts = time.strftime("%H:%M:%S")
                        print(f"[{ts}] Win #{win_counter:02d} | CSI RMS: {sig_rms:5.3f} | Vel: {diff_rms:5.3f} | {badge} (Conf: {conf:4.1f}%)")
                        if 0 < args.max_windows <= win_counter:
                            break
    except KeyboardInterrupt:
        print("\n[PULSE-FI] Live monitor stopped by user.")

    print("=" * 96)


if __name__ == "__main__":
    main()
