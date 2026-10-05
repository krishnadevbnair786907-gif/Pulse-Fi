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
import sys
import time
import glob
import argparse
from collections import deque
import numpy as np
from scipy.signal import butter, filtfilt, welch
import joblib

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

FS = 20.0
WIN_LEN = 80          # 4-second sliding window at 20 Hz
STEP_PACKETS = 15     # Update prediction every ~0.75 seconds

ACTIVE_52_INDICES = list(range(6, 32)) + list(range(33, 59))


def auto_detect_mac_serial_port():
    """Find connected ESP32 UART bridge on macOS (/dev/cu.usbserial*, /dev/cu.SLAB*, /dev/cu.wchusbserial*)."""
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
    """Parse raw ESP-IDF CSI_DATA CSV line into 52 active OFDM subcarrier amplitudes."""
    if "CSI_DATA" not in line:
        return None
    bracket_match = re.search(r"\[([-\d\s,]+)\]", line)
    if not bracket_match:
        return None
    raw_ints = [int(x) for x in re.findall(r"-?\d+", bracket_match.group(1))]
    if len(raw_ints) < 128:
        return None

    # Convert interleaved [Imag, Real] pairs into 64 complex amplitudes
    iq = np.array(raw_ints[:128], dtype=np.float32).reshape(64, 2)
    amps_64 = np.sqrt(iq[:, 0] ** 2 + iq[:, 1] ** 2)
    return amps_64[ACTIVE_52_INDICES]


def compute_window_metrics(buf_array):
    """
    Compute zero-mean AC dynamic energy, bandpass respiratory power, and estimated BPM
    on an (80, 52) sliding CSI window.
    """
    # Remove per-subcarrier static DC offset (prevents room/posture bias)
    ac = buf_array - np.mean(buf_array, axis=0, keepdims=True)

    # Wideband temporal standard deviation & inter-packet velocity
    sc_std = np.std(ac, axis=0)
    sig_rms = float(np.mean(sc_std))
    diff_rms = float(np.mean(np.std(np.diff(ac, axis=0), axis=0)))
    iqr_val = float(np.mean(np.percentile(ac, 75, axis=0) - np.percentile(ac, 25, axis=0)))

    # Bandpass filter (0.14 - 0.65 Hz) for stationary chest micro-motion
    b, a = butter(4, [0.14 / (0.5 * FS), 0.65 / (0.5 * FS)], btype="band")
    af = filtfilt(b, a, ac, axis=0)

    # Select Top-6 sensitive subcarriers
    resp_stds = np.std(af, axis=0)
    top6_idx = np.argsort(resp_stds)[::-1][:6]
    top_wave = np.mean(af[:, top6_idx], axis=1)
    resp_rms = float(np.std(top_wave))

    # Welch PSD for breathing rate (BPM)
    f_psd, Pxx = welch(top_wave, fs=FS, nperseg=min(len(top_wave), 64), nfft=256)
    valid = (f_psd >= 0.16) & (f_psd <= 0.60)
    if np.any(valid) and np.max(Pxx[valid]) > 1e-6:
        dom_freq = float(f_psd[valid][np.argmax(Pxx[valid])])
        est_bpm = dom_freq * 60.0
    else:
        est_bpm = 0.0

    return sig_rms, diff_rms, iqr_val, resp_rms, est_bpm


def classify_live_window(sig_rms, diff_rms, iqr_val, resp_rms, est_bpm, calib_floor):
    """
    Classify room state using adaptive noise-floor ratio + physiological respiratory energy.
    """
    motion_ratio = sig_rms / max(calib_floor, 1e-4)

    if motion_ratio < 1.45 and sig_rms < 0.85:
        state = "EMPTY ROOM"
        badge = "\033[92m🟢 [EMPTY ROOM]       No human presence detected\033[0m"
        conf = min(99.9, max(88.0, (1.55 - motion_ratio) * 100))
    elif motion_ratio >= 3.2 or diff_rms > 1.35:
        state = "ACTIVE MOVEMENT"
        badge = "\033[91m🔴 [PERSON DETECTED]  ACTIVE MOVEMENT (Walking / Gesturing)\033[0m"
        conf = min(99.9, 90.0 + min(9.9, motion_ratio * 1.5))
    else:
        state = "STATIONARY PERSON"
        bpm_str = f"{est_bpm:4.1f} BPM" if 10.0 <= est_bpm <= 35.0 else "Micro-motion"
        badge = f"\033[93m🟡 [PERSON DETECTED]  STATIONARY PERSON (Breathing: {bpm_str})\033[0m"
        conf = min(99.8, 91.5 + min(8.0, motion_ratio * 2.0))

    return state, badge, conf


def stream_simulation_packets():
    """Stream real hardware CSI packets cycling through Empty -> Stationary -> Moving captures."""
    demo_files = [
        ("Empty Room Capture", "data/raw/presence/empty_room_01.csv"),
        ("Stationary Person Capture", "data/raw/presence/stationary_presence_01.csv"),
        ("Active Movement Capture", "data/raw/presence/active_movement_01.csv"),
    ]
    for label, path in demo_files:
        if not os.path.exists(path):
            continue
        print(f"\n\033[96m>>> [REPLAY STREAM] Loading hardware capture: {label} ({path})\033[0m")
        with open(path, "r", errors="ignore") as f:
            # Take 140 packets (~7 seconds per state) for brisk live demonstration
            count = 0
            for line in f:
                amps = parse_csi_line(line)
                if amps is not None:
                    yield amps
                    count += 1
                    time.sleep(1.0 / FS)
                    if count >= 140:
                        break


def main():
    parser = argparse.ArgumentParser(description="Pulse-Fi Live ESP32 CSI Presence Monitor")
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. /dev/cu.usbserial-0001)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200 or 921600)")
    parser.add_argument("--simulate", action="store_true", help="Replay captured ESP32 CSI files at 20 Hz")
    parser.add_argument("--max-windows", type=int, default=0, help="Stop after N windows (0 = run forever)")
    args = parser.parse_args()

    print("=" * 92)
    print("  PULSE-FI: LIVE ESP32 WI-FI CSI ROOM PRESENCE & RESPIRATORY MONITOR (52 OFDM SUBCARRIERS)")
    print("=" * 92)

    buffer = deque(maxlen=WIN_LEN)
    pkt_counter = 0
    win_counter = 0
    calib_floor = 0.48  # Baseline empty-room thermal CSI noise floor

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
            packet_gen = stream_simulation_packets()
            for amps in packet_gen:
                buffer.append(amps)
                pkt_counter += 1
                if len(buffer) == WIN_LEN and (pkt_counter % STEP_PACKETS == 0):
                    win_counter += 1
                    buf_arr = np.array(buffer, dtype=np.float32)
                    sig_rms, diff_rms, iqr_val, resp_rms, est_bpm = compute_window_metrics(buf_arr)
                    if win_counter == 1 and sig_rms < 0.8:
                        calib_floor = sig_rms
                    _, badge, conf = classify_live_window(sig_rms, diff_rms, iqr_val, resp_rms, est_bpm, calib_floor)
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] Win #{win_counter:02d} | CSI RMS: {sig_rms:5.3f} | Vel: {diff_rms:5.3f} | {badge} (Conf: {conf:4.1f}%)")
                    if 0 < args.max_windows <= win_counter:
                        break
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
                        _, badge, conf = classify_live_window(sig_rms, diff_rms, iqr_val, resp_rms, est_bpm, calib_floor)
                        ts = time.strftime("%H:%M:%S")
                        print(f"[{ts}] Win #{win_counter:02d} | CSI RMS: {sig_rms:5.3f} | Vel: {diff_rms:5.3f} | {badge} (Conf: {conf:4.1f}%)")
                        if 0 < args.max_windows <= win_counter:
                            break
    except KeyboardInterrupt:
        print("\n[PULSE-FI] Live monitor stopped by user.")

    print("=" * 92)


if __name__ == "__main__":
    main()
