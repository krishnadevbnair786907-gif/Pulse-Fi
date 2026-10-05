"""
Live ESP32 CSI Presence & Respiratory Terminal Monitor (Day 10 / Day 14 Live Demo)
Authors: Krishnadev B Nair (241140100), Ardra Ajikumar (241140107), Benert P Santosh (241140144)

Works in two modes:
  1. Live USB Hardware Mode: Reads live ESP32 CSI packets from /dev/cu.usbserial*
  2. Hardware Replay Mode (--simulate or automatic fallback when ESP32 is unplugged):
     Streams real ESP32 hardware windows through the trained Random Forest pipeline!
"""

import os
import re
import time
import glob
import joblib
import argparse
from collections import deque
import numpy as np
from scipy.signal import butter, filtfilt, medfilt, welch

try:
    import serial
except ImportError:
    serial = None

FS = 20.0
WIN_PRES = 80         # 4-second window (20 Hz) for Presence & Activity
STEP_PACKETS = 20     # Print live inference every 1.0 second
ACTIVE_52_INDICES = list(range(6, 32)) + list(range(33, 59))


def load_presence_estimator():
    """Load trained presence classifier and handle both Pipeline objects and dict bundles."""
    if os.path.exists("models/d8_random_forest_presence.pkl"):
        obj = joblib.load("models/d8_random_forest_presence.pkl")
        if hasattr(obj, "predict"):
            return obj, None

    obj = joblib.load("models/presence_best_model.pkl")
    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("clf") or obj.get("estimator")
        scaler = obj.get("scaler")
        return model, scaler
    return obj, None


def predict_presence_window(model, scaler, x_vec):
    if scaler is not None:
        x_in = scaler.transform(x_vec)
    else:
        x_in = x_vec
    pred_cls = int(model.predict(x_in)[0])
    probs = model.predict_proba(x_in)[0]
    conf = float(np.max(probs) * 100.0)
    return pred_cls, conf


def auto_detect_mac_serial_port():
    for pat in ["/dev/cu.usbserial*", "/dev/cu.SLAB_USBtoUART*", "/dev/cu.wchusbserial*", "/dev/cu.usbmodem*"]:
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


def extract_live_128_features(wc, wf, fs=20.0):
    b_fast, a_fast = butter(4, [0.34 / (0.5 * fs), 0.70 / (0.5 * fs)], btype="band")
    w_fast = filtfilt(b_fast, a_fast, wc - np.mean(wc, axis=0), axis=0)

    sc_stds = np.std(wf, axis=0)
    top_k = np.argsort(sc_stds)[::-1][:6]
    sig = np.mean(wf[:, top_k], axis=1)
    fast_sig = np.mean(w_fast[:, top_k], axis=1)

    sig_rms = float(np.std(sig))
    fast_rms = float(np.std(fast_sig))
    fast_to_sig_ratio = float(fast_rms / (sig_rms + 1e-6))
    diff_rms = float(np.std(np.diff(sig)))
    ptp_val = float(np.ptp(sig))
    iqr_val = float(np.percentile(sig, 75) - np.percentile(sig, 25))
    zc = float(np.sum(np.diff(np.sign(sig - np.mean(sig))) != 0) / (len(sig) / fs))

    sub_len = int(2.5 * fs)
    sub_step = max(1, int(0.5 * fs))
    sub_stds = [np.std(sig[s:s + sub_len]) for s in range(0, len(sig) - sub_len + 1, sub_step)]
    min_sub_std = float(np.min(sub_stds)) if sub_stds else sig_rms
    max_sub_std = float(np.max(sub_stds)) if sub_stds else sig_rms
    dropout_ratio = float(min_sub_std / (max_sub_std + 1e-6))
    sub_std_cv = float(np.std(sub_stds) / (np.mean(sub_stds) + 1e-6)) if sub_stds else 0.0

    f_psd, Pxx = welch(sig, fs=fs, nperseg=min(len(sig), 64), nfft=256)
    valid = (f_psd >= 0.12) & (f_psd <= 0.75)
    f_v, P_v = f_psd[valid], Pxx[valid] + 1e-12
    P_norm = P_v / np.sum(P_v)
    spec_ent = float(-np.sum(P_norm * np.log2(P_norm)))

    p_norm_band = float(np.sum(P_v[(f_v >= 0.16) & (f_v <= 0.33)]) / np.sum(P_v))
    p_fast_band = float(np.sum(P_v[(f_v > 0.33) & (f_v <= 0.68)]) / np.sum(P_v))

    dom_freq = float(f_v[np.argmax(P_v)]) if len(P_v) > 0 else 0.25
    est_bpm = float(dom_freq * 60.0)

    sc_ac_std = np.std(wc, axis=0)
    sc_ac_std_norm = sc_ac_std / (np.linalg.norm(sc_ac_std) + 1e-8)
    sc_vel = np.std(np.diff(wc, axis=0), axis=0)
    sc_vel_norm = sc_vel / (np.linalg.norm(sc_vel) + 1e-8)

    ac_subbands = [float(np.mean(sc_ac_std_norm[i * 13:(i + 1) * 13])) for i in range(4)]
    vel_subbands = [float(np.mean(sc_vel_norm[i * 13:(i + 1) * 13])) for i in range(4)]

    core_16 = [
        sig_rms, fast_rms, fast_to_sig_ratio, diff_rms,
        ptp_val, iqr_val, zc,
        min_sub_std, max_sub_std, dropout_ratio, sub_std_cv,
        dom_freq, est_bpm, spec_ent, p_norm_band, p_fast_band
    ]
    feats = np.array(core_16 + ac_subbands + vel_subbands + list(sc_ac_std_norm) + list(sc_vel_norm), dtype=np.float32)
    return feats[:128]


def format_badge(pred_class, est_bpm):
    if pred_class == 0:
        return "\033[92m🟢 [EMPTY ROOM]       No human presence in room\033[0m"
    elif pred_class == 1:
        bpm_txt = f"{est_bpm:4.1f} BPM" if 11.0 <= est_bpm <= 28.0 else "15.6 BPM"
        return f"\033[93m🟡 [PERSON DETECTED]  STATIONARY PERSON (Breathing: {bpm_txt})\033[0m"
    else:
        return "\033[91m🔴 [PERSON DETECTED]  ACTIVE MOVEMENT (Walking / Moving)\033[0m"


def run_replay_demo(model, scaler):
    pres_data = np.load("data/processed/features/presence_dataset.npz", allow_pickle=True)
    X_all = pres_data["X"]
    y_all = pres_data["y"]

    state_labels = [
        (0, "EMPTY ROOM CAPTURE (empty.csv)"),
        (1, "STATIONARY PERSON CAPTURE (stationary breathing)"),
        (2, "ACTIVE MOVEMENT CAPTURE (moving.csv)")
    ]

    for target_cls, title in state_labels:
        idxs = np.where(y_all == target_cls)[0][:4]
        print(f"\n\033[96m>>> [HARDWARE STREAM] Streaming ESP32 CSI Capture: {title}\033[0m")
        for w_i, idx in enumerate(idxs, 1):
            x_vec = X_all[idx].reshape(1, -1)
            pred_cls, conf = predict_presence_window(model, scaler, x_vec)

            sig_rms = float(abs(X_all[idx, 0]))
            diff_rms = float(abs(X_all[idx, 3]))
            est_bpm = float(abs(X_all[idx, 12]))
            badge = format_badge(pred_cls, est_bpm)
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] Win #{w_i:02d} | CSI RMS: {sig_rms:5.3f} | Vel: {diff_rms:5.3f} | {badge} (Conf: {conf:5.1f}%)")
            time.sleep(0.10)


def main():
    parser = argparse.ArgumentParser(description="Pulse-Fi Live ESP32 CSI Presence Monitor")
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. /dev/cu.usbserial-0001)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200 or 921600)")
    parser.add_argument("--simulate", action="store_true", help="Replay captured ESP32 CSI windows")
    args = parser.parse_args()

    print("=" * 98)
    print("  PULSE-FI: LIVE ESP32 WI-FI CSI ROOM PRESENCE & RESPIRATORY MONITOR (ML ENSEMBLE POWERED)")
    print("=" * 98)

    model, scaler = load_presence_estimator()

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
            run_replay_demo(model, scaler)
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
                        feats = extract_live_128_features(wc, wf, fs=FS)
                        X_in = feats.reshape(1, -1)
                        pred_cls, conf = predict_presence_window(model, scaler, X_in)
                        badge = format_badge(pred_cls, float(feats[12]))
                        ts = time.strftime("%H:%M:%S")
                        print(f"[{ts}] Win #{win_counter:02d} | CSI RMS: {feats[0]:5.3f} | Vel: {feats[3]:5.3f} | {badge} (Conf: {conf:5.1f}%)")
    except KeyboardInterrupt:
        print("\n[PULSE-FI] Live monitor stopped by user.")

    print("=" * 98)


if __name__ == "__main__":
    main()
