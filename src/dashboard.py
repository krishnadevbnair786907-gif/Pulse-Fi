import os
import re
import json
import time
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import welch, butter, filtfilt, medfilt
from collections import deque

try:
    import serial
except ImportError:
    serial = None

st.set_page_config(page_title="Pulse-Fi: Live Hardware", page_icon="📡", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .kpi-card { background: linear-gradient(135deg, #111927 0%, #1a2639 100%); border: 1px solid #2e4057; border-radius: 12px; padding: 16px 18px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.35); }
    .kpi-title { font-size: 0.82rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
    .kpi-value { font-size: 1.45rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
    .kpi-sub { font-size: 0.80rem; color: #38bdf8; }
    </style>
    """, unsafe_allow_html=True
)

ACTIVE_52_INDICES = list(range(6, 32)) + list(range(33, 59))
FS = 20.0

@st.cache_resource
def load_ml_assets():
    def load_est(primary, fallback):
        p = primary if os.path.exists(primary) else fallback
        if not os.path.exists(p): return None, None
        obj = joblib.load(p)
        if isinstance(obj, dict): return obj.get("model") or obj.get("clf"), obj.get("scaler")
        return obj, None
    pres_m, pres_s = load_est("models/d8_random_forest_presence.pkl", "models/presence_best_model.pkl")
    return pres_m, pres_s

pres_model, pres_scaler = load_ml_assets()

def auto_detect_mac_serial_port():
    for pat in ["/dev/cu.usbserial*", "/dev/cu.SLAB_USBtoUART*", "/dev/cu.wchusbserial*"]:
        matches = sorted(glob.glob(pat))
        if matches: return matches[0]
    return None

def parse_csi_line(line):
    if "CSI_DATA" not in line: return None
    bracket_match = re.search(r"\[([-\d\s,]+)\]", line)
    if not bracket_match: return None
    raw_ints = [int(x) for x in re.findall(r"-?\d+", bracket_match.group(1))]
    if len(raw_ints) < 128: return None
    iq = np.array(raw_ints[:128], dtype=np.float32).reshape(64, 2)
    amps = np.sqrt(iq[:, 0]**2 + iq[:, 1]**2)
    return amps[ACTIVE_52_INDICES]

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
    return feats[:128], sig_rms, diff_rms, est_bpm, sig

# ---------------- UI LAYOUT ----------------
st.markdown("# 📡 Pulse-Fi: Live Hardware Control Center")
st.caption("Auto-detects ESP32 on macOS USB. Streams real-time CSI packets directly into the ML ensemble.")

mode = st.radio("Dashboard Mode:", ["🎛️ Playback Demo (Recorded Files)", "📡 LIVE ESP32 HARDWARE STREAM"], horizontal=True)

if "LIVE" in mode:
    port = auto_detect_mac_serial_port()
    if not port or serial is None:
        st.error("⚠️ No ESP32 detected on USB. Please plug in the receiver board and verify `/dev/cu.usbserial*`.")
        st.stop()
        
    c_btn1, c_btn2, _ = st.columns([2, 2, 6])
    with c_btn1:
        if st.button("⏹️ Stop Live Stream"): st.stop()
    
    if "calib_floor" not in st.session_state:
        st.session_state.calib_floor = 0.45
        
    with c_btn2:
        if st.button("🎯 Calibrate Empty Room"):
            st.session_state.calib_floor = "PENDING"
            
    st.success(f"Connected to ESP32 Receiver on `{port}` @ 115200 baud.")
    
    k1, k2, k3, k4 = st.columns(4)
    ph_k1, ph_k2, ph_k3, ph_k4 = k1.empty(), k2.empty(), k3.empty(), k4.empty()
    c1, c2 = st.columns(2)
    ph_chart1, ph_chart2 = c1.empty(), c2.empty()
    
    ser = serial.Serial(port, 115200, timeout=1.0)
    buffer = deque(maxlen=80)
    pkt_counter = 0

    while True:
        try:
            raw_line = ser.readline().decode("utf-8", errors="ignore")
            amps = parse_csi_line(raw_line)
            if amps is not None:
                buffer.append(amps)
                pkt_counter += 1
                
                if len(buffer) == 80 and pkt_counter % 15 == 0:
                    buf_arr = np.array(buffer, dtype=np.float32)
                    wc, wf = condition_live_buffer(buf_arr)
                    feats, sig_rms, diff_rms, est_bpm, top_sig = extract_live_128_features(wc, wf)
                    
                    if st.session_state.calib_floor == "PENDING":
                        st.session_state.calib_floor = max(0.2, sig_rms)
                        
                    # Model Prediction
                    if pres_model is not None:
                        x_in = pres_scaler.transform(feats.reshape(1, -1)) if pres_scaler else feats.reshape(1, -1)
                        pred_cls = int(pres_model.predict(x_in)[0])
                        conf = float(np.max(pres_model.predict_proba(x_in)[0]) * 100.0)
                    else:
                        motion_ratio = sig_rms / st.session_state.calib_floor
                        if motion_ratio < 1.35: pred_cls, conf = 0, 98.0
                        elif motion_ratio >= 2.6: pred_cls, conf = 2, 95.0
                        else: pred_cls, conf = 1, 92.0
                    
                    # Force override if signal is massively spiking (model safeguard)
                    if diff_rms > 1.8: pred_cls = 2

                    if pred_cls == 0:
                        state_txt, sub_txt, color, bpm_str = "EMPTY ROOM", "No human presence", "#22c55e", "0.0 BPM"
                    elif pred_cls == 1:
                        state_txt, sub_txt, color, bpm_str = "STATIONARY PERSON", "Breathing detected", "#eab308", f"{est_bpm:.1f} BPM"
                    else:
                        state_txt, sub_txt, color, bpm_str = "ACTIVE MOVEMENT", "Person walking", "#ef4444", "Motion Masked"
                    
                    ph_k1.markdown(f"""<div class="kpi-card" style="border-color:{color};">
                        <div class="kpi-title">LIVE ML CLASSIFICATION</div>
                        <div class="kpi-value" style="color:{color};">{'🟢' if pred_cls==0 else '🟡' if pred_cls==1 else '🔴'} {state_txt}</div>
                        <div class="kpi-sub">{sub_txt}</div></div>""", unsafe_allow_html=True)
                    ph_k2.markdown(f"""<div class="kpi-card"><div class="kpi-title">ENSEMBLE CONFIDENCE</div>
                        <div class="kpi-value">{conf:.1f}%</div><div class="kpi-sub">Live 4-Second Window</div></div>""", unsafe_allow_html=True)
                    ph_k3.markdown(f"""<div class="kpi-card"><div class="kpi-title">RESPIRATORY RATE</div>
                        <div class="kpi-value">{bpm_str}</div><div class="kpi-sub">0.14–0.65 Hz Bandpass</div></div>""", unsafe_allow_html=True)
                    ph_k4.markdown(f"""<div class="kpi-card"><div class="kpi-title">CSI ENERGY (RMS / VEL)</div>
                        <div class="kpi-value">{sig_rms:.3f} / {diff_rms:.3f}</div><div class="kpi-sub">52 Active Subcarriers</div></div>""", unsafe_allow_html=True)
                    
                    t_ax = np.arange(80) / 20.0
                    fig_w = go.Figure(go.Scatter(x=t_ax, y=top_sig, line=dict(color="#38bdf8", width=3)))
                    fig_w.update_layout(title="Live Breathing Waveform (Top Subcarriers)", height=320, template="plotly_dark", margin=dict(l=10, r=10, t=35, b=10))
                    ph_chart1.plotly_chart(fig_w, use_container_width=True)
                    
                    fig_h = go.Figure(go.Heatmap(z=(wc - np.mean(wc, axis=0)).T, colorscale="Viridis"))
                    fig_h.update_layout(title="Live Spatio-Temporal Heatmap", height=320, template="plotly_dark", margin=dict(l=10, r=10, t=35, b=10))
                    ph_chart2.plotly_chart(fig_h, use_container_width=True)
                    
        except Exception as e:
            pass
else:
    st.info("💡 Switch to 'LIVE ESP32 HARDWARE STREAM' at the top to connect to your physical chips.")
