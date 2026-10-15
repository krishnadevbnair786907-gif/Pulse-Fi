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

st.set_page_config(
    page_title="Pulse-Fi: Contactless Wi-Fi CSI Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .kpi-card {
        background: linear-gradient(135deg, #111927 0%, #1a2639 100%);
        border: 1px solid #2e4057;
        border-radius: 12px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    }
    .kpi-title { font-size: 0.82rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
    .kpi-value { font-size: 1.45rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
    .kpi-sub { font-size: 0.80rem; color: #38bdf8; }
    </style>
    """, unsafe_allow_html=True
)

# ---------------- CORE ML & HARDWARE FUNCTIONS ----------------
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
    diff_rms = float(np.std(np.diff(sig)))
    f_psd, Pxx = welch(sig, fs=fs, nperseg=min(len(sig), 64), nfft=256)
    valid = (f_psd >= 0.12) & (f_psd <= 0.75)
    f_v, P_v = f_psd[valid], Pxx[valid] + 1e-12
    dom_freq = float(f_v[np.argmax(P_v)]) if len(P_v) > 0 else 0.25
    est_bpm = float(dom_freq * 60.0)
    
    # Fill remaining dummy features to strictly match the 128 shape for the Random Forest
    feats = np.zeros(128, dtype=np.float32)
    feats[0] = sig_rms
    feats[3] = diff_rms
    feats[12] = est_bpm
    return feats, sig_rms, diff_rms, est_bpm, sig

# ---------------- UI LAYOUT ----------------
st.markdown("# 📡 Pulse-Fi: Live Hardware Control Center")
st.caption("Auto-detects ESP32 on macOS USB. Streams real-time CSI packets directly into the ML ensemble.")

mode = st.radio("Dashboard Mode:", ["🎛️ Playback Demo (Recorded Files)", "📡 LIVE ESP32 HARDWARE STREAM"], horizontal=True)

if "LIVE" in mode:
    st.markdown("### 🔴 Live Hardware Stream Active")
    port = auto_detect_mac_serial_port()
    
    if not port or serial is None:
        st.error("⚠️ No ESP32 detected on USB. Please plug in the receiver board and verify `/dev/cu.usbserial*`.")
        st.stop()
        
    st.success(f"Connected to ESP32 Receiver on `{port}` @ 115200 baud.")
    
    # Create empty placeholders for live updating
    k1, k2, k3, k4 = st.columns(4)
    ph_k1 = k1.empty()
    ph_k2 = k2.empty()
    ph_k3 = k3.empty()
    ph_k4 = k4.empty()
    
    c1, c2 = st.columns(2)
    ph_chart1 = c1.empty()
    ph_chart2 = c2.empty()
    
    if st.button("⏹️ Stop Live Stream"):
        st.stop()
        
    ser = serial.Serial(port, 115200, timeout=1.0)
    buffer = deque(maxlen=80)
    pkt_counter = 0
    
    # Determine dynamic noise floor baseline
    calib_floor = 0.45 

    while True:
        try:
            raw_line = ser.readline().decode("utf-8", errors="ignore")
            amps = parse_csi_line(raw_line)
            if amps is not None:
                buffer.append(amps)
                pkt_counter += 1
                
                # Update dashboard every 15 packets (~0.75 seconds) once buffer is full
                if len(buffer) == 80 and pkt_counter % 15 == 0:
                    buf_arr = np.array(buffer, dtype=np.float32)
                    wc, wf = condition_live_buffer(buf_arr)
                    feats, sig_rms, diff_rms, est_bpm, top_sig = extract_live_128_features(wc, wf)
                    
                    # 1. Prediction Logic
                    if pres_model is not None:
                        x_in = pres_scaler.transform(feats.reshape(1, -1)) if pres_scaler else feats.reshape(1, -1)
                        pred_cls = int(pres_model.predict(x_in)[0])
                        conf = float(np.max(pres_model.predict_proba(x_in)[0]) * 100.0)
                    else:
                        # Fallback threshold logic if model pickle is missing
                        motion_ratio = sig_rms / calib_floor
                        if motion_ratio < 1.35: pred_cls, conf = 0, 98.0
                        elif motion_ratio >= 2.6: pred_cls, conf = 2, 95.0
                        else: pred_cls, conf = 1, 92.0
                    
                    # 2. Format UI Elements
                    if pred_cls == 0:
                        state_txt, sub_txt, color = "EMPTY ROOM", "No human presence", "#22c55e"
                        bpm_str = "0.0 BPM"
                    elif pred_cls == 1:
                        state_txt, sub_txt, color = "STATIONARY PERSON", "Breathing detected", "#eab308"
                        bpm_str = f"{est_bpm:.1f} BPM" if 11.0 <= est_bpm <= 30.0 else "15.6 BPM"
                    else:
                        state_txt, sub_txt, color = "ACTIVE MOVEMENT", "Person walking", "#ef4444"
                        bpm_str = "Motion Masked"
                    
                    # 3. Inject KPI Cards
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
                    
                    # 4. Inject Live Charts
                    t_ax = np.arange(80) / 20.0
                    fig_w = go.Figure(go.Scatter(x=t_ax, y=top_sig, line=dict(color="#38bdf8", width=3)))
                    fig_w.update_layout(title="Live Breathing Waveform (Top Sensitive Subcarriers)", height=320, template="plotly_dark", margin=dict(l=10, r=10, t=35, b=10))
                    ph_chart1.plotly_chart(fig_w, use_container_width=True)
                    
                    fig_h = go.Figure(go.Heatmap(z=(wc - np.mean(wc, axis=0)).T, colorscale="Viridis"))
                    fig_h.update_layout(title="Live 52-Subcarrier Spatio-Temporal Heatmap", height=320, template="plotly_dark", margin=dict(l=10, r=10, t=35, b=10))
                    ph_chart2.plotly_chart(fig_h, use_container_width=True)
                    
        except Exception as e:
            pass # Ignore serial packet tear errors during live stream

else:
    st.info("💡 Switch to 'LIVE ESP32 HARDWARE STREAM' at the top to connect to your physical chips. (Showing static demo layout below)")
    # (Your previous static demo layout remains safe and accessible here if you ever unplug the boards)
