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

def extract_live_metrics(wc, wf, fs=20.0):
    sc_stds = np.std(wf, axis=0)
    top_k = np.argsort(sc_stds)[::-1][:6]
    sig = np.mean(wf[:, top_k], axis=1)

    sig_rms = float(np.std(sig))
    diff_rms = float(np.std(np.diff(sig)))

    f_psd, Pxx = welch(sig, fs=fs, nperseg=min(len(sig), 64), nfft=256)
    valid = (f_psd >= 0.12) & (f_psd <= 0.75)
    f_v, P_v = f_psd[valid], Pxx[valid] + 1e-12
    dom_freq = float(f_v[np.argmax(P_v)]) if len(P_v) > 0 else 0.25
    est_bpm = float(dom_freq * 60.0)

    return sig_rms, diff_rms, est_bpm, sig

# ---------------- UI LAYOUT ----------------
st.markdown("# 📡 Pulse-Fi: Live Hardware Control Center")
st.caption("Auto-detects ESP32 on macOS USB. Features live sensitivity tuning for variable room distances.")

mode = st.radio("Dashboard Mode:", ["🎛️ Playback Demo (Recorded Files)", "📡 LIVE ESP32 HARDWARE STREAM"], horizontal=True)

if "LIVE" in mode:
    port = auto_detect_mac_serial_port()
    if not port or serial is None:
        st.error("⚠️ No ESP32 detected on USB. Please plug in the receiver board and verify `/dev/cu.usbserial*`.")
        st.stop()
        
    st.markdown("### ⚙️ Live System Calibration & Tuning")
    c_btn1, c_btn2, c_t1, c_t2 = st.columns([1.5, 1.5, 3.5, 3.5])
    with c_btn1:
        if st.button("⏹️ Stop Stream"): st.stop()
    with c_btn2:
        if st.button("🎯 Calibrate Empty"): st.session_state.calib_floor = "PENDING"
    with c_t1:
        stat_thresh = st.slider("🟡 Stationary Trigger Ratio", 1.01, 1.30, 1.05, 0.01, help="If stuck on Empty, lower this.")
    with c_t2:
        move_thresh = st.slider("🔴 Movement Trigger Ratio", 1.10, 2.50, 1.25, 0.05, help="If stuck on Stationary when walking, lower this.")
    
    if "calib_floor" not in st.session_state:
        st.session_state.calib_floor = 0.45
        
    st.success(f"Connected to ESP32 on `{port}`. **Step 1:** Clear the room and click Calibrate. **Step 2:** Watch the Live Ratio below and tune the sliders.")
    
    if "live_buffer" not in st.session_state:
        st.session_state.live_buffer = deque(maxlen=80)
        st.session_state.pred_history = deque(maxlen=9)
        st.session_state.bpm_history = deque(maxlen=5)
        st.session_state.pkt_counter = 0

    try:
        if "ser" in st.session_state and st.session_state.ser.is_open:
            st.session_state.ser.close()
    except Exception:
        pass
        
    try:
        st.session_state.ser = serial.Serial(port, 115200, timeout=1.0)
    except Exception as e:
        st.error(f"Port busy. Unplug and replug the ESP32 USB cable, then refresh the page.")
        st.stop()
    
    k1, k2, k3, k4 = st.columns(4)
    ph_k1, ph_k2, ph_k3, ph_k4 = k1.empty(), k2.empty(), k3.empty(), k4.empty()
    c1, c2 = st.columns(2)
    ph_chart1, ph_chart2 = c1.empty(), c2.empty()
    
    if len(st.session_state.live_buffer) < 80:
        ph_k1.info("⏳ Buffering Live CSI Data... (~4 seconds)")

    while True:
        try:
            raw_line = st.session_state.ser.readline().decode("utf-8", errors="ignore")
            amps = parse_csi_line(raw_line)
            if amps is not None:
                st.session_state.live_buffer.append(amps)
                st.session_state.pkt_counter += 1
                
                if len(st.session_state.live_buffer) == 80 and st.session_state.pkt_counter % 15 == 0:
                    buf_arr = np.array(st.session_state.live_buffer, dtype=np.float32)
                    wc, wf = condition_live_buffer(buf_arr)
                    sig_rms, diff_rms, raw_bpm, top_sig = extract_live_metrics(wc, wf)
                    
                    if st.session_state.calib_floor == "PENDING":
                        st.session_state.calib_floor = max(0.05, sig_rms)
                        
                    motion_ratio = sig_rms / st.session_state.calib_floor
                    vel_ratio = diff_rms / (st.session_state.calib_floor + 1e-6)
                    
                    # TUNED INTERACTIVE THRESHOLDS
                    if motion_ratio >= move_thresh or vel_ratio >= (move_thresh * 0.8):
                        raw_pred_cls = 2
                    elif motion_ratio >= stat_thresh or vel_ratio >= (stat_thresh * 0.9):
                        raw_pred_cls = 1
                    else:
                        raw_pred_cls = 0

                    st.session_state.pred_history.append(raw_pred_cls)
                    pred_cls = max(set(st.session_state.pred_history), key=st.session_state.pred_history.count)
                    
                    if pred_cls == 1: st.session_state.bpm_history.append(raw_bpm)
                    avg_bpm = np.mean(st.session_state.bpm_history) if len(st.session_state.bpm_history) > 0 else 15.6

                    if pred_cls == 0:
                        state_txt, sub_txt, color, bpm_str = "EMPTY ROOM", "No human presence", "#22c55e", "0.0 BPM"
                    elif pred_cls == 1:
                        state_txt, sub_txt, color, bpm_str = "STATIONARY PERSON", "Breathing detected", "#eab308", f"{avg_bpm:.1f} BPM"
                    else:
                        state_txt, sub_txt, color, bpm_str = "ACTIVE MOVEMENT", "Person walking", "#ef4444", "Motion Masked"
                    
                    ph_k1.markdown(f"""<div class="kpi-card" style="border-color:{color};">
                        <div class="kpi-title">LIVE ADAPTIVE ENGINE</div>
                        <div class="kpi-value" style="color:{color};">{'🟢' if pred_cls==0 else '🟡' if pred_cls==1 else '🔴'} {state_txt}</div>
                        <div class="kpi-sub">{sub_txt}</div></div>""", unsafe_allow_html=True)
                    
                    # NEW: Show the actual ratio on screen so the user can tune the sliders perfectly!
                    ph_k2.markdown(f"""<div class="kpi-card"><div class="kpi-title">LIVE SIGNAL RATIO</div>
                        <div class="kpi-value">{motion_ratio:.3f}x</div><div class="kpi-sub">Set sliders based on this number</div></div>""", unsafe_allow_html=True)
                    
                    ph_k3.markdown(f"""<div class="kpi-card"><div class="kpi-title">RESPIRATORY RATE</div>
                        <div class="kpi-value">{bpm_str}</div><div class="kpi-sub">0.14–0.65 Hz Bandpass</div></div>""", unsafe_allow_html=True)
                    ph_k4.markdown(f"""<div class="kpi-card"><div class="kpi-title">CSI ENERGY (RMS / VEL)</div>
                        <div class="kpi-value">{sig_rms:.3f} / {diff_rms:.3f}</div><div class="kpi-sub">Baseline: {st.session_state.calib_floor:.3f}</div></div>""", unsafe_allow_html=True)
                    
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
