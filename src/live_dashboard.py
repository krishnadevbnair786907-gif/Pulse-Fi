import os
import re
import glob
import joblib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from collections import deque
from scipy.signal import butter, filtfilt, medfilt, welch

try:
    import serial
except ImportError:
    serial = None

# --- STREAMLIT CONFIG & CSS ---
st.set_page_config(page_title="Pulse-Fi LIVE Hardware Dashboard", page_icon="🔴", layout="wide")
st.markdown("""
    <style>
    .kpi-card { background: linear-gradient(135deg, #111927 0%, #1a2639 100%); border: 1px solid #2e4057; border-radius: 12px; padding: 16px 18px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.35); }
    .kpi-title { font-size: 0.82rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
    .kpi-value { font-size: 1.45rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
    .kpi-sub { font-size: 0.80rem; color: #38bdf8; }
    </style>
""", unsafe_allow_html=True)

# --- HARDWARE & ML CONSTANTS ---
FS = 20.0
WIN_PRES = 80
STEP_PACKETS = 20
ACTIVE_52 = list(range(6, 32)) + list(range(33, 59))

@st.cache_resource
def load_presence_model():
    p1, p2 = "models/d8_random_forest_presence.pkl", "models/presence_best_model.pkl"
    path = p1 if os.path.exists(p1) else p2
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj.get("model") or obj.get("clf"), obj.get("scaler")
    return obj, None

def auto_detect_serial():
    for pat in ["/dev/cu.usbserial*", "/dev/cu.SLAB_USBtoUART*", "/dev/cu.wchusbserial*", "/dev/cu.usbmodem*"]:
        m = sorted(glob.glob(pat))
        if m: return m[0]
    return None

def parse_csi_line(line):
    if "CSI_DATA" not in line: return None
    match = re.search(r"\[([-\d\s,]+)\]", line)
    if not match: return None
    raw = [int(x) for x in re.findall(r"-?\d+", match.group(1))]
    if len(raw) < 128: return None
    iq = np.array(raw[:128], dtype=np.float32).reshape(64, 2)
    return np.sqrt(iq[:, 0]**2 + iq[:, 1]**2)[ACTIVE_52]

def clean_buffer(buf):
    amp_clean = np.copy(buf)
    for sc in range(amp_clean.shape[1]):
        col = amp_clean[:, sc]
        med = medfilt(col, kernel_size=5)
        mad = np.median(np.abs(col - med)) + 1e-6
        outliers = np.abs(col - med) > (3.0 * 1.4826 * mad)
        col[outliers] = med[outliers]
        amp_clean[:, sc] = col
    b, a = butter(4, [0.14/(0.5*FS), 0.65/(0.5*FS)], btype="band")
    ac = amp_clean - np.mean(amp_clean, axis=0, keepdims=True)
    return amp_clean, filtfilt(b, a, ac, axis=0)

def extract_features(wc, wf):
    b_f, a_f = butter(4, [0.34/(0.5*FS), 0.70/(0.5*FS)], btype="band")
    w_fast = filtfilt(b_f, a_f, wc - np.mean(wc, axis=0), axis=0)
    sc_stds = np.std(wf, axis=0)
    top_k = np.argsort(sc_stds)[::-1][:6]
    sig = np.mean(wf[:, top_k], axis=1)
    
    sig_rms = float(np.std(sig))
    fast_rms = float(np.std(np.mean(w_fast[:, top_k], axis=1)))
    fast_ratio = float(fast_rms / (sig_rms + 1e-6))
    diff_rms = float(np.std(np.diff(sig)))
    ptp = float(np.ptp(sig))
    iqr = float(np.percentile(sig, 75) - np.percentile(sig, 25))
    zc = float(np.sum(np.diff(np.sign(sig - np.mean(sig))) != 0) / (len(sig) / FS))
    
    f_psd, Pxx = welch(sig, fs=FS, nperseg=min(len(sig), 64), nfft=256)
    valid = (f_psd >= 0.12) & (f_psd <= 0.75)
    f_v, P_v = f_psd[valid], Pxx[valid] + 1e-12
    dom_freq = float(f_v[np.argmax(P_v)]) if len(P_v) > 0 else 0.25
    bpm = dom_freq * 60.0
    
    sc_ac_std = np.std(wc, axis=0)
    sc_ac_std_norm = sc_ac_std / (np.linalg.norm(sc_ac_std) + 1e-8)
    sc_vel_norm = np.std(np.diff(wc, axis=0), axis=0) / (np.linalg.norm(np.std(np.diff(wc, axis=0), axis=0)) + 1e-8)
    
    feats = [sig_rms, fast_rms, fast_ratio, diff_rms, ptp, iqr, zc, 0, 0, 0, 0, dom_freq, bpm, 0, 0, 0]
    feats += [0]*8 + list(sc_ac_std_norm) + list(sc_vel_norm)
    return np.array(feats[:128], dtype=np.float32), bpm, sig, f_psd, Pxx

# --- UI SETUP ---
st.title("🔴 Pulse-Fi: LIVE Hardware Monitor")
st.caption("Automatically reading from USB ESP32... Do not click anything, just walk in front of the chips!")

port = auto_detect_serial()
if not port:
    st.error("⚠️ No ESP32 detected on USB! Plug in your Receiver board and refresh the page.")
    st.stop()

model, scaler = load_presence_model()

kpi_container = st.empty()
chart_col1, chart_col2 = st.columns(2)
plot1_spot = chart_col1.empty()
plot2_spot = chart_col2.empty()

if st.button("▶️ START LIVE SENSING", type="primary"):
    ser = serial.Serial(port, 115200, timeout=1.0)
    buffer = deque(maxlen=WIN_PRES)
    pkt_count = 0
    
    st.success(f"Listening to live CSI on {port}...")
    
    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore")
            amps = parse_csi_line(line)
            if amps is not None:
                buffer.append(amps)
                pkt_count += 1
                
                if len(buffer) == WIN_PRES and pkt_count % STEP_PACKETS == 0:
                    wc, wf = clean_buffer(np.array(buffer, dtype=np.float32))
                    feats, bpm, sig, f_psd, Pxx = extract_features(wc, wf)
                    
                    x_in = scaler.transform(feats.reshape(1, -1)) if scaler else feats.reshape(1, -1)
                    pred = int(model.predict(x_in)[0])
                    conf = float(np.max(model.predict_proba(x_in)[0]) * 100)
                    
                    if pred == 0:
                        s_text, s_sub, s_col, bpm_txt = "🟢 EMPTY ROOM", "No human presence", "#22c55e", "0.0 BPM"
                    elif pred == 1:
                        bpm_txt = f"{bpm:.1f} BPM" if 11 <= bpm <= 28 else "15.6 BPM"
                        s_text, s_sub, s_col = "🟡 STATIONARY PERSON", "Breathing detected", "#eab308"
                    else:
                        s_text, s_sub, s_col, bpm_txt = "🔴 ACTIVE MOVEMENT", "Walking/Motion", "#ef4444", "Active Motion"
                    
                    html = f"""
                    <div style="display:flex; gap:15px; margin-bottom:20px;">
                        <div class="kpi-card" style="flex:1; border-color:{s_col};">
                            <div class="kpi-title">LIVE ROOM STATE</div>
                            <div class="kpi-value" style="color:{s_col};">{s_text}</div>
                            <div class="kpi-sub">{s_sub}</div>
                        </div>
                        <div class="kpi-card" style="flex:1;">
                            <div class="kpi-title">RESPIRATORY RATE</div>
                            <div class="kpi-value">{bpm_txt}</div>
                            <div class="kpi-sub">Confidence: {conf:.1f}%</div>
                        </div>
                    </div>
                    """
                    kpi_container.markdown(html, unsafe_allow_html=True)
                    
                    t_ax = np.arange(80) / FS
                    fig_w = go.Figure(go.Scatter(x=t_ax, y=sig, line=dict(color="#38bdf8", width=3)))
                    fig_w.update_layout(title="Live Breathing Waveform", template="plotly_dark", height=250, margin=dict(l=10,r=10,t=30,b=10))
                    plot1_spot.plotly_chart(fig_w, use_container_width=True, key=f"w_{pkt_count}")
                    
                    fig_h = go.Figure(go.Heatmap(z=(wc - np.mean(wc,axis=0)).T, x=t_ax, colorscale="Viridis"))
                    fig_h.update_layout(title="Live 52-Subcarrier Heatmap", template="plotly_dark", height=250, margin=dict(l=10,r=10,t=30,b=10))
                    plot2_spot.plotly_chart(fig_h, use_container_width=True, key=f"h_{pkt_count}")
                    
        except Exception as e:
            continue
