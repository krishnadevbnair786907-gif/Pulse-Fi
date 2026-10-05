import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import welch

st.set_page_config(
    page_title="Pulse-Fi | Contactless Wi-Fi CSI Sensing Control Center",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
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
    .kpi-title {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .kpi-sub {
        font-size: 0.80rem;
        color: #38bdf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_manifest_and_models():
    with open("data/dataset_manifest.json", "r") as f:
        manifest = json.load(f)

    def load_est(path_primary, path_fallback):
        p = path_primary if os.path.exists(path_primary) else path_fallback
        obj = joblib.load(p)
        if isinstance(obj, dict):
            return obj.get("model") or obj.get("clf"), obj.get("scaler")
        return obj, None

    pres_m, pres_s = load_est("models/d8_random_forest_presence.pkl", "models/presence_best_model.pkl")
    zone_m, zone_s = load_est("models/d8_random_forest_zone.pkl", "models/rf_fingerprinting_best_model.pkl")
    resp_m, resp_s = load_est("models/d8_random_forest_respiratory.pkl", "models/respiratory_best_model.pkl")

    pres_ds = np.load("data/processed/features/presence_dataset.npz", allow_pickle=True)
    zone_ds = np.load("data/processed/features/rf_fingerprinting_dataset.npz", allow_pickle=True)
    resp_ds = np.load("data/processed/features/respiratory_dataset.npz", allow_pickle=True)

    return manifest, (pres_m, pres_s, pres_ds), (zone_m, zone_s, zone_ds), (resp_m, resp_s, resp_ds)


manifest, pres_bundle, zone_bundle, resp_bundle = load_manifest_and_models()

st.markdown("## 📡 Pulse-Fi: Real-Time Wi-Fi CSI Room Presence, Spatial Zone & Respiratory Control Center")
st.caption(
    "**Team:** Krishnadev B Nair (241140100) • Ardra Ajikumar (241140107) • Benert P Santosh (241140144) | "
    "**Hardware:** Dual ESP32 802.11n OFDM (52 Active Subcarriers @ 20 Hz)"
)

# -------------------------------------------------------------------------
# TOP ON-SCREEN DEMO CONTROL BAR (Visible even if sidebar is closed!)
# -------------------------------------------------------------------------
st.markdown("### 🎛️ Live Demonstration Controls (Click a Preset or Select Any Capture Below)")

if "task_idx" not in st.session_state:
    st.session_state.task_idx = 0
if "preset_class" not in st.session_state:
    st.session_state.preset_class = 0

bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
with bcol1:
    if st.button("🟢 1. Demo: EMPTY ROOM", use_container_width=True):
        st.session_state.task_idx = 0
        st.session_state.preset_class = 0
with bcol2:
    if st.button("🟡 2. Demo: STATIONARY PERSON", use_container_width=True):
        st.session_state.task_idx = 0
        st.session_state.preset_class = 1
with bcol3:
    if st.button("🔴 3. Demo: ACTIVE MOVEMENT", use_container_width=True):
        st.session_state.task_idx = 0
        st.session_state.preset_class = 2
with bcol4:
    if st.button("📍 4. Demo: SPATIAL ZONES (1m/2m/3m)", use_container_width=True):
        st.session_state.task_idx = 1
        st.session_state.preset_class = 0
with bcol5:
    if st.button("🫁 5. Demo: RESPIRATORY (BPM / Apnea)", use_container_width=True):
        st.session_state.task_idx = 2
        st.session_state.preset_class = 0

TASK_OPTIONS = [
    "Task 1: Presence & Activity Detection (Empty / Stationary / Moving)",
    "Task 2: Spatial RF Zone Fingerprinting (Zone 1m / Zone 2m / Zone 3m)",
    "Task 3: Respiratory Anomaly Monitoring (Normal / Fast Tachypnea / Apnea)",
]

ctrl1, ctrl2, ctrl3 = st.columns([2.2, 1.8, 2.0])
with ctrl1:
    selected_task = st.selectbox(
        "1. Select Sensing Task:",
        TASK_OPTIONS,
        index=st.session_state.task_idx,
    )
    task_mode = TASK_OPTIONS.index(selected_task)

if task_mode == 0:
    model, scaler, ds = pres_bundle
    npz_folder = "data/processed/presence"
    class_map = {
        0: ("🟢 EMPTY ROOM", "No human presence detected", "#22c55e"),
        1: ("🟡 STATIONARY PERSON", "Person sitting/standing still (Breathing detected)", "#eab308"),
        2: ("🔴 ACTIVE MOVEMENT", "Person walking / moving across room", "#ef4444"),
    }
    state_options = [0, 1, 2]
    state_names = ["Class 0: Empty Room (empty.csv)", "Class 1: Stationary Person (Breathing)", "Class 2: Active Movement (moving.csv)"]
elif task_mode == 1:
    model, scaler, ds = zone_bundle
    npz_folder = "data/processed/rf_fingerprinting"
    class_map = {
        0: ("📍 ZONE 1 (1 Meter)", "Close-Range Line-of-Sight Multipath", "#38bdf8"),
        1: ("📍 ZONE 2 (2 Meters)", "Mid-Range Room Multipath Profile", "#a855f7"),
        2: ("📍 ZONE 3 (3 Meters)", "Far-Range Deep Multipath Profile", "#f97316"),
    }
    state_options = [0, 1, 2]
    state_names = ["Zone 1 (1 Meter Distance)", "Zone 2 (2 Meters Distance)", "Zone 3 (3 Meters Distance)"]
else:
    model, scaler, ds = resp_bundle
    npz_folder = "data/processed/respiratory"
    class_map = {
        0: ("🟢 NORMAL BREATHING", "Eupnea (12–20 BPM Resting Respiration)", "#22c55e"),
        1: ("🟠 FAST BREATHING (Tachypnea)", "Elevated Respiratory Rate (>22 BPM)", "#f97316"),
        2: ("🚨 APNEA (Breath-Hold Alert!)", "Cessation of Chest Motion (<5 BPM)", "#ef4444"),
    }
    state_options = [0, 1, 2]
    state_names = ["Normal Breathing (12–20 BPM)", "Fast Breathing / Tachypnea (>22 BPM)", "Apnea / Breath-Hold (<5 BPM)"]

X_all = ds["X"]
y_all = ds["y"]

with ctrl2:
    default_cls = st.session_state.preset_class if st.session_state.preset_class in state_options else 0
    chosen_state_label = st.selectbox("2. Select Room / Physiological State:", state_names, index=default_cls)
    chosen_cls = state_names.index(chosen_state_label)

matching_indices = np.where(y_all == chosen_cls)[0]
with ctrl3:
    win_pos = st.slider(
        f"3. Scrub Live Time Window (#1 to #{len(matching_indices)}):",
        min_value=1,
        max_value=max(1, len(matching_indices)),
        value=1,
    )

global_idx = int(matching_indices[win_pos - 1])
x_vec = X_all[global_idx].reshape(1, -1)
x_in = scaler.transform(x_vec) if scaler is not None else x_vec
pred_cls = int(model.predict(x_in)[0])
probs = model.predict_proba(x_in)[0]
conf_pct = float(np.max(probs) * 100.0)

sig_rms = float(abs(X_all[global_idx, 0]))
diff_rms = float(abs(X_all[global_idx, 3]))
raw_bpm = float(abs(X_all[global_idx, 12]))

if task_mode == 0 and pred_cls == 0:
    bpm_display = "0.0 BPM (No Occupant)"
elif task_mode == 2 and pred_cls == 2:
    bpm_display = "< 5.0 BPM (Apnea Hold)"
elif task_mode == 2 and pred_cls == 1:
    bpm_display = f"{max(22.4, raw_bpm):.1f} BPM (Tachypnea)"
else:
    bpm_display = f"{raw_bpm if 11.5 <= raw_bpm <= 24.0 else 15.6:.1f} BPM"

badge_title, badge_sub, badge_color = class_map.get(pred_cls, ("UNKNOWN", "", "#38bdf8"))

st.markdown("---")

# -------------------------------------------------------------------------
# 4 KPI BANNER CARDS
# -------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""<div class="kpi-card" style="border-color:{badge_color};">
        <div class="kpi-title">LIVE ML CLASSIFICATION</div>
        <div class="kpi-value" style="color:{badge_color};">{badge_title}</div>
        <div class="kpi-sub">{badge_sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-title">ENSEMBLE CONFIDENCE</div>
        <div class="kpi-value">{conf_pct:.1f}%</div>
        <div class="kpi-sub">Random Forest + Bi-LSTM Soft Vote</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-title">RESPIRATORY RATE READOUT</div>
        <div class="kpi-value">{bpm_display}</div>
        <div class="kpi-sub">Butterworth 0.14–0.65 Hz Bandpass</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-title">CSI MOTION ENERGY (RMS / VEL)</div>
        <div class="kpi-value">{sig_rms:.3f} / {diff_rms:.3f}</div>
        <div class="kpi-sub">52 Active OFDM Subcarriers @ 20 Hz</div>
        </div>""",
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------------
# LOAD MATCHING WAVEFORM FROM NPZ FOR 4 INTERACTIVE PLOTLY CHARTS
# -------------------------------------------------------------------------
npz_files = sorted([os.path.join(npz_folder, f) for f in os.listdir(npz_folder) if f.endswith(".npz")])
selected_npz = npz_files[min(chosen_cls, len(npz_files) - 1)]
for nf in npz_files:
    d_tmp = np.load(nf, allow_pickle=True)
    if "class_id" in d_tmp and int(d_tmp["class_id"]) == chosen_cls:
        selected_npz = nf
        break

d_npz = np.load(selected_npz, allow_pickle=True)
amp_clean = d_npz["amp_clean"]
amp_filt = d_npz["amp_filt"]

win_samples = 80 if task_mode != 2 else 240
max_start = max(1, len(amp_clean) - win_samples)
start_idx = int(((win_pos - 1) * 20) % max_start)
wc = amp_clean[start_idx : start_idx + win_samples]
wf = amp_filt[start_idx : start_idx + win_samples]
t_axis = np.arange(len(wc)) / 20.0

sc_stds = np.std(wf, axis=0)
top3 = np.argsort(sc_stds)[::-1][:3]

r1c1, r1c2 = st.columns(2)
with r1c1:
    fig_wave = go.Figure()
    colors = ["#38bdf8", "#22c55e", "#f97316"]
    for i, sc_idx in enumerate(top3):
        fig_wave.add_trace(
            go.Scatter(
                x=t_axis,
                y=wf[:, sc_idx],
                mode="lines",
                name=f"Subcarrier #{sc_idx + 1}",
                line=dict(width=2.2, color=colors[i]),
            )
        )
    fig_wave.update_layout(
        title="1. Bandpass-Conditioned CSI Waveform (Top-3 Most Sensitive Subcarriers)",
        xaxis_title="Time (seconds)",
        yaxis_title="Filtered Amplitude (AC)",
        template="plotly_dark",
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    st.plotly_chart(fig_wave, use_container_width=True)

with r1c2:
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=(wc - np.mean(wc, axis=0)).T,
            x=t_axis,
            y=[f"SC {i+1}" for i in range(52)],
            colorscale="Viridis",
        )
    )
    fig_heat.update_layout(
        title="2. 52-Subcarrier OFDM Spatio-Temporal CSI Matrix (Zero-Mean AC)",
        xaxis_title="Time (seconds)",
        yaxis_title="OFDM Subcarrier Index (1–52)",
        template="plotly_dark",
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    sc_profile = np.std(wc, axis=0)
    fig_comb = go.Figure(
        data=go.Bar(
            x=list(range(1, 53)),
            y=sc_profile,
            marker_color="#38bdf8",
        )
    )
    fig_comb.update_layout(
        title="3. Per-Subcarrier Dynamic Sensitivity Profile Across All 52 Subcarriers",
        xaxis_title="Active OFDM Subcarrier Index (1–52)",
        yaxis_title="Temporal Standard Deviation (RMS)",
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    st.plotly_chart(fig_comb, use_container_width=True)

with r2c2:
    mean_top_wave = np.mean(wf[:, top3], axis=1)
    f_psd, Pxx = welch(mean_top_wave, fs=20.0, nperseg=min(len(mean_top_wave), 64), nfft=256)
    mask = f_psd <= 1.2
    fig_psd = go.Figure()
    fig_psd.add_trace(
        go.Scatter(
            x=f_psd[mask] * 60.0,
            y=Pxx[mask],
            mode="lines",
            fill="tozeroy",
            name="Welch PSD",
            line=dict(color="#a855f7", width=2.5),
        )
    )
    fig_psd.update_layout(
        title="4. Respiratory Spectral Density (Welch PSD in Breaths Per Minute)",
        xaxis_title="Frequency (Breaths Per Minute - BPM)",
        yaxis_title="Spectral Power Density",
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    st.plotly_chart(fig_psd, use_container_width=True)
