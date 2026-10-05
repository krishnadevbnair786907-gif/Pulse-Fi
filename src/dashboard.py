import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from features import (
    extract_motion_features,
    extract_rf_fingerprint_features,
    extract_respiratory_features,
    select_top_subcarriers,
    FS
)

st.set_page_config(
    page_title="Pulse-Fi | Wi-Fi CSI Sensing Dashboard",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Pulse-Fi: Contactless Wi-Fi CSI Human Presence, Localization & Respiratory Monitor")
st.caption("Dual ESP32 802.11n (MCS7, 2.4 GHz, 52 Active OFDM Subcarriers @ 20 Hz Uniform Resampling)")


@st.cache_resource
def load_models():
    models = {}
    for task in ['presence', 'rf_fingerprinting', 'respiratory']:
        path = f"models/{task}_best_model.pkl"
        if os.path.exists(path):
            models[task] = joblib.load(path)
    return models


@st.cache_data
def load_manifest_and_metrics():
    with open('data/dataset_manifest.json', 'r') as f:
        manifest = json.load(f)
    metrics_df = pd.read_csv('data/processed/model_metrics.csv')
    summary_df = pd.read_csv('data/processed/preprocessing_summary.csv')
    return manifest, metrics_df, summary_df


models = load_models()
manifest, metrics_df, summary_df = load_manifest_and_metrics()

# Sidebar controls
st.sidebar.header("🎛️ Capture Playback & Window Inspector")
category = st.sidebar.selectbox(
    "Select Experiment Domain",
    options=['respiratory', 'rf_fingerprinting', 'presence'],
    format_func=lambda x: {
        'respiratory': '🫁 Respiratory Anomaly Monitoring',
        'rf_fingerprinting': '📍 Spatial RF Fingerprinting (Zones 1–3)',
        'presence': '🚶 Presence & Motion Detection'
    }[x]
)

files_in_cat = manifest['datasets'][category]
selected_item = st.sidebar.selectbox(
    "Select Hardware Capture File",
    options=files_in_cat,
    format_func=lambda item: f"{os.path.basename(item['file'])} ({item['label']})"
)

base_id = os.path.splitext(os.path.basename(selected_item['file']))[0]
npz_path = f"data/processed/{category}/{base_id}.npz"
data = np.load(npz_path)

t = data['t']
amp_clean = data['amp_clean']
amp_filt = data['amp_filt']
rssi = data['rssi']
max_time = float(t[-1])

win_sec = 12.0 if category == 'respiratory' else (3.0 if category == 'rf_fingerprinting' else 4.0)
start_sec = st.sidebar.slider(
    f"Sliding Window Start Time (Window = {win_sec:.0f}s)",
    min_value=0.0,
    max_value=max(0.5, round(max_time - win_sec, 1)),
    value=min(10.0, max(0.0, round(max_time - win_sec, 1))),
    step=0.5
)

start_idx = int(start_sec * FS)
end_idx = min(len(t), start_idx + int(win_sec * FS))

win_t = t[start_idx:end_idx]
win_clean = amp_clean[start_idx:end_idx]
win_filt = amp_filt[start_idx:end_idx]
win_rssi = rssi[start_idx:end_idx]

# Run real-time inference on the selected window
pres_feats = np.array(extract_motion_features(win_filt, win_clean, win_rssi)).reshape(1, -1)
pres_pred_idx = models['presence']['model'].predict(pres_feats)[0]
pres_label = models['presence']['classes'][pres_pred_idx]

rf_feats = np.array(extract_rf_fingerprint_features(win_clean, win_filt, win_rssi)).reshape(1, -1)
rf_pred_idx = models['rf_fingerprinting']['model'].predict(rf_feats)[0]
rf_label = models['rf_fingerprinting']['classes'][rf_pred_idx]

resp_feats_vec = extract_respiratory_features(win_clean)
resp_feats = np.array(resp_feats_vec).reshape(1, -1)
resp_pred_idx = models['respiratory']['model'].predict(resp_feats)[0]
resp_label = models['respiratory']['classes'][resp_pred_idx]
est_bpm = resp_feats_vec[14] if resp_pred_idx != 2 else 0.0

# Top KPI Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("🚶 Presence & Activity State", pres_label, f"CV F1: {models['presence']['cv_f1_pct']}%")
col2.metric("📍 Spatial Zone Fingerprint", rf_label, f"CV F1: {models['rf_fingerprinting']['cv_f1_pct']}%")
col3.metric("🫁 Respiratory Classification", resp_label, f"CV F1: {models['respiratory']['cv_f1_pct']}%")
col4.metric("💓 Estimated Breathing Rate", f"{est_bpm:.1f} BPM" if resp_pred_idx != 2 else "APNEA (<5 BPM)", f"Mean RSSI: {np.mean(win_rssi):.1f} dBm")

st.divider()

# Interactive Plotly Visualizations
top_sc = select_top_subcarriers(win_filt, k=3)
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        f"Filtered CSI Waveform (Top Subcarriers #{top_sc[0]}, #{top_sc[1]}, #{top_sc[2]})",
        "52-Subcarrier Spatial Comb Profile (Mean ± SD)",
        "52-Subcarrier Spatio-Temporal CSI Heatmap",
        "Respiratory / Motion FFT Spectrum"
    ),
    vertical_spacing=0.14,
    horizontal_spacing=0.08
)

# 1. Filtered Time-Domain Waveform
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for idx, sc in enumerate(top_sc):
    fig.add_trace(
        go.Scatter(x=win_t, y=win_filt[:, sc], mode='lines', name=f"Subcarrier #{sc}", line=dict(color=colors[idx], width=2)),
        row=1, col=1
    )

# 2. 52-Subcarrier Comb Fingerprint
subcarrier_ids = np.arange(1, 53)
mean_profile = np.mean(win_clean, axis=0)
fig.add_trace(
    go.Scatter(x=subcarrier_ids, y=mean_profile, mode='lines+markers', name="Mean CSI Amp", line=dict(color='#9467bd', width=2.5)),
    row=1, col=2
)

# 3. Spatio-Temporal CSI Heatmap
fig.add_trace(
    go.Heatmap(
        z=win_filt.T,
        x=win_t,
        y=subcarrier_ids,
        colorscale='Viridis',
        showscale=False
    ),
    row=2, col=1
)

# 4. Windowed FFT Spectrum
sig_primary = win_filt[:, top_sc[0]]
n_fft = 1024
fft_mag = np.abs(np.fft.rfft(sig_primary * np.hanning(len(sig_primary)), n=n_fft))
freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / FS)
mask_f = (freqs_hz >= 0.1) & (freqs_hz <= 2.5)

fig.add_trace(
    go.Scatter(x=freqs_hz[mask_f] * 60.0, y=fft_mag[mask_f], mode='lines', name="FFT Magnitude", line=dict(color='#d62728', width=2), fill='tozeroy'),
    row=2, col=2
)

fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Filtered Amplitude", row=1, col=1)
fig.update_xaxes(title_text="Active Subcarrier Index (1–52)", row=1, col=2)
fig.update_yaxes(title_text="CSI Amplitude", row=1, col=2)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_yaxes(title_text="Subcarrier (1–52)", row=2, col=1)
fig.update_xaxes(title_text="Frequency (Cycles / Breaths Per Minute)", row=2, col=2)
fig.update_yaxes(title_text="Spectral Energy", row=2, col=2)
fig.update_layout(height=680, showlegend=True)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 View Full 5-Fold Cross-Validation Benchmark & Dataset Summary"):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("5-Fold Stratified CV Classifier Metrics")
        st.dataframe(metrics_df, use_container_width=True)
    with c2:
        st.subheader("Hardware Capture Preprocessing Summary")
        st.dataframe(summary_df, use_container_width=True)
