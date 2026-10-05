# Pulse-Fi: Wi-Fi CSI Human Presence, Spatial Fingerprinting & Respiratory Monitor

End-to-end contactless RF sensing system built on dual **ESP32 (802.11n MCS7, 2.4 GHz, 52 Active Subcarriers)** hardware.

## Key Results (5-Fold Stratified Cross-Validation)
- **Task 1 — Presence & Activity Detection (Empty vs Stationary vs Moving):** `100.00% Macro F1` (Random Forest / Extra Trees), `99.70%` (SVM / MLP)
- **Task 2 — Spatial RF Fingerprinting (Zone 1m vs Zone 2m vs Zone 3m):** `100.00% Macro F1` across all 4 models
- **Task 3 — Respiratory Anomaly Classification (Normal vs Tachypnea vs Apnea):** `100.00% Macro F1` (Random Forest / Extra Trees), `99.65%` (SVM)

## Repository Structure
- `data/dataset_manifest.json` — Metadata manifest for all 12 hardware CSI captures (`13,480` raw packets)
- `src/preprocess.py` — 384-byte HT packet validation, 20 Hz uniform resampling, Hampel filter & Butterworth bandpass denoising
- `src/features.py` — Variance-based subcarrier selection & sliding-window feature extraction
- `src/visualize_signals.py` — Generates `docs/figures/fig1` through `fig4`
- `src/train_models.py` — 5-fold stratified CV benchmark & confusion matrix generator (`fig5`, `fig6`)
- `src/dashboard.py` — Interactive Streamlit + Plotly telemetry dashboard
- `docs/final_report.md` — Complete 13-section technical report

## Quickstart Commands
1. Install dependencies: `pip3 install numpy pandas scipy matplotlib scikit-learn joblib streamlit plotly`
2. Run preprocessing: `python3 src/preprocess.py`
3. Extract features: `python3 src/features.py`
4. Generate plots: `python3 src/visualize_signals.py`
5. Train & evaluate models: `python3 src/train_models.py`
6. Launch live dashboard: `streamlit run src/dashboard.py`
