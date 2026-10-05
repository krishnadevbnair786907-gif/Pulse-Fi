# Pulse-Fi: Contactless Wi-Fi CSI Presence, Spatial Zone & Respiratory Anomaly Sensing

**Pulse-Fi** is an end-to-end contactless RF sensing and machine learning system built on **ESP32 802.11n Channel State Information (CSI)**. By extracting amplitude dynamics across 52 active OFDM subcarriers, cleaning multipath phase/amplitude artifacts via Hampel spike rejection and zero-phase Butterworth bandpass filtering, and engineering a **leak-free 129-feature physiological + zero-mean AC dynamic profile**, Pulse-Fi performs three simultaneous sensing tasks without any wearable sensor:

1. **Presence & Activity Detection:** `Empty Room` vs. `Stationary Person` vs. `Active Movement` (**100.00% 5-Fold Macro F1**)
2. **Spatial RF Zone Fingerprinting:** `Zone 1 (1m)` vs. `Zone 2 (2m)` vs. `Zone 3 (3m)` (**100.00% 5-Fold Macro F1**)
3. **Contactless Respiratory Anomaly & BPM Sensing:** `Normal (12-20 BPM)` vs. `Fast / Tachypnea (>22 BPM)` vs. `Apnea (Hold)` (**100.00% Tuned Random Forest / Extra Trees / Bi-Recurrent Sequence F1**)

---

## Team & Deliverables Architecture (`D1` - `D8`)

| Contributor | Role & Deliverable Track | Core Scripts | Exported Datasets & Models | Figures |
| :--- | :--- | :--- | :--- | :--- |
| **Krishna** | **End-to-End Architecture, Hardware CSI Pipeline, Leak-Free Feature Engineering, Multi-Task Benchmark & Live Streamlit App** | `src/preprocess.py`<br>`src/features.py`<br>`src/train_models.py`<br>`src/generate_plots.py`<br>`src/dashboard.py` | `data/processed/preprocessing_summary.csv`<br>`data/processed/model_metrics.csv`<br>`data/processed/features/respiratory_features_named.csv`<br>`data/processed/features/session_bias_diagnostics.csv`<br>`models/*_best_model.pkl` | `fig1` - `fig6` |
| **Ardra** | **Days 1-4 (`D1-D4`) Signal Conditioning, Subcarrier Ranking & Trajectory Engineering + Day 5 (`D5`) Temporal Sequence Classifier** | `src/ardra_d1_to_d4_pipeline.py`<br>`src/d5_lstm_sequence_classifier.py`<br>`src/export_d5_d8_deliverables.py` | `data/processed/d1_temporal_jitter_audit.csv`<br>`data/processed/d2_denoising_snr_audit.csv`<br>`data/processed/d3_subcarrier_sensitivity.csv`<br>`data/processed/d4_temporal_step_trajectories.csv`<br>`data/processed/features/lstm_respiratory_sequences.npz`<br>`data/processed/d5_sequence_model_metrics.csv`<br>`data/processed/d5_per_class_report.csv`<br>`models/d5_temporal_sequence_model.pkl` | `fig9`, `fig10` |
| **Benert** | **Day 8 (`D8`) Random Forest Respiratory & Activity Classifier, 3-Stage Feature Ablation Study & OOB Convergence Analysis** | `src/d8_random_forest_classifier.py`<br>`src/export_d5_d8_deliverables.py` | `data/processed/d8_ablation_study.csv`<br>`data/processed/d8_per_class_report.csv`<br>`models/d8_random_forest_respiratory.pkl`<br>`models/d8_random_forest_presence.pkl` | `fig7`, `fig8` |

---

## Key Experimental Results

### 1. Multi-Task Benchmark Summary (`data/processed/model_metrics.csv`)

| Sensing Task | Windows | Features | Best Model | 5-Fold CV Accuracy | 5-Fold Macro F1 |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **Task 1: Presence & Motion** | `312` | `128` | Random Forest / Extra Trees | **100.00%** | **100.00%** |
| **Task 2: Spatial RF Zone (1m/2m/3m)** | `149` | `128` | Random Forest / Extra Trees / SVM | **100.00%** | **100.00%** |
| **Task 3: Respiratory Anomaly** | `367` | `129` (Leak-Free) | Tuned RF (`D8`) / Bi-Recurrent (`D5`) / Extra Trees | **100.00%** | **100.00%** |

### 2. Ardra's `D1`-`D4` Signal Conditioning & `D5` Temporal Sequence Benchmark

* **D1 (Packet & Resampling Audit):** Audited `12,492` raw ESP32 CSI packets across 12 captures (mean arrival rate `12.37 Hz`) and validated cubic-spline interpolation onto a uniform `20.0 Hz` grid (`dt = 50 ms`).
* **D2 (Hampel & Bandpass SNR Gain):** Zero-phase 4th-order Butterworth bandpass filtering (`0.14-0.65 Hz`) improved in-band respiratory SNR from **`-6.31 dB` to `+9.75 dB`** (mean **`+16.07 dB` SNR gain**).
* **D3 (Subcarrier Sensitivity & Phase Cancellation):** Top-6 sensitive subcarrier selection preserves **`1.23x` to `1.35x` higher respiratory RMS amplitude** compared to naive 52-subcarrier averaging.
* **D4 & D5 (3D Sequence Tensor `367 x 10 x 17` & Temporal Sequence Classifier):**

| Sequence Architecture (`src/d5_lstm_sequence_classifier.py`) | Horizon | Features / Step | 5-Fold Accuracy | 5-Fold Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1:** Single 3s Step Snapshot (`t1` only) | `1 step (3s)` | `17` | 51.77% | 49.32% |
| **Model 2:** 10-Step Unrolled Sequence MLP (`t1..t10`) | `10 steps (12s)` | `17` | 85.01% | 84.43% |
| **Model 3 (`D5 Final`):** Bi-Recurrent Gate + Temporal Transition Net | `10 steps (12s)` | `129` | **100.00%** | **100.00%** |

### 3. Benert's `D8` Random Forest 3-Stage Feature Ablation Study (`data/processed/d8_ablation_study.csv`)

| Feature Configuration (`src/d8_random_forest_classifier.py`) | Features | 5-Fold CV Accuracy | 5-Fold Macro F1 | Out-of-Bag (OOB) Score |
| :--- | :---: | :---: | :---: | :---: |
| **Config A:** 17 Physiological Scalars Only (`est_bpm`, `sig_rms`, `dropout_ratio`, etc.) | `17` | 85.29% | 83.25% | 86.92% |
| **Config B:** 112 Zero-Mean L2-Normalized AC Dynamic Subcarrier Profile | `112` | 99.73% | 99.75% | 99.46% |
| **Config C (`D8 Final`):** Combined 129 Leak-Free Features (`GridSearchCV` tuned) | `129` | **100.00%** (`99.73%` default) | **100.00%** (`99.75%` default) | **100.00%** |

---

## Visual Diagnostic Gallery (`docs/figures/fig1` - `fig10`)

### Figure 1-6: Core Multi-Task Signal & Model Diagnostics (Krishna)
![Figure 1: Presence Waveforms](docs/figures/fig1_presence_waveforms.png)
![Figure 2: Spatial RF Zone Profiles](docs/figures/fig2_rf_zone_profiles.png)
![Figure 3: Respiratory Time-Frequency Analysis](docs/figures/fig3_respiratory_time_freq.png)
![Figure 4: Breathing Rate Distribution](docs/figures/fig4_bpm_distribution.png)
![Figure 5: PCA Feature Clusters](docs/figures/fig5_pca_feature_clusters.png)
![Figure 6: Multi-Task Confusion Matrices](docs/figures/fig6_confusion_matrices.png)

### Figure 7-8: Deliverable D8 Random Forest & Ablation Diagnostics (Benert)
![Figure 7: D8 Random Forest Feature Importance](docs/figures/fig7_rf_feature_importance.png)
![Figure 8: D8 Random Forest Confusion Matrix and OOB Convergence](docs/figures/fig8_d8_rf_analysis.png)

### Figure 9-10: Deliverables D1-D5 Signal Conditioning & Temporal Sequence Analysis (Ardra)
![Figure 9: Ardra D1-D4 Signal Conditioning and Trajectory Pipeline](docs/figures/fig9_ardra_d1_d4_pipeline.png)
![Figure 10: Ardra D5 Temporal Sequence Confusion Matrix and Horizon Scaling](docs/figures/fig10_d5_sequence_analysis.png)

---

## Quick Start & Reproducibility

    # 1. Clone the repository
    git clone https://github.com/krishnadevbnair786907-gif/Pulse-Fi.git
    cd Pulse-Fi

    # 2. Launch the Interactive Streamlit Live Dashboard
    ~/.espressif/python_env/idf5.2_py3.14_env/bin/streamlit run src/dashboard.py

    # 3. Re-run the complete pipeline end-to-end
    PY=~/.espressif/python_env/idf5.2_py3.14_env/bin/python3
    $PY src/preprocess.py
    $PY src/features.py
    $PY src/train_models.py
    $PY src/generate_plots.py
    $PY src/export_d5_d8_deliverables.py
    $PY src/d8_random_forest_classifier.py
    $PY src/ardra_d1_to_d4_pipeline.py
    $PY src/d5_lstm_sequence_classifier.py
