# Pulse-Fi: Wi-Fi CSI-Based Human Presence, Spatial Fingerprinting, and Respiratory Monitoring

## 1. Abstract
- Summary of contactless RF sensing using commodity ESP32 Wi-Fi hardware, key signal processing stages, machine learning classifiers, and overall accuracy metrics.

## 2. Introduction & Motivation
- Limitations of camera-based (privacy) and wearable-based (compliance) monitoring.
- Advantages of ubiquitous Wi-Fi Channel State Information (CSI) over coarse RSSI.

## 3. Literature Survey & Related Work
- Evolution of Wi-Fi sensing from NIC toolkits (Intel 5300) to microcontroller-based ESP32 CSI systems.
- Prior approaches to presence detection, indoor localization, and vital sign estimation.

## 4. Mathematical & Physical Background of Wi-Fi CSI
- OFDM Channel Frequency Response $H(f_k, t)$ representation:
  $$H(f_k, t) = \vert{}H(f_k, t)\vert{} e^{j \angle H(f_k, t)}$$
- Multipath propagation, Fresnel zone diffraction, and chest-wall micro-displacement effects on subcarrier amplitude.

## 5. System Architecture & Hardware Setup
- Dual ESP32 active transmitter-receiver topology (`active_ap` and `active_sta`).
- Packet configuration: 2.4 GHz Channel 6, 802.11n HT20/HT40, 64 subcarriers (128 I/Q bytes), ~11–25 Hz sampling rate.

## 6. Data Collection Protocol & Dataset Organization
- Experimental room geometry and fixed TX-RX baseline separation.
- Summary of the 12-file dataset across Presence, Spatial RF Fingerprinting (Zones 1–3), and Respiratory Monitoring (Normal, Fast/Tachypnea, Apnea).

## 7. Signal Preprocessing & Denoising Pipeline
- Packet filtering (MAC validation, `len == 384` HT frame selection) and I/Q amplitude extraction:
  $$A_k = \sqrt{I_k^2 + Q_k^2}$$
- DC offset removal, Hampel outlier identifier, and Butterworth bandpass filtering ($0.1\text{–}0.6\text{ Hz}$ for respiration; $0.5\text{–}4.0\text{ Hz}$ for motion).

## 8. Feature Engineering & Subcarrier Selection
- Variance-based subcarrier sensitivity ranking.
- Time-domain features (variance, IQR, RMS, zero-crossing rate, kurtosis) and frequency-domain FFT spectral features (dominant breathing frequency, PSD ratio, spectral entropy).

## 9. Machine Learning Models & Classification Pipeline
- Presence & Motion Classifier (Empty vs. Stationary vs. Moving).
- Spatial RF Fingerprinting Classifier (Zone 1m vs. Zone 2m vs. Zone 3m).
- Respiratory Anomaly Classifier (Normal vs. Fast Breathing vs. Apnea) + BPM Estimator.

## 10. Experimental Results & Performance Evaluation
- Cross-validation accuracy, precision, recall, F1-scores, and confusion matrices.
- Breathing rate estimation error (MAE in Breaths Per Minute).

## 11. Real-Time Visualization & Dashboard Implementation
- Architecture of the live monitoring and evaluation dashboard.

## 12. Limitations, Challenges & Future Scope
- Sensitivity to non-target ambient motion, multi-person separation, and 5 GHz / Wi-Fi 6E extensions.

## 13. Conclusion & References
- Summary of achievements and IEEE-formatted bibliography.
