import os
import json
import numpy as np
import pandas as pd
from scipy.stats import iqr, kurtosis, skew
from scipy.signal import welch

FS = 20.0  # 20 Hz uniform sampling rate


def select_top_subcarriers(amp_matrix, k=5):
    """Rank subcarriers by temporal variance and return indices of top-k most sensitive."""
    variances = np.var(amp_matrix, axis=0)
    top_idx = np.argsort(variances)[::-1][:k]
    return np.sort(top_idx)


def spectral_entropy(psd):
    """Compute normalized Shannon spectral entropy from power spectral density."""
    psd_norm = psd / (np.sum(psd) + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm)) / np.log2(len(psd) + 1e-12)


def extract_motion_features(win_filt, win_clean, win_rssi):
    """Extract time and frequency features for Presence & Activity Detection."""
    top_sc = select_top_subcarriers(win_filt, k=5)
    sig = win_filt[:, top_sc]

    # Time-domain features across top-5 sensitive subcarriers
    sc_std = np.mean(np.std(sig, axis=0))
    sc_var = np.mean(np.var(sig, axis=0))
    sc_iqr = np.mean(iqr(sig, axis=0))
    sc_rms = np.mean(np.sqrt(np.mean(sig**2, axis=0)))
    sc_max_ptp = np.mean(np.ptp(sig, axis=0))
    sc_kurt = np.mean(kurtosis(sig, axis=0))
    sc_mad = np.mean(np.mean(np.abs(sig - np.mean(sig, axis=0)), axis=0))

    # Zero-crossing rate on zero-mean filtered signal
    zcr = np.mean(np.sum(np.diff(np.signbit(sig), axis=0), axis=0) / len(sig))

    # Cross-subcarrier correlation & full-band energy
    full_std = np.mean(np.std(win_filt, axis=0))
    clean_std = np.mean(np.std(win_clean, axis=0))
    rssi_std = np.std(win_rssi)

    # Frequency-domain energy via Welch PSD
    freqs, psd = welch(sig, fs=FS, nperseg=min(len(sig), 64), axis=0)
    mean_psd = np.mean(psd, axis=1)
    low_band_energy = np.sum(mean_psd[(freqs >= 0.1) & (freqs < 0.8)])
    motion_band_energy = np.sum(mean_psd[(freqs >= 0.8) & (freqs <= 4.0)])
    dom_freq = freqs[np.argmax(mean_psd)]
    spec_ent = spectral_entropy(mean_psd)

    return [
        sc_std, sc_var, sc_iqr, sc_rms, sc_max_ptp, sc_kurt, sc_mad,
        zcr, full_std, clean_std, rssi_std,
        low_band_energy, motion_band_energy, dom_freq, spec_ent
    ]


def extract_rf_fingerprint_features(win_clean, win_filt, win_rssi):
    """
    Extract spatial multipath comb-pattern features across all 52 active subcarriers.
    Combines normalized amplitude profile shape with local attenuation ratios.
    """
    mean_profile = np.mean(win_clean, axis=0)  # (52,) static multipath shape
    std_profile = np.std(win_filt, axis=0)     # (52,) dynamic fluctuation per subcarrier

    # L2-normalized comb shape (captures frequency-selective fading independent of TX power)
    norm_profile = mean_profile / (np.linalg.norm(mean_profile) + 1e-8)

    # Sub-band energy ratios (dividing 52 subcarriers into 4 contiguous frequency blocks)
    b1 = np.mean(mean_profile[0:13])
    b2 = np.mean(mean_profile[13:26])
    b3 = np.mean(mean_profile[26:39])
    b4 = np.mean(mean_profile[39:52])
    total_b = b1 + b2 + b3 + b4 + 1e-8

    profile_slope = np.polyfit(np.arange(52), norm_profile, 1)[0]
    profile_iqr = iqr(norm_profile)
    profile_kurt = kurtosis(norm_profile)
    mean_rssi = np.mean(win_rssi)

    summary_stats = [
        b1 / total_b, b2 / total_b, b3 / total_b, b4 / total_b,
        profile_slope, profile_iqr, profile_kurt,
        np.mean(mean_profile), np.std(mean_profile),
        np.mean(std_profile), mean_rssi
    ]
    return list(norm_profile) + list(mean_profile) + summary_stats


def extract_respiratory_features(win_filt):
    """
    Extract breathing waveform & FFT spectral features for Respiratory Anomaly Detection
    (Normal vs. Fast/Tachypnea vs. Apnea) plus BPM estimation.
    """
    top_sc = select_top_subcarriers(win_filt, k=5)
    sig = win_filt[:, top_sc]
    primary_wave = np.mean(sig, axis=1)

    # Waveform amplitude metrics (crucial for catching Apnea flatlines)
    wave_std = np.std(primary_wave)
    wave_rms = np.sqrt(np.mean(primary_wave**2))
    wave_iqr = iqr(primary_wave)
    wave_ptp = np.ptp(primary_wave)
    wave_kurt = kurtosis(primary_wave)

    # Sub-window dropout ratio: checks if any 5-second sub-segment flatlines (Apnea signature)
    sub_len = int(5 * FS)
    sub_stds = [np.std(primary_wave[i:i + sub_len]) for i in range(0, len(primary_wave) - sub_len + 1, sub_len // 2)]
    min_max_std_ratio = (np.min(sub_stds) / (np.max(sub_stds) + 1e-8)) if len(sub_stds) > 1 else 1.0
    min_sub_std = np.min(sub_stds) if len(sub_stds) > 0 else wave_std

    # Zero-crossing rate (directly proportional to breathing frequency)
    zcr = np.sum(np.diff(np.signbit(primary_wave))) / (len(primary_wave) / FS)

    # High-resolution padded FFT for accurate BPM & band power ratio
    n_fft = 1024
    fft_vals = np.abs(np.fft.rfft(primary_wave * np.hanning(len(primary_wave)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / FS)
    psd = fft_vals**2

    resp_mask = (freqs >= 0.1) & (freqs <= 0.65)
    resp_freqs = freqs[resp_mask]
    resp_psd = psd[resp_mask]

    dom_freq = resp_freqs[np.argmax(resp_psd)] if len(resp_psd) > 0 else 0.25
    est_bpm = dom_freq * 60.0

    # Spectral power in normal band (0.15-0.35 Hz / 9-21 BPM) vs fast band (0.35-0.65 Hz / 21-39 BPM)
    normal_band_power = np.sum(psd[(freqs >= 0.15) & (freqs <= 0.35)])
    fast_band_power = np.sum(psd[(freqs > 0.35) & (freqs <= 0.65)])
    total_resp_power = normal_band_power + fast_band_power + 1e-8

    fast_to_normal_ratio = fast_band_power / (normal_band_power + 1e-8)
    normal_power_frac = normal_band_power / total_resp_power
    spec_ent = spectral_entropy(resp_psd)

    return [
        wave_std, wave_rms, wave_iqr, wave_ptp, wave_kurt,
        min_max_std_ratio, min_sub_std, zcr,
        dom_freq, est_bpm, normal_band_power, fast_band_power,
        fast_to_normal_ratio, normal_power_frac, spec_ent
    ]


def build_datasets():
    with open('data/dataset_manifest.json', 'r') as f:
        manifest = json.load(f)

    os.makedirs('data/processed/features', exist_ok=True)

    # 1. Presence & Motion Dataset (4-second window = 80 samples, 50% overlap = 40 samples)
    # Classes: 0 = Empty, 1 = Stationary Human (Zones 1-3), 2 = Active Movement
    X_pres, y_pres, meta_pres = [], [], []
    win_p, step_p = int(4 * FS), int(2 * FS)

    pres_sources = (
        [(item, 0, 'empty') for item in manifest['datasets']['presence'] if item['label'] == 'empty'] +
        [(item, 1, 'stationary') for item in manifest['datasets']['rf_fingerprinting']] +
        [(item, 2, 'moving') for item in manifest['datasets']['presence'] if item['label'] == 'moving']
    )

    for item, cls_id, cls_name in pres_sources:
        base_id = os.path.splitext(os.path.basename(item['file']))[0]
        cat_dir = 'rf_fingerprinting' if 'zone' in base_id else 'presence'
        data = np.load(f"data/processed/{cat_dir}/{base_id}.npz")
        amp_f, amp_c, rssi = data['amp_filt'], data['amp_clean'], data['rssi']

        for start in range(0, len(amp_f) - win_p + 1, step_p):
            feats = extract_motion_features(
                amp_f[start:start + win_p],
                amp_c[start:start + win_p],
                rssi[start:start + win_p]
            )
            X_pres.append(feats)
            y_pres.append(cls_id)
            meta_pres.append((base_id, cls_name))

    X_pres, y_pres = np.array(X_pres, dtype=np.float32), np.array(y_pres, dtype=np.int64)
    np.savez_compressed('data/processed/features/presence_dataset.npz', X=X_pres, y=y_pres, meta=meta_pres)

    # 2. Spatial RF Fingerprinting Dataset (3-second window = 60 samples, step = 15 samples)
    # Classes: 0 = Zone 1 (1m), 1 = Zone 2 (2m), 2 = Zone 3 (3m)
    X_rf, y_rf, meta_rf = [], [], []
    win_r, step_r = int(3 * FS), int(0.75 * FS)

    for item in manifest['datasets']['rf_fingerprinting']:
        base_id = os.path.splitext(os.path.basename(item['file']))[0]
        data = np.load(f"data/processed/rf_fingerprinting/{base_id}.npz")
        amp_f, amp_c, rssi = data['amp_filt'], data['amp_clean'], data['rssi']
        cls_id = int(item['class_id'])

        for start in range(0, len(amp_c) - win_r + 1, step_r):
            feats = extract_rf_fingerprint_features(
                amp_c[start:start + win_r],
                amp_f[start:start + win_r],
                rssi[start:start + win_r]
            )
            X_rf.append(feats)
            y_rf.append(cls_id)
            meta_rf.append((base_id, item['label']))

    X_rf, y_rf = np.array(X_rf, dtype=np.float32), np.array(y_rf, dtype=np.int64)
    np.savez_compressed('data/processed/features/rf_fingerprinting_dataset.npz', X=X_rf, y=y_rf, meta=meta_rf)

    # 3. Respiratory Anomaly Dataset (15-second window = 300 samples, step = 20 samples / 1s)
    # Classes: 0 = Normal Breathing, 1 = Fast Breathing (Tachypnea), 2 = Apnea / Breath-Hold
    X_resp, y_resp, meta_resp = [], [], []
    win_resp, step_resp = int(15 * FS), int(1.0 * FS)

    for item in manifest['datasets']['respiratory']:
        base_id = os.path.splitext(os.path.basename(item['file']))[0]
        data = np.load(f"data/processed/respiratory/{base_id}.npz")
        amp_f = data['amp_filt']
        cls_id = int(item['class_id'])

        for start in range(0, len(amp_f) - win_resp + 1, step_resp):
            feats = extract_respiratory_features(amp_f[start:start + win_resp])
            X_resp.append(feats)
            y_resp.append(cls_id)
            meta_resp.append((base_id, item['label']))

    X_resp, y_resp = np.array(X_resp, dtype=np.float32), np.array(y_resp, dtype=np.int64)
    np.savez_compressed('data/processed/features/respiratory_dataset.npz', X=X_resp, y=y_resp, meta=meta_resp)

    print("=" * 80)
    print("SLIDING-WINDOW FEATURE EXTRACTION SUMMARY (DAY 6)")
    print("=" * 80)
    print(f"1. Presence & Motion Dataset   : X = {str(X_pres.shape):<12} | Classes (Empty/Stat/Move): {np.bincount(y_pres)}")
    print(f"2. Spatial RF Fingerprinting   : X = {str(X_rf.shape):<12} | Classes (1m / 2m / 3m)   : {np.bincount(y_rf)}")
    print(f"3. Respiratory Anomaly Dataset : X = {str(X_resp.shape):<12} | Classes (Norm/Fast/Apnea): {np.bincount(y_resp)}")
    print("=" * 80)


if __name__ == '__main__':
    build_datasets()
