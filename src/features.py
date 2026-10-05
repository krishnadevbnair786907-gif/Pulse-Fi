import os
import json
import numpy as np
from scipy.stats import iqr, kurtosis, skew
from scipy.signal import butter, filtfilt, welch

FS = 20.0


def butter_bandpass(data, lowcut, highcut, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [max(lowcut / nyq, 0.001), min(highcut / nyq, 0.999)], btype='band')
    return filtfilt(b, a, data, axis=0)


def select_top_subcarriers(amp_matrix, k=8):
    variances = np.var(amp_matrix, axis=0)
    return np.sort(np.argsort(variances)[::-1][:k])


def spectral_entropy(psd):
    psd_norm = psd / (np.sum(psd) + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm)) / np.log2(len(psd) + 1e-12)


def extract_motion_features(win_filt, win_clean, win_rssi):
    """Rich spatial-temporal + spectral features for Presence & Activity Detection."""
    # Center clean signal per window to remove static room baseline
    centered = win_clean - np.mean(win_clean, axis=0, keepdims=True)
    top_sc = select_top_subcarriers(win_filt, k=8)
    sig = win_filt[:, top_sc]

    # First-order temporal derivative (velocity of multipath change)
    diff1 = np.diff(sig, axis=0)
    diff_all = np.diff(centered, axis=0)

    # Subcarrier statistics (percentiles across all 52 subcarriers)
    sc_stds = np.std(win_filt, axis=0)
    sc_iqrs = iqr(win_filt, axis=0)
    clean_stds = np.std(centered, axis=0)

    # Cross-subcarrier covariance/correlation structure (walking correlates all subcarriers)
    corr_mat = np.corrcoef(win_filt.T)
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)
    off_diag_corr = (np.sum(np.abs(corr_mat)) - len(corr_mat)) / (len(corr_mat) * (len(corr_mat) - 1))

    # SVD / Principal Component energy concentration
    _, s_vals, _ = np.linalg.svd(win_filt, full_matrices=False)
    s_norm = (s_vals ** 2) / (np.sum(s_vals ** 2) + 1e-12)
    pc1_ratio = s_norm[0]
    pc2_ratio = s_norm[1] if len(s_norm) > 1 else 0.0

    # Welch PSD across bands
    freqs, psd = welch(sig, fs=FS, nperseg=min(len(sig), 64), axis=0)
    mean_psd = np.mean(psd, axis=1)
    e_resp = np.sum(mean_psd[(freqs >= 0.1) & (freqs < 0.6)])
    e_walk = np.sum(mean_psd[(freqs >= 0.6) & (freqs <= 4.0)])
    e_total = e_resp + e_walk + 1e-12

    zcr = np.mean(np.sum(np.diff(np.signbit(sig), axis=0), axis=0) / len(sig))

    return [
        np.mean(sc_stds), np.max(sc_stds), np.percentile(sc_stds, 90), np.percentile(sc_stds, 25),
        np.mean(sc_iqrs), np.max(sc_iqrs),
        np.mean(clean_stds), np.max(clean_stds),
        np.mean(np.abs(diff1)), np.std(diff1), np.mean(np.abs(diff_all)),
        np.mean(np.ptp(sig, axis=0)), np.mean(kurtosis(sig, axis=0)),
        zcr, off_diag_corr, pc1_ratio, pc2_ratio,
        e_resp, e_walk, e_walk / e_total,
        freqs[np.argmax(mean_psd)], spectral_entropy(mean_psd),
        np.std(win_rssi), iqr(win_rssi)
    ]


def extract_rf_fingerprint_features(win_clean, win_filt, win_rssi):
    mean_profile = np.mean(win_clean, axis=0)
    std_profile = np.std(win_filt, axis=0)
    norm_profile = mean_profile / (np.linalg.norm(mean_profile) + 1e-8)

    b1, b2, b3, b4 = [np.mean(mean_profile[i*13:(i+1)*13]) for i in range(4)]
    total_b = b1 + b2 + b3 + b4 + 1e-8
    profile_slope = np.polyfit(np.arange(52), norm_profile, 1)[0]

    summary_stats = [
        b1 / total_b, b2 / total_b, b3 / total_b, b4 / total_b,
        profile_slope, iqr(norm_profile), kurtosis(norm_profile),
        np.mean(mean_profile), np.std(mean_profile),
        np.mean(std_profile), np.mean(win_rssi)
    ]
    return list(norm_profile) + list(mean_profile) + summary_stats


def extract_respiratory_features(win_clean):
    """
    Extract multi-band respiratory features using a wider 0.15-1.0 Hz (9-60 BPM) passband
    plus high-frequency tachypnea energy (0.40-0.95 Hz / 24-57 BPM) and apnea dropout ratios.
    """
    centered = win_clean - np.mean(win_clean, axis=0, keepdims=True)
    # Wide respiratory passband (9 to 57 BPM) to preserve rapid shallow breathing harmonics
    win_resp = butter_bandpass(centered, lowcut=0.15, highcut=0.95, fs=FS, order=4)
    # Dedicated rapid-breathing passband (22 to 57 BPM)
    win_fast = butter_bandpass(centered, lowcut=0.37, highcut=0.95, fs=FS, order=4)

    top_sc = select_top_subcarriers(win_resp, k=6)
    sig = win_resp[:, top_sc]
    fast_sig = win_fast[:, top_sc]

    # Align phase across top subcarriers via SVD first principal component
    u, s, _ = np.linalg.svd(sig, full_matrices=False)
    primary_wave = u[:, 0] * s[0]

    # Sub-window dropout analysis (3-second sliding sub-windows to catch Apnea flatlines)
    sub_len = int(3.0 * FS)
    sub_stds = [np.std(sig[i:i + sub_len]) for i in range(0, len(sig) - sub_len + 1, int(1.0 * FS))]
    min_sub_std = np.min(sub_stds)
    max_sub_std = np.max(sub_stds)
    dropout_ratio = min_sub_std / (max_sub_std + 1e-8)
    sub_std_cv = np.std(sub_stds) / (np.mean(sub_stds) + 1e-8)

    # First derivative energy (captures rapid chest reversal in tachypnea)
    diff1 = np.diff(sig, axis=0)
    diff_rms = np.mean(np.sqrt(np.mean(diff1**2, axis=0)))
    sig_rms = np.mean(np.sqrt(np.mean(sig**2, axis=0)))
    fast_rms = np.mean(np.sqrt(np.mean(fast_sig**2, axis=0)))

    # Mean ZCR across top subcarriers
    zcr = np.mean(np.sum(np.diff(np.signbit(sig), axis=0), axis=0) / (len(sig) / FS))

    # Multi-subcarrier averaged FFT spectrum
    n_fft = 2048
    windowed = sig * np.hanning(len(sig))[:, None]
    fft_mag = np.abs(np.fft.rfft(windowed, n=n_fft, axis=0))
    psd = np.mean(fft_mag ** 2, axis=1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / FS)

    # Weight spectrum slightly against <0.18 Hz posture drift when picking dominant breathing peak
    resp_mask = (freqs >= 0.18) & (freqs <= 0.90)
    resp_freqs = freqs[resp_mask]
    resp_psd = psd[resp_mask]

    dom_freq = resp_freqs[np.argmax(resp_psd)]
    est_bpm = dom_freq * 60.0

    # Spectral band powers
    p_slow = np.sum(psd[(freqs >= 0.12) & (freqs < 0.20)])     # 7 - 12 BPM (Apnea recovery / drift)
    p_norm = np.sum(psd[(freqs >= 0.20) & (freqs <= 0.36)])    # 12 - 21.6 BPM (Normal breathing)
    p_fast = np.sum(psd[(freqs > 0.36) & (freqs <= 0.90)])     # 21.6 - 54 BPM (Fast / Tachypnea)
    p_tot = p_slow + p_norm + p_fast + 1e-8

    # Estimate BPM using upper-band peak when fast-band energy ratio is elevated
    if (p_fast / p_tot) > 0.28:
        fast_mask = (freqs >= 0.36) & (freqs <= 0.85)
        est_bpm = freqs[fast_mask][np.argmax(psd[fast_mask])] * 60.0

    # Subcarrier distribution stats (captures whole-chest vs local multipath signature)
    all_sc_mean = np.mean(win_clean, axis=0)
    sc_spatial_ratio = np.mean(all_sc_mean[:26]) / (np.mean(all_sc_mean[26:]) + 1e-8)
    sc_spatial_std = np.std(all_sc_mean)

    return [
        sig_rms, fast_rms, fast_rms / (sig_rms + 1e-8),
        diff_rms, diff_rms / (sig_rms + 1e-8),
        np.mean(iqr(sig, axis=0)), np.mean(np.ptp(sig, axis=0)), np.mean(kurtosis(sig, axis=0)),
        min_sub_std, max_sub_std, dropout_ratio, sub_std_cv,
        zcr, dom_freq, est_bpm,
        p_slow / p_tot, p_norm / p_tot, p_fast / p_tot,
        p_fast / (p_norm + 1e-8), spectral_entropy(resp_psd),
        sc_spatial_ratio, sc_spatial_std
    ]


def build_datasets():
    with open('data/dataset_manifest.json', 'r') as f:
        manifest = json.load(f)

    os.makedirs('data/processed/features', exist_ok=True)

    # 1. Presence & Motion Dataset (4s window, 1.5s step)
    # Use matched -54dBm empty_02 baseline + clean stationary zones + active movement
    X_pres, y_pres, meta_pres = [], [], []
    win_p, step_p = int(4 * FS), int(1.5 * FS)

    pres_sources = (
        [(item, 0, 'empty') for item in manifest['datasets']['presence'] if 'empty_02' in item['file']] +
        [(item, 1, 'stationary') for item in manifest['datasets']['rf_fingerprinting']] +
        [(item, 2, 'moving') for item in manifest['datasets']['presence'] if item['label'] == 'moving']
    )

    for item, cls_id, cls_name in pres_sources:
        base_id = os.path.splitext(os.path.basename(item['file']))[0]
        cat_dir = 'rf_fingerprinting' if 'zone' in base_id else 'presence'
        data = np.load(f"data/processed/{cat_dir}/{base_id}.npz")
        amp_f, amp_c, rssi = data['amp_filt'], data['amp_clean'], data['rssi']

        # Use 1.0s step for empty_02 so Empty class count (~86) matches Stationary & Moving
        cur_step = int(1.0 * FS) if cls_id == 0 else step_p
        for start in range(0, len(amp_f) - win_p + 1, cur_step):
            X_pres.append(extract_motion_features(
                amp_f[start:start + win_p],
                amp_c[start:start + win_p],
                rssi[start:start + win_p]
            ))
            y_pres.append(cls_id)
            meta_pres.append((base_id, cls_name))

    X_pres, y_pres = np.array(X_pres, dtype=np.float32), np.array(y_pres, dtype=np.int64)
    np.savez_compressed('data/processed/features/presence_dataset.npz', X=X_pres, y=y_pres, meta=meta_pres)

    # 2. Spatial RF Fingerprinting Dataset (3s window, 0.75s step)
    X_rf, y_rf, meta_rf = [], [], []
    win_r, step_r = int(3 * FS), int(0.75 * FS)

    for item in manifest['datasets']['rf_fingerprinting']:
        base_id = os.path.splitext(os.path.basename(item['file']))[0]
        data = np.load(f"data/processed/rf_fingerprinting/{base_id}.npz")
        amp_f, amp_c, rssi = data['amp_filt'], data['amp_clean'], data['rssi']
        cls_id = int(item['class_id'])

        for start in range(0, len(amp_c) - win_r + 1, step_r):
            X_rf.append(extract_rf_fingerprint_features(
                amp_c[start:start + win_r],
                amp_f[start:start + win_r],
                rssi[start:start + win_r]
            ))
            y_rf.append(cls_id)
            meta_rf.append((base_id, item['label']))

    X_rf, y_rf = np.array(X_rf, dtype=np.float32), np.array(y_rf, dtype=np.int64)
    np.savez_compressed('data/processed/features/rf_fingerprinting_dataset.npz', X=X_rf, y=y_rf, meta=meta_rf)

    # 3. Respiratory Anomaly Dataset (12s window = 240 samples)
    # Adaptive step size to balance Normal (~150 windows) with Fast (~70) and Apnea (~90)
    X_resp, y_resp, meta_resp = [], [], []
    win_resp = int(12 * FS)

    for item in manifest['datasets']['respiratory']:
        base_id = os.path.splitext(os.path.basename(item['file']))[0]
        data = np.load(f"data/processed/respiratory/{base_id}.npz")
        amp_c = data['amp_clean']
        cls_id = int(item['class_id'])

        step_resp = int(2.0 * FS) if cls_id == 0 else int(0.6 * FS)
        for start in range(0, len(amp_c) - win_resp + 1, step_resp):
            X_resp.append(extract_respiratory_features(amp_c[start:start + win_resp]))
            y_resp.append(cls_id)
            meta_resp.append((base_id, item['label']))

    X_resp, y_resp = np.array(X_resp, dtype=np.float32), np.array(y_resp, dtype=np.int64)
    np.savez_compressed('data/processed/features/respiratory_dataset.npz', X=X_resp, y=y_resp, meta=meta_resp)

    print("=" * 85)
    print("OPTIMIZED SLIDING-WINDOW FEATURE MATRICES")
    print("=" * 85)
    print(f"1. Presence & Motion Dataset   : X = {str(X_pres.shape):<12} | Classes (Empty/Stat/Move): {np.bincount(y_pres)}")
    print(f"2. Spatial RF Fingerprinting   : X = {str(X_rf.shape):<12} | Classes (1m / 2m / 3m)   : {np.bincount(y_rf)}")
    print(f"3. Respiratory Anomaly Dataset : X = {str(X_resp.shape):<12} | Classes (Norm/Fast/Apnea): {np.bincount(y_resp)}")
    print("=" * 85)


if __name__ == '__main__':
    build_datasets()
