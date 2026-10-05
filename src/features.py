import os
import json
import numpy as np
from scipy.stats import iqr, kurtosis
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
    """Combine temporal fluctuation features with spatial multipath attenuation profile."""
    centered = win_clean - np.mean(win_clean, axis=0, keepdims=True)
    top_sc = select_top_subcarriers(win_filt, k=8)
    sig = win_filt[:, top_sc]

    diff1 = np.diff(sig, axis=0)
    diff_all = np.diff(centered, axis=0)

    sc_stds = np.std(win_filt, axis=0)
    sc_iqrs = iqr(win_filt, axis=0)
    clean_stds = np.std(centered, axis=0)

    corr_mat = np.nan_to_num(np.corrcoef(win_filt.T), nan=0.0)
    off_diag_corr = (np.sum(np.abs(corr_mat)) - len(corr_mat)) / (len(corr_mat) * (len(corr_mat) - 1))

    _, s_vals, _ = np.linalg.svd(win_filt, full_matrices=False)
    s_norm = (s_vals ** 2) / (np.sum(s_vals ** 2) + 1e-12)

    freqs, psd = welch(sig, fs=FS, nperseg=min(len(sig), 64), axis=0)
    mean_psd = np.mean(psd, axis=1)
    e_resp = np.sum(mean_psd[(freqs >= 0.1) & (freqs < 0.6)])
    e_walk = np.sum(mean_psd[(freqs >= 0.6) & (freqs <= 4.0)])
    e_total = e_resp + e_walk + 1e-12

    zcr = np.mean(np.sum(np.diff(np.signbit(sig), axis=0), axis=0) / len(sig))

    # Spatial multipath profile (distinguishes Empty Room static reflections from Stationary Human body absorption)
    mean_profile = np.mean(win_clean, axis=0)
    norm_profile = mean_profile / (np.linalg.norm(mean_profile) + 1e-8)

    temporal_feats = [
        np.mean(sc_stds), np.max(sc_stds), np.percentile(sc_stds, 90), np.percentile(sc_stds, 25),
        np.mean(sc_iqrs), np.max(sc_iqrs),
        np.mean(clean_stds), np.max(clean_stds),
        np.mean(np.abs(diff1)), np.std(diff1), np.mean(np.abs(diff_all)),
        np.mean(np.ptp(sig, axis=0)), np.mean(kurtosis(sig, axis=0)),
        zcr, off_diag_corr, s_norm[0], s_norm[1] if len(s_norm) > 1 else 0.0,
        e_resp, e_walk, e_walk / e_total,
        freqs[np.argmax(mean_psd)], spectral_entropy(mean_psd),
        np.std(win_rssi), iqr(win_rssi)
    ]
    return temporal_feats + list(norm_profile) + list(sc_stds)


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


RESP_FEATURE_NAMES = [
    # 1. Physiologically grounded scalar features (from pruning table)
    'sig_rms',               # 0: Excursion depth (wideband RMS)
    'fast_rms',              # 1: Tachypnea band energy (0.36-0.75 Hz)
    'fast_to_sig_rms_ratio', # 2: Ratio of rapid-to-wideband RMS
    'diff_rms',              # 3: Chest wall velocity RMS
    'iqr',                   # 4: Interquartile amplitude spread
    'ptp',                   # 5: Peak-to-peak wave swing
    'min_sub_std',           # 6: Apnea flatline detector (min 3s sub-window std)
    'max_sub_std',           # 7: Apnea flatline detector (max 3s sub-window std)
    'dropout_ratio',         # 8: Apnea dropout index (min_sub_std / max_sub_std)
    'sub_std_cv',            # 9: Sub-window coefficient of variation
    'zcr',                   # 10: Smoothed zero-crossing rate
    'dom_freq',              # 11: Fundamental breathing frequency (Hz)
    'est_bpm',               # 12: Estimated breathing rate (BPM)
    'p_norm_frac',           # 13: Normal band power fraction (0.18-0.35 Hz)
    'p_fast_frac',           # 14: Fast band power fraction (0.36-0.75 Hz)
    'log_p_fast_to_norm',    # 15: Log-transformed fast/normal power ratio
    'spectral_entropy',      # 16: Normalized Shannon spectral entropy
    # 2. Zero-mean AC dynamic subcarrier fluctuation & velocity profile (L2-normalized, zero static DC bias)
    'ac_subband1_energy', 'ac_subband2_energy', 'ac_subband3_energy', 'ac_subband4_energy',
    'ac_vel_subband1', 'ac_vel_subband2', 'ac_vel_subband3', 'ac_vel_subband4'
] + [f'ac_norm_std_sc{i+1}' for i in range(52)] + [f'ac_norm_vel_sc{i+1}' for i in range(52)]


def extract_respiratory_features(win_clean, win_filt=None, return_diagnostics=False):
    centered = win_clean - np.mean(win_clean, axis=0, keepdims=True)
    if win_filt is None:
        win_filt = butter_bandpass(centered, lowcut=0.14, highcut=0.65, fs=FS, order=4)

    win_fast = butter_bandpass(centered, lowcut=0.36, highcut=0.75, fs=FS, order=4)
    win_smooth = butter_bandpass(centered, lowcut=0.15, highcut=0.52, fs=FS, order=4)

    top_sc = select_top_subcarriers(win_filt, k=6)
    sig = win_filt[:, top_sc]
    fast_sig = win_fast[:, top_sc]
    smooth_sig = win_smooth[:, top_sc]

    sub_len = int(3.0 * FS)
    sub_stds = [np.std(sig[i:i + sub_len]) for i in range(0, len(sig) - sub_len + 1, int(1.0 * FS))]
    min_sub_std, max_sub_std = np.min(sub_stds), np.max(sub_stds)
    dropout_ratio = min_sub_std / (max_sub_std + 1e-8)
    sub_std_cv = np.std(sub_stds) / (np.mean(sub_stds) + 1e-8)

    diff1 = np.diff(sig, axis=0)
    diff_rms = np.mean(np.sqrt(np.mean(diff1**2, axis=0)))
    sig_rms = np.mean(np.sqrt(np.mean(sig**2, axis=0)))
    fast_rms = np.mean(np.sqrt(np.mean(fast_sig**2, axis=0)))
    fast_to_sig_ratio = fast_rms / (sig_rms + 1e-8)

    zcr = np.mean(np.sum(np.diff(np.signbit(smooth_sig), axis=0), axis=0) / (len(smooth_sig) / FS))

    # Zero-mean AC dynamic fluctuation & velocity across all 52 subcarriers (L2-normalized -> zero RSSI/DC bias)
    sc_std = np.std(win_filt, axis=0)
    sc_std_norm = sc_std / (np.linalg.norm(sc_std) + 1e-8)
    sc_vel = np.std(np.diff(win_filt, axis=0), axis=0)
    sc_vel_norm = sc_vel / (np.linalg.norm(sc_vel) + 1e-8)

    ac_sb = [np.mean(sc_std_norm[i*13:(i+1)*13]) for i in range(4)]
    vel_sb = [np.mean(sc_vel_norm[i*13:(i+1)*13]) for i in range(4)]

    # Multi-subcarrier FFT spectrum
    n_fft = 2048
    windowed = sig * np.hanning(len(sig))[:, None]
    fft_mag = np.abs(np.fft.rfft(windowed, n=n_fft, axis=0))
    psd = np.mean(fft_mag ** 2, axis=1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / FS)

    p_norm = np.sum(psd[(freqs >= 0.18) & (freqs <= 0.35)])
    p_fast = np.sum(psd[(freqs > 0.35) & (freqs <= 0.75)])
    p_tot = p_norm + p_fast + 1e-8

    # Select fundamental resting peak (12-20 BPM) vs rapid tachypnea peak (23-35 BPM)
    # using AC dynamic velocity dispersion (no static DC spatial stats used!)
    norm_mask = (freqs >= 0.19) & (freqs <= 0.32)
    fast_mask = (freqs >= 0.38) & (freqs <= 0.58)
    ac_high_to_low = (ac_sb[2] + ac_sb[3]) / (ac_sb[0] + ac_sb[1] + 1e-8)
    if ac_high_to_low > 0.96 and sig_rms < 0.58:
        dom_freq = freqs[fast_mask][np.argmax(psd[fast_mask])]
    else:
        dom_freq = freqs[norm_mask][np.argmax(psd[norm_mask])]

    est_bpm = dom_freq * 60.0
    log_p_fast_to_norm = np.log1p(p_fast / (p_norm + 1e-8))
    resp_psd = psd[(freqs >= 0.16) & (freqs <= 0.75)]

    feats = [
        sig_rms, fast_rms, fast_to_sig_ratio, diff_rms,
        np.mean(iqr(sig, axis=0)), np.mean(np.ptp(sig, axis=0)),
        min_sub_std, max_sub_std, dropout_ratio, sub_std_cv,
        zcr, dom_freq, est_bpm,
        p_norm / p_tot, p_fast / p_tot, log_p_fast_to_norm,
        spectral_entropy(resp_psd)
    ] + ac_sb + vel_sb + list(sc_std_norm) + list(sc_vel_norm)

    if return_diagnostics:
        all_sc_mean = np.mean(win_clean, axis=0)
        sc_spatial_ratio = np.mean(all_sc_mean[:26]) / (np.mean(all_sc_mean[26:]) + 1e-8)
        sc_spatial_std = np.std(all_sc_mean)
        return feats, [sc_spatial_ratio, sc_spatial_std]
    return feats


def build_datasets():
    with open('data/dataset_manifest.json', 'r') as f:
        manifest = json.load(f)

    os.makedirs('data/processed/features', exist_ok=True)

    # 1. Presence & Motion Dataset
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

    # 2. Spatial RF Fingerprinting Dataset
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

    # 3. Respiratory Anomaly Dataset
    X_resp, y_resp, meta_resp = [], [], []
    win_resp = int(12 * FS)

    for item in manifest['datasets']['respiratory']:
        base_id = os.path.splitext(os.path.basename(item['file']))[0]
        data = np.load(f"data/processed/respiratory/{base_id}.npz")
        amp_c = data['amp_clean']
        amp_f = data['amp_filt']
        cls_id = int(item['class_id'])

        step_resp = int(2.0 * FS) if cls_id == 0 else int(0.6 * FS)
        for start in range(0, len(amp_c) - win_resp + 1, step_resp):
            X_resp.append(extract_respiratory_features(amp_c[start:start + win_resp], amp_f[start:start + win_resp]))
            y_resp.append(cls_id)
            meta_resp.append((base_id, item['label']))

    X_resp, y_resp = np.array(X_resp, dtype=np.float32), np.array(y_resp, dtype=np.int64)
    np.savez_compressed('data/processed/features/respiratory_dataset.npz', X=X_resp, y=y_resp, meta=meta_resp)

    print("=" * 85)
    print("FINAL FEATURE MATRICES")
    print("=" * 85)
    print(f"1. Presence & Motion Dataset   : X = {str(X_pres.shape):<12} | Classes (Empty/Stat/Move): {np.bincount(y_pres)}")
    print(f"2. Spatial RF Fingerprinting   : X = {str(X_rf.shape):<12} | Classes (1m / 2m / 3m)   : {np.bincount(y_rf)}")
    print(f"3. Respiratory Anomaly Dataset : X = {str(X_resp.shape):<12} | Classes (Norm/Fast/Apnea): {np.bincount(y_resp)}")
    print("=" * 85)


if __name__ == '__main__':
    build_datasets()
