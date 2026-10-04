import os
import json
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

ACTIVE_SUBCARRIERS = [i for i in range(6, 59) if i != 32]
TARGET_FS = 20.0


def parse_csi_array(csi_str):
    if not isinstance(csi_str, str):
        return None
    nums = re.findall(r'-?\d+', csi_str)
    if len(nums) != 128:
        return None
    iq = np.array(nums, dtype=np.float32)
    imag = iq[0::2]
    real = iq[1::2]
    amplitudes = np.sqrt(real**2 + imag**2)
    return amplitudes[ACTIVE_SUBCARRIERS]


def hampel_filter_1d(x, window_size=11, n_sigmas=3.0):
    n = len(x)
    new_x = x.copy()
    k = 1.4826
    half_w = window_size // 2
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        window = x[start:end]
        median = np.median(window)
        mad = k * np.median(np.abs(window - median))
        if mad > 1e-6 and np.abs(x[i] - median) > n_sigmas * mad:
            new_x[i] = median
    return new_x


def butter_bandpass(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)


def process_csv_file(filepath, category):
    df = pd.read_csv(filepath, on_bad_lines='skip')
    valid_384 = df[(df['len'] == 384) & (df['mac'] != '00:00:00:00:00:00')].copy()
    df_valid = valid_384 if len(valid_384) >= 50 else df[df['len'] == 128].copy()

    timestamps = pd.to_numeric(df_valid['real_timestamp'], errors='coerce').values
    rssi = pd.to_numeric(df_valid['rssi'], errors='coerce').values

    amp_list, valid_idx = [], []
    for idx, raw_csi in enumerate(df_valid['CSI_DATA']):
        amps = parse_csi_array(raw_csi)
        if amps is not None and not np.isnan(timestamps[idx]):
            amp_list.append(amps)
            valid_idx.append(idx)

    amp_matrix = np.vstack(amp_list)
    t_raw = timestamps[valid_idx]
    rssi_raw = rssi[valid_idx]

    t_raw = t_raw - t_raw[0]
    unique_mask = np.concatenate(([True], np.diff(t_raw) > 1e-4))
    t_raw, amp_matrix, rssi_raw = t_raw[unique_mask], amp_matrix[unique_mask], rssi_raw[unique_mask]

    duration = t_raw[-1]
    t_uniform = np.arange(0, duration, 1.0 / TARGET_FS)

    amp_uniform = interp1d(t_raw, amp_matrix, axis=0, kind='linear', fill_value='extrapolate')(t_uniform)
    rssi_uniform = interp1d(t_raw, rssi_raw, kind='linear', fill_value='extrapolate')(t_uniform)

    amp_hampel = np.apply_along_axis(hampel_filter_1d, 0, amp_uniform, window_size=11, n_sigmas=3.0)
    amp_centered = amp_hampel - np.mean(amp_hampel, axis=0, keepdims=True)

    high_cut = 0.6 if category == 'respiratory' else 4.0
    amp_filtered = butter_bandpass(amp_centered, lowcut=0.1, highcut=high_cut, fs=TARGET_FS, order=4)

    return {
        't_uniform': t_uniform,
        'rssi': rssi_uniform,
        'amp_raw_hampel': amp_hampel,
        'amp_filtered': amp_filtered,
        'raw_packets': len(df),
        'valid_packets': len(t_raw),
        'duration_s': float(duration),
        'mean_rssi': float(np.mean(rssi_raw))
    }


def main():
    with open('data/dataset_manifest.json', 'r') as f:
        manifest = json.load(f)

    os.makedirs('data/processed', exist_ok=True)
    summary_rows = []

    print(f"{'Category':<18} | {'File':<26} | {'Valid/Raw':<11} | {'Dur (s)':<8} | {'Mean RSSI':<10} | {'Output Shape'}")
    print("-" * 98)

    for category, items in manifest['datasets'].items():
        out_dir = os.path.join('data/processed', category)
        os.makedirs(out_dir, exist_ok=True)

        for item in items:
            fpath = item['file']
            fname = os.path.basename(fpath)
            base_id = os.path.splitext(fname)[0]

            res = process_csv_file(fpath, category)
            out_npz = os.path.join(out_dir, f"{base_id}.npz")
            np.savez_compressed(
                out_npz,
                t=res['t_uniform'],
                rssi=res['rssi'],
                amp_clean=res['amp_raw_hampel'],
                amp_filt=res['amp_filtered'],
                label=item['label'],
                class_id=item['class_id']
            )

            pkt_str = f"{res['valid_packets']}/{res['raw_packets']}"
            print(f"{category:<18} | {fname:<26} | {pkt_str:<11} | {res['duration_s']:<8.1f} | {res['mean_rssi']:<10.1f} | {str(res['amp_filtered'].shape)}")

            summary_rows.append({
                'category': category,
                'file': fname,
                'label': item['label'],
                'raw_packets': res['raw_packets'],
                'valid_packets': res['valid_packets'],
                'duration_s': round(res['duration_s'], 2),
                'mean_rssi_dbm': round(res['mean_rssi'], 2),
                'resampled_frames_20hz': res['amp_filtered'].shape[0],
                'active_subcarriers': res['amp_filtered'].shape[1]
            })

    pd.DataFrame(summary_rows).to_csv('data/processed/preprocessing_summary.csv', index=False)
    print("\nDone! All 12 files denoised and saved to data/processed/.")


if __name__ == '__main__':
    main()
