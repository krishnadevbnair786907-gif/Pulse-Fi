import os
import numpy as np
import matplotlib.pyplot as plt

FS = 20.0
os.makedirs('docs/figures', exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})


def get_best_subcarrier(amp_filt):
    return int(np.argmax(np.var(amp_filt, axis=0)))


def plot_presence_comparison():
    empty = np.load('data/processed/presence/empty_02.npz')
    stat = np.load('data/processed/rf_fingerprinting/zone1_1m.npz')
    move = np.load('data/processed/presence/moving_02.npz')

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, sharey=True)
    datasets = [
        (empty, 'Empty Room Baseline (empty_02.csv) — Flat Multipath Floor', '#2b5c8f'),
        (stat, 'Stationary Presence at 1m (zone1_1m.csv) — Micro-Postural & Chest Ripples', '#2ca02c'),
        (move, 'Active Walking Across Zones (moving_02.csv) — High-Amplitude Multipath Fluctuations', '#d62728')
    ]

    for ax, (data, title, color) in zip(axes, datasets):
        t = data['t']
        mask = t <= 45.0
        sc = get_best_subcarrier(data['amp_filt'])
        ax.plot(t[mask], data['amp_filt'][mask, sc], color=color, lw=1.3)
        ax.set_title(f"{title} (Subcarrier #{sc}, Variance = {np.var(data['amp_filt'][mask, sc]):.3f})", fontweight='bold')
        ax.set_ylabel('Filtered CSI Amp')

    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig('docs/figures/fig1_presence_activity_comparison.png', dpi=200)
    plt.close()


def plot_rf_fingerprints():
    z1 = np.load('data/processed/rf_fingerprinting/zone1_1m.npz')
    z2 = np.load('data/processed/rf_fingerprinting/zone2_2m.npz')
    z3 = np.load('data/processed/rf_fingerprinting/zone3_3m.npz')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    subcarriers = np.arange(1, 53)

    for data, label, color in [
        (z1, 'Zone 1 (1.0m LoS)', '#1f77b4'),
        (z2, 'Zone 2 (2.0m Mid-Range)', '#ff7f0e'),
        (z3, 'Zone 3 (3.0m Far Corner)', '#2ca02c')
    ]:
        mean_amp = np.mean(data['amp_clean'], axis=0)
        std_amp = np.std(data['amp_clean'], axis=0)
        norm_amp = mean_amp / np.linalg.norm(mean_amp)

        ax1.plot(subcarriers, mean_amp, label=label, color=color, lw=2)
        ax1.fill_between(subcarriers, mean_amp - std_amp, mean_amp + std_amp, color=color, alpha=0.15)
        ax2.plot(subcarriers, norm_amp, label=label, color=color, lw=2, marker='o', markersize=3)

    ax1.set_title('Raw Hampel-Cleaned CSI Comb Profile (Mean ± 1 SD)', fontweight='bold')
    ax1.set_xlabel('Active OFDM Subcarrier Index (1 to 52)')
    ax1.set_ylabel('CSI Amplitude (A = sqrt(I² + Q²))')
    ax1.legend()

    ax2.set_title('L2-Normalized Multipath Fingerprint Signature', fontweight='bold')
    ax2.set_xlabel('Active OFDM Subcarrier Index (1 to 52)')
    ax2.set_ylabel('Normalized Subcarrier Energy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('docs/figures/fig2_spatial_rf_fingerprints.png', dpi=200)
    plt.close()


def plot_respiratory_waveforms_and_fft():
    norm = np.load('data/processed/respiratory/normal_breathing_01.npz')
    fast = np.load('data/processed/respiratory/fast_breathing_01.npz')
    apnea = np.load('data/processed/respiratory/apnea_01.npz')

    fig, axes = plt.subplots(3, 2, figsize=(14, 8.5))
    records = [
        (norm, 'Normal Resting Respiration (normal_breathing_01)', '#1f77b4'),
        (fast, 'Tachypnea / Fast Shallow Breathing (fast_breathing_01)', '#ff7f0e'),
        (apnea, 'Apnea / Periodic Breath-Hold Intervals (apnea_01)', '#d62728')
    ]

    for row_idx, (data, title, color) in enumerate(records):
        t = data['t']
        mask = (t >= 5.0) & (t <= 55.0)
        t_win = t[mask]
        sc = get_best_subcarrier(data['amp_filt'][mask])
        sig = data['amp_filt'][mask, sc]

        # Time waveform plot
        ax_t = axes[row_idx, 0]
        ax_t.plot(t_win, sig, color=color, lw=1.6)
        ax_t.set_title(f"{title} — Subcarrier #{sc}", fontweight='bold')
        ax_t.set_ylabel('Bandpass Amp\n(0.1–0.6 Hz)')
        if row_idx == 2:
            ax_t.set_xlabel('Time (seconds)')

        # FFT Spectrum plot
        ax_f = axes[row_idx, 1]
        n_fft = 2048
        fft_mag = np.abs(np.fft.rfft(sig * np.hanning(len(sig)), n=n_fft))
        freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / FS)
        freqs_bpm = freqs_hz * 60.0

        f_mask = (freqs_bpm >= 5.0) & (freqs_bpm <= 42.0)
        peak_idx = np.argmax(fft_mag[f_mask])
        peak_bpm = freqs_bpm[f_mask][peak_idx]
        peak_hz = freqs_hz[f_mask][peak_idx]

        ax_f.plot(freqs_bpm[f_mask], fft_mag[f_mask], color=color, lw=1.8)
        ax_f.axvline(peak_bpm, color='black', linestyle='--', lw=1.2, label=f'Peak: {peak_bpm:.1f} BPM ({peak_hz:.2f} Hz)')
        ax_f.set_title(f"FFT Respiratory Spectrum — Est. Rate: {peak_bpm:.1f} BPM", fontweight='bold')
        ax_f.set_ylabel('Spectral Magnitude')
        ax_f.legend(loc='upper right')
        if row_idx == 2:
            ax_f.set_xlabel('Breathing Rate (Breaths Per Minute - BPM)')

    plt.tight_layout()
    plt.savefig('docs/figures/fig3_respiratory_waveforms_and_fft.png', dpi=200)
    plt.close()


def plot_denoising_stages():
    norm = np.load('data/processed/respiratory/normal_breathing_01.npz')
    t = norm['t']
    mask = (t >= 10.0) & (t <= 40.0)
    sc = get_best_subcarrier(norm['amp_filt'][mask])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax1.plot(t[mask], norm['amp_clean'][mask, sc], color='#7f7f7f', lw=1.2, label='Stage 1: 20 Hz Resampled + Hampel Outlier Cleaned')
    ax1.set_title(f'CSI Subcarrier #{sc} Before vs. After Butterworth Bandpass Denoising (0.1–0.6 Hz)', fontweight='bold')
    ax1.set_ylabel('Raw CSI Amplitude')
    ax1.legend(loc='upper right')

    ax2.plot(t[mask], norm['amp_filt'][mask, sc], color='#1f77b4', lw=1.8, label='Stage 2: DC-Removed + 4th-Order Butterworth Bandpass (0.1–0.6 Hz)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Filtered Wave Amp')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('docs/figures/fig4_preprocessing_before_after.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    plot_presence_comparison()
    plot_rf_fingerprints()
    plot_respiratory_waveforms_and_fft()
    plot_denoising_stages()
    print("\nSaved 4 verification figures to docs/figures/:")
    for f in sorted(os.listdir('docs/figures')):
        print(f"  - docs/figures/{f}")
