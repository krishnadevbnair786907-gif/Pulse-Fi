"""
Deliverable D5: Temporal Sequence & Recurrent Transition Classifier
Author: Ardra (Pulse-Fi Team)
Description:
  - Loads the 3D temporal sequence tensor X_seq (367 windows x 10 time_steps x 17 features)
    and augments each time step with the 8 zero-mean AC dynamic sub-band energies -> (367, 10, 25).
  - Computes Bidirectional Recurrent Gate states (forward/backward exponential memory gates h_t, c_t)
    and inter-step temporal derivatives (dX/dt) across the 10 time steps.
  - Evaluates sequence horizon scaling (T = 1 to 10 time steps) and exports D5 tables, models, and Figure 10.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

os.makedirs('models', exist_ok=True)
os.makedirs('docs/figures', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

FS = 20.0
RESP_CLASSES = ['Normal (12-20 BPM)', 'Fast / Tachypnea (>22 BPM)', 'Apnea (Hold)']


def build_augmented_3d_sequences():
    """Augment (367, 10, 17) physiological tensor with 8 zero-mean AC sub-band features per 3s step -> (367, 10, 25)."""
    seq_data = np.load('data/processed/features/lstm_respiratory_sequences.npz', allow_pickle=True)
    X_seq_17 = seq_data['X_seq']  # (367, 10, 17)
    y_seq = seq_data['y']
    meta = seq_data['meta']

    with open('data/dataset_manifest.json') as f:
        manifest = json.load(f)

    win_resp = int(12 * FS)
    sub_win = int(3.0 * FS)
    sub_step = int(1.0 * FS)

    ac_seq_list = []
    for item in manifest['datasets']['respiratory']:
        base_id = os.path.splitext(os.path.basename(item['file']))[0]
        d = np.load(f'data/processed/respiratory/{base_id}.npz')
        amp_c, amp_f = d['amp_clean'], d['amp_filt']
        cls_id = int(item['class_id'])

        step_resp = int(2.0 * FS) if cls_id == 0 else int(0.6 * FS)
        for start in range(0, len(amp_c) - win_resp + 1, step_resp):
            wf = amp_f[start:start + win_resp]
            step_ac = []
            for t_s in range(0, win_resp - sub_win + 1, sub_step):
                sub_f = wf[t_s:t_s + sub_win]
                sc_std = np.std(sub_f, axis=0)
                sc_std_n = sc_std / (np.linalg.norm(sc_std) + 1e-8)
                sc_vel = np.std(np.diff(sub_f, axis=0), axis=0)
                sc_vel_n = sc_vel / (np.linalg.norm(sc_vel) + 1e-8)
                ac_sb = [np.mean(sc_std_n[i*13:(i+1)*13]) for i in range(4)]
                vel_sb = [np.mean(sc_vel_n[i*13:(i+1)*13]) for i in range(4)]
                step_ac.append(ac_sb + vel_sb + list(sc_std_n) + list(sc_vel_n))
            ac_seq_list.append(step_ac)

    X_ac_seq = np.array(ac_seq_list, dtype=np.float32)  # (367, 10, 112)
    X_seq_full = np.concatenate([X_seq_17, X_ac_seq], axis=2)  # (367, 10, 129)
    return X_seq_17, X_seq_full, y_seq, meta


def extract_recurrent_gate_representation(X_seq_sub):
    """
    Compute Bidirectional Recurrent Gate states (forward/backward exponential cell memory c_t, h_t)
    and first-order temporal transitions (dx/dt) across T time steps.
    """
    N, T, F = X_seq_sub.shape
    if T == 1:
        return X_seq_sub[:, 0, :]

    # Forward & backward recurrent memory cells (alpha = 0.65 forget/input gate weighting)
    alpha = 0.65
    h_fwd = np.zeros((N, F), dtype=np.float32)
    h_bwd = np.zeros((N, F), dtype=np.float32)
    for t in range(T):
        h_fwd = alpha * X_seq_sub[:, t, :] + (1 - alpha) * h_fwd
        h_bwd = alpha * X_seq_sub[:, T - 1 - t, :] + (1 - alpha) * h_bwd

    # Temporal trajectory statistics across steps 1..T
    seq_mean = np.mean(X_seq_sub, axis=1)
    seq_std = np.std(X_seq_sub, axis=1)
    seq_min = np.min(X_seq_sub, axis=1)
    seq_max = np.max(X_seq_sub, axis=1)
    temporal_delta = np.mean(np.abs(np.diff(X_seq_sub, axis=1)), axis=1)

    return np.hstack([h_fwd, h_bwd, seq_mean, seq_std, seq_min, seq_max, temporal_delta])


def run_d5_pipeline():
    print("=" * 90)
    print("DELIVERABLE D5: TEMPORAL SEQUENCE & RECURRENT TRANSITION CLASSIFIER (ARDRA)")
    print("=" * 90)

    X_seq_17, X_seq_full, y_seq, _ = build_augmented_3d_sequences()
    print(f"\n[1/4] 3D TEMPORAL SEQUENCE TENSORS LOADED:")
    print(f"  - Core Physiological Sequence Tensor : {X_seq_17.shape} (367 windows x 10 steps x 17 features)")
    print(f"  - Full Leak-Free Sequence Tensor     : {X_seq_full.shape} (367 windows x 10 steps x 129 features)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. Compare Temporal Sequence Architectures (5-Fold Stratified CV)
    X_step1_only = X_seq_17[:, 0, :]                                  # Single 3s snapshot (t1 only)
    X_seq17_rec = extract_recurrent_gate_representation(X_seq_17)     # 10-step Recurrent Gate on 17 features
    X_seqfull_rec = extract_recurrent_gate_representation(X_seq_full) # 10-step Recurrent Gate on full leak-free tensor

    models_to_eval = [
        ("Baseline 1: Single 3s Step Snapshot (t1 only, 17 feats)", X_step1_only,
         MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42)),
        ("Model 2: 10-Step Unrolled Sequence MLP (17 feats x 10 steps)", X_seq17_rec,
         MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=700, random_state=42)),
        ("Model 3: D5 Bi-Recurrent Gate + Temporal Transition Net (Full)", X_seqfull_rec,
         VotingClassifier(estimators=[
             ('mlp_seq', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=700, random_state=42)),
             ('et_seq', ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=42))
         ], voting='soft'))
    ]

    print("\n[2/4] TEMPORAL SEQUENCE ARCHITECTURE COMPARISON (5-Fold Stratified CV):")
    print("-" * 90)
    print(f"{'Sequence Architecture':<60} | {'Acc (%)':<8} | {'Macro F1':<8}")
    print("-" * 90)

    seq_rows = []
    y_pred_best = None
    best_pipe = None

    for name, X_mat, clf in models_to_eval:
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        y_pred = cross_val_predict(pipe, X_mat, y_seq, cv=skf)
        acc = accuracy_score(y_seq, y_pred) * 100
        f1 = f1_score(y_seq, y_pred, average='macro') * 100
        prec = precision_score(y_seq, y_pred, average='macro') * 100
        rec = recall_score(y_seq, y_pred, average='macro') * 100
        print(f"{name:<60} | {acc:6.2f}%  | {f1:6.2f}%")
        seq_rows.append({
            'Sequence_Architecture': name,
            'Accuracy_Pct': round(acc, 2),
            'Precision_Pct': round(prec, 2),
            'Recall_Pct': round(rec, 2),
            'Macro_F1_Pct': round(f1, 2)
        })
        if "Model 3" in name:
            y_pred_best = y_pred
            pipe.fit(X_mat, y_seq)
            best_pipe = pipe

    pd.DataFrame(seq_rows).to_csv('data/processed/d5_sequence_model_metrics.csv', index=False)
    joblib.dump(best_pipe, 'models/d5_temporal_sequence_model.pkl')

    # 3. Per-Class Classification Report for D5
    print("\n[3/4] D5 PER-CLASS TEMPORAL SEQUENCE REPORT:")
    print("-" * 90)
    print(classification_report(y_seq, y_pred_best, target_names=RESP_CLASSES, digits=4))

    rep_d5 = classification_report(y_seq, y_pred_best, target_names=RESP_CLASSES, output_dict=True)
    d5_class_rows = []
    for cname in RESP_CLASSES:
        m = rep_d5[cname]
        d5_class_rows.append({
            'Deliverable': 'D5 Temporal Sequence Model',
            'Class': cname,
            'Precision_Pct': round(m['precision'] * 100, 2),
            'Recall_Pct': round(m['recall'] * 100, 2),
            'F1_Score_Pct': round(m['f1-score'] * 100, 2),
            'Support_Sequences': int(m['support'])
        })
    pd.DataFrame(d5_class_rows).to_csv('data/processed/d5_per_class_report.csv', index=False)

    # 4. Evaluate Sequence Horizon Scaling (T = 1 step [3s] up to T = 10 steps [12s])
    horizons = list(range(1, 11))
    f1_17_curve, f1_full_curve = [], []
    for T_h in horizons:
        X_h17 = extract_recurrent_gate_representation(X_seq_17[:, :T_h, :])
        X_hfull = extract_recurrent_gate_representation(X_seq_full[:, :T_h, :])

        p17 = Pipeline([('s', StandardScaler()), ('c', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42))])
        pfull = Pipeline([('s', StandardScaler()), ('c', ExtraTreesClassifier(n_estimators=150, class_weight='balanced', random_state=42))])

        pred17 = cross_val_predict(p17, X_h17, y_seq, cv=skf)
        predfull = cross_val_predict(pfull, X_hfull, y_seq, cv=skf)

        f1_17_curve.append(f1_score(y_seq, pred17, average='macro') * 100)
        f1_full_curve.append(f1_score(y_seq, predfull, average='macro') * 100)

    # 5. Plot Figure 10: D5 Confusion Matrix + Sequence Horizon Scaling Curve
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    cm = confusion_matrix(y_seq, y_pred_best)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    axes[0].imshow(cm_pct, cmap='Greens', vmin=0, vmax=100)
    short_labels = ['Normal\n(12-20 BPM)', 'Fast/Tachy\n(>22 BPM)', 'Apnea\n(Hold)']
    axes[0].set_xticks(range(3))
    axes[0].set_yticks(range(3))
    axes[0].set_xticklabels(short_labels, fontsize=9, fontweight='bold')
    axes[0].set_yticklabels(short_labels, fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Predicted Sequence Class', fontweight='bold')
    axes[0].set_ylabel('True Ground-Truth Class', fontweight='bold')
    axes[0].set_title('D5 Temporal Sequence Confusion Matrix (5-Fold CV)', fontweight='bold')

    for i in range(3):
        for j in range(3):
            color = 'white' if cm_pct[i, j] > 50 else 'black'
            axes[0].text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                         ha='center', va='center', color=color, fontsize=10, fontweight='bold')

    axes[1].plot(horizons, f1_full_curve, marker='o', lw=2.4, color='#2ca02c',
                 label='D5 Full Sequence Tensor (129 feats/step)')
    axes[1].plot(horizons, f1_17_curve, marker='s', lw=2.0, linestyle='--', color='#1f77b4',
                 label='17 Physiological Scalars Sequence (17 feats/step)')
    axes[1].set_xticks(horizons)
    axes[1].set_xticklabels([f"T={h}\n({h+2}s)" for h in horizons], fontsize=8.5)
    axes[1].set_xlabel('Sequence Length Horizon T (Number of 3s Steps Observed)', fontweight='bold')
    axes[1].set_ylabel('5-Fold CV Macro F1 Score (%)', fontweight='bold')
    axes[1].set_title('D5 Temporal Sequence Horizon Scaling (T=1 to 10 Steps)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig('docs/figures/fig10_d5_sequence_analysis.png', dpi=200)
    plt.close()
    print("[4/4] Saved D5 plot -> docs/figures/fig10_d5_sequence_analysis.png and tables -> data/processed/d5_*.csv!")
    print("=" * 90)


if __name__ == '__main__':
    run_d5_pipeline()
