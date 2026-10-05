"""
Deliverable D8: Random Forest Respiratory & Presence Classification Pipeline
Author: Benert (Pulse-Fi Team)
Description:
  - Loads the leak-free named respiratory dataset (367 x 129) and presence dataset (312 x 128).
  - Performs a 3-stage Feature Ablation Study (17 Physiological Scalars vs 112 AC Profile vs 129 Combined).
  - Runs Stratified 5-Fold Cross-Validation + Out-of-Bag (OOB) error analysis + GridSearchCV.
  - Exports per-class Precision/Recall/F1 tables, dedicated D8 models, and Figure 8 diagnostic plots.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
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

RESP_CLASSES = ['Normal (12-20 BPM)', 'Fast / Tachypnea (>22 BPM)', 'Apnea (Hold)']
PRES_CLASSES = ['Empty Room', 'Stationary Person', 'Active Movement']


def run_d8_pipeline():
    print("=" * 88)
    print("DELIVERABLE D8: RANDOM FOREST CLASSIFIER & ABLATION STUDY (BENERT)")
    print("=" * 88)

    # 1. Load Named Respiratory Dataset (Leak-free: sc_spatial_ratio & sc_spatial_std excluded)
    df_resp = pd.read_csv('data/processed/features/respiratory_features_named.csv')
    feature_cols = [c for c in df_resp.columns if c not in ('class_id', 'class_label', 'source_file')]
    X_resp = df_resp[feature_cols].values
    y_resp = df_resp['class_id'].values

    # Split feature groups for ablation study
    X_phys_17 = X_resp[:, :17]    # 17 physiologically grounded scalar features
    X_ac_112 = X_resp[:, 17:]     # 112 zero-mean L2-normalized AC dynamic subcarrier features
    X_full_129 = X_resp           # Full 129 leak-free feature set

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. Feature Ablation Study
    ablation_configs = [
        ("Config A: 17 Physiological Scalars Only", X_phys_17),
        ("Config B: 112 Zero-Mean AC Dynamic Profile", X_ac_112),
        ("Config C: Combined 129 Leak-Free Features (D8 Final)", X_full_129),
    ]

    ablation_rows = []
    print("\n[1/4] FEATURE SET ABLATION STUDY (5-Fold Stratified CV + OOB Score):")
    print("-" * 88)
    print(f"{'Feature Configuration':<52} | {'Acc (%)':<8} | {'Macro F1':<8} | {'OOB Score'}")
    print("-" * 88)

    for label, X_sub in ablation_configs:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)
        rf = RandomForestClassifier(
            n_estimators=200,
            max_features='sqrt',
            class_weight='balanced',
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        y_pred = cross_val_predict(rf, X_scaled, y_resp, cv=skf)
        rf.fit(X_scaled, y_resp)

        acc = accuracy_score(y_resp, y_pred) * 100
        f1 = f1_score(y_resp, y_pred, average='macro') * 100
        oob = rf.oob_score_ * 100
        print(f"{label:<52} | {acc:6.2f}%  | {f1:6.2f}%  | {oob:6.2f}%")
        ablation_rows.append({
            'Configuration': label,
            'Num_Features': X_sub.shape[1],
            'CV_Accuracy_Pct': round(acc, 2),
            'CV_Macro_F1_Pct': round(f1, 2),
            'OOB_Score_Pct': round(oob, 2)
        })

    pd.DataFrame(ablation_rows).to_csv('data/processed/d8_ablation_study.csv', index=False)

    # 3. Hyperparameter Tuning via GridSearchCV on Full Leak-Free Set
    print("\n[2/4] HYPERPARAMETER GRID SEARCH (Random Forest):")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(class_weight='balanced', oob_score=True, random_state=42, n_jobs=-1))
    ])
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 10, 16],
        'rf__min_samples_split': [2, 4]
    }
    grid = GridSearchCV(pipe, param_grid, cv=skf, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_full_129, y_resp)
    print(f"  Best Parameters : {grid.best_params_}")
    print(f"  Best 5-Fold F1  : {grid.best_score_ * 100:.2f}%")

    # 4. Detailed Per-Class Report for Respiratory & Presence Tasks
    best_rf_resp = grid.best_estimator_
    y_pred_resp = cross_val_predict(best_rf_resp, X_full_129, y_resp, cv=skf)

    print("\n[3/4] D8 PER-CLASS CLASSIFICATION REPORT (RESPIRATORY ANOMALY):")
    print("-" * 88)
    print(classification_report(y_resp, y_pred_resp, target_names=RESP_CLASSES, digits=4))

    rep_dict = classification_report(y_resp, y_pred_resp, target_names=RESP_CLASSES, output_dict=True)
    per_class_rows = []
    for cls_name in RESP_CLASSES:
        m = rep_dict[cls_name]
        per_class_rows.append({
            'Task': 'Respiratory Anomaly (D8)',
            'Class': cls_name,
            'Precision_Pct': round(m['precision'] * 100, 2),
            'Recall_Pct': round(m['recall'] * 100, 2),
            'F1_Score_Pct': round(m['f1-score'] * 100, 2),
            'Support_Windows': int(m['support'])
        })

    # Also evaluate Presence & Motion Random Forest for D8 completeness
    pres_data = np.load('data/processed/features/presence_dataset.npz')
    X_pres, y_pres = pres_data['X'], pres_data['y']
    rf_pres_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', oob_score=True, random_state=42))
    ])
    y_pred_pres = cross_val_predict(rf_pres_pipe, X_pres, y_pres, cv=skf)
    rf_pres_pipe.fit(X_pres, y_pres)
    rep_pres = classification_report(y_pres, y_pred_pres, target_names=PRES_CLASSES, output_dict=True)
    for cls_name in PRES_CLASSES:
        m = rep_pres[cls_name]
        per_class_rows.append({
            'Task': 'Presence & Activity (D8)',
            'Class': cls_name,
            'Precision_Pct': round(m['precision'] * 100, 2),
            'Recall_Pct': round(m['recall'] * 100, 2),
            'F1_Score_Pct': round(m['f1-score'] * 100, 2),
            'Support_Windows': int(m['support'])
        })

    df_per_class = pd.DataFrame(per_class_rows)
    df_per_class.to_csv('data/processed/d8_per_class_report.csv', index=False)

    # Save dedicated D8 Random Forest model artifacts
    joblib.dump(best_rf_resp, 'models/d8_random_forest_respiratory.pkl')
    joblib.dump(rf_pres_pipe, 'models/d8_random_forest_presence.pkl')

    # 5. Generate Figure 8: D8 Random Forest Diagnostic Suite (Confusion Matrix + OOB Convergence)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # Left panel: Annotated Confusion Matrix with counts + percentages
    cm = confusion_matrix(y_resp, y_pred_resp)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    im = axes[0].imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    short_labels = ['Normal\n(12-20 BPM)', 'Fast/Tachy\n(>22 BPM)', 'Apnea\n(Hold)']
    axes[0].set_xticks(range(3))
    axes[0].set_yticks(range(3))
    axes[0].set_xticklabels(short_labels, fontsize=9, fontweight='bold')
    axes[0].set_yticklabels(short_labels, fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Predicted Class', fontweight='bold')
    axes[0].set_ylabel('True Ground-Truth Class', fontweight='bold')
    axes[0].set_title('D8 Random Forest Confusion Matrix (5-Fold CV)', fontweight='bold')

    for i in range(3):
        for j in range(3):
            color = 'white' if cm_pct[i, j] > 50 else 'black'
            axes[0].text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                         ha='center', va='center', color=color, fontsize=10, fontweight='bold')

    # Right panel: OOB Accuracy vs Number of Trees (15 to 300 trees)
    tree_counts = [15, 25, 40, 60, 80, 100, 140, 180, 220, 260, 300]
    oob_scores_full, oob_scores_phys = [], []
    X_full_s = StandardScaler().fit_transform(X_full_129)
    X_phys_s = StandardScaler().fit_transform(X_phys_17)

    for n_t in tree_counts:
        rf_f = RandomForestClassifier(n_estimators=n_t, class_weight='balanced', oob_score=True, random_state=42, n_jobs=-1)
        rf_f.fit(X_full_s, y_resp)
        oob_scores_full.append(rf_f.oob_score_ * 100)

        rf_p = RandomForestClassifier(n_estimators=n_t, class_weight='balanced', oob_score=True, random_state=42, n_jobs=-1)
        rf_p.fit(X_phys_s, y_resp)
        oob_scores_phys.append(rf_p.oob_score_ * 100)

    axes[1].plot(tree_counts, oob_scores_full, marker='o', lw=2.2, color='#1f77b4',
                 label='Config C: Full Leak-Free Set (129 feats)')
    axes[1].plot(tree_counts, oob_scores_phys, marker='s', lw=2.0, linestyle='--', color='#ff7f0e',
                 label='Config A: 17 Physiological Scalars Only')
    axes[1].set_xlabel('Number of Decision Trees (n_estimators)', fontweight='bold')
    axes[1].set_ylabel('Out-of-Bag (OOB) Generalization Accuracy (%)', fontweight='bold')
    axes[1].set_title('Random Forest OOB Convergence & Feature Ablation', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig('docs/figures/fig8_d8_rf_analysis.png', dpi=200)
    plt.close()
    print("[4/4] Saved D8 plot -> docs/figures/fig8_d8_rf_analysis.png and tables -> data/processed/d8_*.csv!")
    print("=" * 88)


if __name__ == '__main__':
    run_d8_pipeline()
