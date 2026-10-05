import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

os.makedirs('models', exist_ok=True)
os.makedirs('docs/figures', exist_ok=True)

TASKS = {
    'presence': {
        'title': 'Task 1: Human Presence & Activity Detection',
        'file': 'data/processed/features/presence_dataset.npz',
        'classes': ['Empty Room', 'Stationary', 'Moving']
    },
    'rf_fingerprinting': {
        'title': 'Task 2: Spatial RF Fingerprinting (Zone Localization)',
        'file': 'data/processed/features/rf_fingerprinting_dataset.npz',
        'classes': ['Zone 1 (1m)', 'Zone 2 (2m)', 'Zone 3 (3m)']
    },
    'respiratory': {
        'title': 'Task 3: Respiratory Anomaly Classification',
        'file': 'data/processed/features/respiratory_dataset.npz',
        'classes': ['Normal', 'Fast (Tachypnea)', 'Apnea (Hold)']
    }
}


def get_candidate_models():
    return {
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
        ]),
        'Extra Trees': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=42))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True, random_state=42))
        ]),
        'MLP Neural Net': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42))
        ])
    }


def evaluate_all_tasks():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_rows = []
    best_cms = {}

    print("=" * 95)
    print(f"{'Task':<20} | {'Model':<16} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'Macro F1':<10}")
    print("=" * 95)

    for task_key, task_info in TASKS.items():
        data = np.load(task_info['file'], allow_pickle=True)
        X, y = data['X'], data['y']
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        candidates = get_candidate_models()
        best_f1 = -1.0
        best_model_name = None
        best_preds = None
        best_pipeline = None

        for model_name, pipeline in candidates.items():
            y_pred = cross_val_predict(pipeline, X, y, cv=skf)
            acc = accuracy_score(y, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='macro', zero_division=0)

            print(f"{task_key:<20} | {model_name:<16} | {acc*100:>8.2f}% | {prec*100:>8.2f}% | {rec*100:>8.2f}% | {f1*100:>8.2f}%")

            metrics_rows.append({
                'task': task_key,
                'model': model_name,
                'accuracy_pct': round(acc * 100, 2),
                'macro_precision_pct': round(prec * 100, 2),
                'macro_recall_pct': round(rec * 100, 2),
                'macro_f1_pct': round(f1 * 100, 2)
            })

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = model_name
                best_preds = y_pred
                best_pipeline = pipeline

        print("-" * 95)

        # Fit best model on full dataset and save artifact for dashboard/inference
        best_pipeline.fit(X, y)
        joblib.dump({
            'model': best_pipeline,
            'model_name': best_model_name,
            'classes': task_info['classes'],
            'cv_f1_pct': round(best_f1 * 100, 2)
        }, f"models/{task_key}_best_model.pkl")

        best_cms[task_key] = {
            'cm': confusion_matrix(y, best_preds),
            'classes': task_info['classes'],
            'model_name': best_model_name,
            'acc': accuracy_score(y, best_preds) * 100,
            'f1': best_f1 * 100,
            'title': task_info['title']
        }

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv('data/processed/model_metrics.csv', index=False)
    plot_confusion_matrices(best_cms)
    plot_model_benchmark(metrics_df)
    evaluate_bpm_accuracy()


def evaluate_bpm_accuracy():
    """Evaluate FFT respiratory rate (BPM) estimation accuracy across normal & fast breathing."""
    data = np.load('data/processed/features/respiratory_dataset.npz', allow_pickle=True)
    X, y = data['X'], data['y']
    est_bpm = X[:, 9]  # Feature index 9 is est_bpm from padded FFT

    norm_bpm = est_bpm[y == 0]
    fast_bpm = est_bpm[y == 1]

    print("\nRESPIRATORY RATE (BPM) ESTIMATION SUMMARY:")
    print(f"  - Normal Breathing Windows (n={len(norm_bpm)}) : Mean = {np.mean(norm_bpm):.1f} ± {np.std(norm_bpm):.1f} BPM (Expected Physiological Range: 12–20 BPM)")
    print(f"  - Fast / Tachypnea Windows (n={len(fast_bpm)}) : Mean = {np.mean(fast_bpm):.1f} ± {np.std(fast_bpm):.1f} BPM (Expected Physiological Range: >22 BPM)")


def plot_confusion_matrices(best_cms):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for ax, (task_key, info) in zip(axes, best_cms.items()):
        cm = info['cm']
        classes = info['classes']
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{info['title']}\nBest: {info['model_name']} (Acc: {info['acc']:.1f}%, F1: {info['f1']:.1f}%)", fontweight='bold', fontsize=10)

        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=20, ha='right')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        ax.grid(False)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight='bold', fontsize=12)

        ax.set_ylabel('True Class')
        ax.set_xlabel('Predicted Class')

    plt.tight_layout()
    plt.savefig('docs/figures/fig5_confusion_matrices.png', dpi=200)
    plt.close()


def plot_model_benchmark(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    tasks = ['presence', 'rf_fingerprinting', 'respiratory']
    task_labels = ['Presence & Motion', 'Spatial RF Zones (1m/2m/3m)', 'Respiratory Anomaly']
    models = df['model'].unique()

    x = np.arange(len(tasks))
    width = 0.2
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

    for i, (model, color) in enumerate(zip(models, colors)):
        f1_scores = [df[(df['task'] == t) & (df['model'] == model)]['macro_f1_pct'].values[0] for t in tasks]
        rects = ax.bar(x + (i - 1.5) * width, f1_scores, width, label=model, color=color)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('5-Fold Stratified CV Macro F1-Score (%)')
    ax.set_title('Pulse-Fi Machine Learning Classifier Benchmark Across All Three CSI Tasks', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('docs/figures/fig6_model_benchmark.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    evaluate_all_tasks()
    print("\nSaved trained models to models/*.pkl, metrics to data/processed/model_metrics.csv, and plots to docs/figures/fig5 & fig6!")
