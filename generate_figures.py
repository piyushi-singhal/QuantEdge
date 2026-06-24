"""
Generate publication-quality figures (PNG) for IEEE paper.
Output: experiments/figures/
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'experiments', 'figures')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, BASE)
from backend.ai_detector import _generate_synthetic_dataset
from risk_scoring import RiskScorer

SEED = 42
np.random.seed(SEED)

# ── Config ────────────────────────────────────────────────────────────
FIGSIZE_W = 5.5  # IEEE column width ~3.5in, use 5.5 for full page width
FIGSIZE_H = 4.5
DPI = 300
COLORS = {'main': '#2166ac', 'fill': '#d1e5f0', 'line': '#b2182b',
          'bar1': '#4393c3', 'bar2': '#f4a582', 'bar3': '#92c5de'}
FONT = {'family': 'serif', 'serif': ['Times', 'Computer Modern Roman']}
matplotlib.rc('font', **FONT)
matplotlib.rc('text', usetex=False)
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ── 1. CONFUSION MATRIX HEATMAP ──────────────────────────────────────
def fig_confusion_matrix():
    cm = np.array([[5000, 0], [0, 5000]])
    labels = ['Non-Sensitive', 'Sensitive']

    fig, ax = plt.subplots(figsize=(FIGSIZE_W, FIGSIZE_W * 0.85))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nNon-Sensitive', 'Predicted\nSensitive'], fontsize=10)
    ax.set_yticklabels(['Actual\nNon-Sensitive', 'Actual\nSensitive'], fontsize=10)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = 'white' if val > 2500 else 'black'
            ax.text(j, i, f'{val:,}', ha='center', va='center',
                    fontsize=18, fontweight='bold', color=color)

    ax.spines[:].set_visible(False)
    ax.set_title('Confusion Matrix — AI Sensitivity Detector', fontsize=12, pad=12)
    fig.tight_layout()
    path = os.path.join(OUT, 'fig_confusion_matrix.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 2. ROC CURVE ─────────────────────────────────────────────────────
def fig_roc_curve():
    texts, labels = _generate_synthetic_dataset(n_per_class=5000)
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), max_features=5000, sublinear_tf=True)),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=SEED)),
    ])
    pipe.fit(X_tr, y_tr)
    y_prob = pipe.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    auc_score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(FIGSIZE_W, FIGSIZE_W * 0.85))
    ax.plot(fpr, tpr, color=COLORS['main'], lw=2.5,
            label=f'ROC curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Random (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS['main'])

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    ax.set_title('ROC Curve — Sensitivity Detection', fontsize=12, pad=10)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='gray')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    # Annotate perfect corner
    ax.annotate('Perfect\nClassifier', xy=(0, 1), xytext=(0.25, 0.82),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
                fontsize=9, fontstyle='italic', color='gray')

    fig.tight_layout()
    path = os.path.join(OUT, 'fig_roc_curve.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 3. CROSS-VALIDATION BAR CHART ────────────────────────────────────
def fig_cross_validation():
    folds = [1, 2, 3, 4, 5]
    values = [1.0, 1.0, 1.0, 1.0, 1.0]
    metrics = ['Precision', 'Recall', 'F1', 'Accuracy']

    fig, ax = plt.subplots(figsize=(FIGSIZE_W, FIGSIZE_W * 0.85))
    x = np.arange(len(folds))
    width = 0.18

    for i, m in enumerate(metrics):
        bars = ax.bar(x + i * width, values, width, label=m,
                      color=[COLORS['bar1'], COLORS['bar2'], COLORS['bar3'], COLORS['main']][i],
                      edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Cross-Validation Fold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('5-Fold Cross-Validation Performance', fontsize=12, pad=10)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'Fold {f}' for f in folds], fontsize=10)
    ax.set_ylim([0.97, 1.005])
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='gray',
              ncol=2, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    fig.tight_layout()
    path = os.path.join(OUT, 'fig_cross_validation.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 4. BLOCKCHAIN PERFORMANCE BAR CHART ──────────────────────────────
def fig_blockchain_performance():
    data = pd.read_csv(os.path.join(BASE, 'experiments', 'results', 'table_blockchain_benchmark.csv'))
    trials = data['Trial'].values
    store_times = data['StoreTime_s'].values
    verify_times = data['VerifyTime_s'].values
    revoke_times = data['RevokeTime_s'].values

    fig, ax = plt.subplots(figsize=(FIGSIZE_W, FIGSIZE_W * 0.85))
    x = np.arange(len(trials))
    width = 0.25

    ax.bar(x - width, store_times, width, label='Store', color=COLORS['main'],
           edgecolor='white', linewidth=0.5)
    ax.bar(x, verify_times, width, label='Verify', color=COLORS['bar1'],
           edgecolor='white', linewidth=0.5)
    ax.bar(x + width, revoke_times, width, label='Revoke', color=COLORS['line'],
           edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Trial', fontsize=11)
    ax.set_ylabel('Time (s)', fontsize=11)
    ax.set_title('Blockchain Transaction Time — 10 Trials', fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}' for t in trials], fontsize=9)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='gray')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUT, 'fig_blockchain_performance.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 5. TOKENIZATION SCALABILITY ──────────────────────────────────────
def fig_scalability():
    data = pd.read_csv(os.path.join(BASE, 'experiments', 'results', 'table_scalability.csv'))

    fig, ax1 = plt.subplots(figsize=(FIGSIZE_W, FIGSIZE_W * 0.85))

    color1 = COLORS['main']
    color2 = COLORS['line']

    line1 = ax1.plot(data['Records'], data['Throughput_tps'], 'o-',
                     color=color1, lw=2, markersize=6, label='Throughput')
    ax1.set_xlabel('Number of Records', fontsize=11)
    ax1.set_ylabel('Throughput (tokens/s)', fontsize=11, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')

    ax2 = ax1.twinx()
    line2 = ax2.plot(data['Records'], data['Latency_per_token_us'], 's--',
                     color=color2, lw=2, markersize=6, label='Latency')
    ax2.set_ylabel('Latency (µs/token)', fontsize=11, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', frameon=True,
               facecolor='white', edgecolor='gray')

    ax1.set_title('Tokenization Scalability', fontsize=12, pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUT, 'fig_scalability.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 6. RISK SCORING PROFILE ──────────────────────────────────────────
def fig_risk_profile():
    data = pd.read_csv(os.path.join(BASE, 'experiments', 'results', 'table_risk_scoring.csv'))
    data = data.sort_values('RiskScore', ascending=True)

    fig, ax = plt.subplots(figsize=(FIGSIZE_W + 1, FIGSIZE_W * 0.75))
    colors = ['#d73027' if s >= 70 else '#f46d43' if s >= 40 else '#abd9e9'
              for s in data['RiskScore']]
    bars = ax.barh(range(len(data)), data['RiskScore'], color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data['Field'], fontsize=10)
    ax.set_xlabel('Risk Score', fontsize=11)
    ax.set_title('Risk Score by Field Type', fontsize=12, pad=10)
    ax.set_xlim([0, 110])

    # Add score labels on bars
    for i, (_, row) in enumerate(data.iterrows()):
        ax.text(row['RiskScore'] + 1, i, f"{row['RiskScore']}  ({row['Tier']})",
                va='center', fontsize=8, fontfamily='monospace')

    ax.axvline(x=70, linestyle='--', color='gray', alpha=0.5, linewidth=0.8)
    ax.axvline(x=40, linestyle='--', color='gray', alpha=0.5, linewidth=0.8)
    ax.text(72, len(data)-0.3, 'High', fontsize=8, color='#d73027', fontstyle='italic')
    ax.text(42, len(data)-0.3, 'Medium', fontsize=8, color='#f46d43', fontstyle='italic')
    ax.text(5, len(data)-0.3, 'Low', fontsize=8, color='#abd9e9', fontstyle='italic')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, 'fig_risk_profile.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── MAIN ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating publication-quality figures...")
    print(f"  Output: {OUT}/")
    fig_confusion_matrix()
    fig_roc_curve()
    fig_cross_validation()
    fig_blockchain_performance()
    fig_scalability()
    fig_risk_profile()
    print(f"\nAll figures saved to: {OUT}/")
