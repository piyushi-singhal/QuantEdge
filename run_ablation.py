"""
QuantEdge — Ablation Study
===========================
5 configurations × 5 benchmark datasets × 5 metrics.
Measures marginal contribution of each module.
"""
import os, sys, csv, json, time, re, gc
import numpy as np
import pandas as pd
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'experiments', 'ablation')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, BASE)
from backend.ai_detector import SensitivityDetector
from quantum_tokenizer import QuantumTokenizer
from risk_scoring import RiskScorer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

SEED = 4201
np.random.seed(SEED)

with open(os.path.join(BASE, 'datasets', 'metadata', 'benchmark_metadata.json')) as f:
    META = json.load(f)

DATASETS = [
    ('HR_Benchmark_20K.csv', 2000),
    ('Healthcare_Benchmark_50K.csv', 2000),
    ('Banking_Benchmark_100K.csv', 2000),
    ('Adversarial_Benchmark_20K.csv', 2000),
    ('Scalability_Benchmark_1M.csv', 2000),
]

SENSITIVE_COLS = {}
ALL_COLS = {}
for name in [d[0] for d in DATASETS]:
    info = META['datasets'][name]
    SENSITIVE_COLS[name] = set(info['high_risk'])
    ALL_COLS[name] = [c for c in info['columns'] if c not in ('RiskScore', 'RiskCategory')]

def load_sample(name, n):
    path = os.path.join(BASE, 'datasets', 'benchmark', name)
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n: break
            rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════
# 5 CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════

PATTERNS = [
    r'^[\w\.\+\-]+@[\w\.\-]+\.[a-zA-Z]{2,}$',
    r'^\+?\d[\d\-\(\)\s]{7,20}\d$',
    r'^\d{3}-\d{2}-\d{4}$',
    r'^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$',
    r'^\$\d{1,3}(,\d{3})*(\.\d{2})?$',
    r'^\d+\s+\w+\s+(St|Ave|Blvd|Dr|Ln|Way|Ct|Rd|Cir|Pl)\b',
]

def _load_ml():
    d = SensitivityDetector()
    d.train(n_per_class=3000)
    return d

def _risk_adjust(fname, fval, prob):
    try:
        risk_info = RiskScorer.compute_risk(fname, str(fval), 'HR')
        risk_score = risk_info.get('score', 50)
    except:
        risk_score = 50
    thresh = 0.3 if risk_score >= 70 else 0.7 if risk_score < 40 else 0.5
    blended = min(1.0, prob * 0.6 + (risk_score / 100) * 0.4)
    return bool(prob >= thresh), round(blended, 4), risk_score

def process_regex(fields):
    t0 = time.perf_counter()
    results = []
    for fname, fval in fields:
        val = str(fval).strip()
        sens = any(re.match(p, val) for p in PATTERNS)
        results.append({'sensitive': sens, 'confidence': 0.9 if sens else 0.1})
    return results, time.perf_counter() - t0

def process_ml(ml, fields):
    texts = [str(v).strip() for _, v in fields]
    t0 = time.perf_counter()
    probs = ml.pipeline.predict_proba(texts)[:, 1]
    results = [{'sensitive': bool(p >= 0.5), 'confidence': round(float(p), 4)} for p in probs]
    return results, time.perf_counter() - t0

def process_ml_risk(ml, fields):
    texts = [str(v).strip() for _, v in fields]
    t0 = time.perf_counter()
    probs = ml.pipeline.predict_proba(texts)[:, 1]
    results = [_risk_adjust(f, v, p)[:2] for (f, v), p in zip(fields, probs)]
    results = [{'sensitive': r[0], 'confidence': r[1]} for r in results]
    return results, time.perf_counter() - t0

def process_ml_risk_token(ml, tokenizer, fields):
    texts = [str(v).strip() for _, v in fields]
    t0 = time.perf_counter()
    probs = ml.pipeline.predict_proba(texts)[:, 1]
    results = []
    for (fname, fval), prob in zip(fields, probs):
        sens, conf, _ = _risk_adjust(fname, fval, prob)
        if sens:
            ta, tb = tokenizer.generate_token_pair()
        results.append({'sensitive': sens, 'confidence': conf})
    return results, time.perf_counter() - t0

def process_full(ml, tokenizer, fields):
    from backend.blockchain_manager import BlockchainManager
    texts = [str(v).strip() for _, v in fields]
    t0 = time.perf_counter()
    probs = ml.pipeline.predict_proba(texts)[:, 1]
    results = []
    for (fname, fval), prob in zip(fields, probs):
        sens, conf, risk_score = _risk_adjust(fname, fval, prob)
        if sens:
            ta, tb = tokenizer.generate_token_pair()
        results.append({'sensitive': sens, 'confidence': conf})
    base_time = time.perf_counter() - t0
    # Add blockchain overhead separately: measure 100 stores, compute avg, add for each sensitive
    bc = BlockchainManager()
    bc_times = []
    for i in range(100):
        ta, tb = tokenizer.generate_token_pair()
        t1 = time.perf_counter()
        bc.store_token(ta, 'Field', 50, 30, 'HR', 'SALES')
        bc_times.append(time.perf_counter() - t1)
    avg_bc = np.mean(bc_times)
    n_sens = sum(1 for r in results if r['sensitive'])
    total_time = base_time + n_sens * avg_bc
    return results, total_time


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def run():
    print("="*70)
    print("  QUANTEDGE — ABLATION STUDY")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Output:  {OUT}/")
    print("="*70)

    # Pre-load data and build field lists per dataset
    ds_data = {}
    for ds_name, sample_n in DATASETS:
        rows = load_sample(ds_name, sample_n)
        sens = SENSITIVE_COLS[ds_name]
        cols = ALL_COLS[ds_name]
        fields = []
        truths = []
        for row in rows:
            for col in cols:
                val = row.get(col, '')
                if not val or str(val).strip() == '':
                    continue
                fields.append((col, val))
                truths.append(1 if col in sens else 0)
        ds_data[ds_name] = {'fields': fields, 'truths': truths}

    # Initialize modules
    print("\n◆ Initializing modules...")
    ml = _load_ml()
    tokenizer = QuantumTokenizer()
    print("   ML + Tokenizer ready\n")

    configs = [
        ('Regex Baseline',         lambda f: process_regex(f)),
        ('ML Detector Only',       lambda f: process_ml(ml, f)),
        ('ML + Risk Scoring',      lambda f: process_ml_risk(ml, f)),
        ('ML + Risk + Token',      lambda f: process_ml_risk_token(ml, tokenizer, f)),
        ('Full QuantEdge',         lambda f: process_full(ml, tokenizer, f)),
    ]

    ablation_rows = []
    detail_rows = []

    for config_name, process_fn in configs:
        print(f"◆ {config_name}")
        all_y_true = []
        all_y_pred = []
        total_lat = 0.0
        total_n = 0

        for ds_name, _ in DATASETS:
            d = ds_data[ds_name]
            fields = d['fields']
            truths = d['truths']
            results, lat = process_fn(fields)
            preds = [1 if r['sensitive'] else 0 for r in results]
            all_y_true.extend(truths)
            all_y_pred.extend(preds)
            total_lat += lat
            total_n += len(fields)

            ds_label = ds_name.replace('_Benchmark_', ' ').replace('.csv','')
            detail_rows.append({
                'Configuration': config_name,
                'Dataset': ds_label,
                'Precision': round(precision_score(truths, preds, zero_division=0), 4),
                'Recall': round(recall_score(truths, preds, zero_division=0), 4),
                'F1': round(f1_score(truths, preds, zero_division=0), 4),
                'Accuracy': round(accuracy_score(truths, preds), 4),
                'Latency_us': round(lat / max(len(fields), 1) * 1_000_000, 2),
            })

        prec = precision_score(all_y_true, all_y_pred, zero_division=0)
        rec = recall_score(all_y_true, all_y_pred, zero_division=0)
        f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
        acc = accuracy_score(all_y_true, all_y_pred)
        avg_lat = total_lat / total_n * 1_000_000 if total_n > 0 else 0

        ablation_rows.append({
            'Configuration': config_name,
            'FieldsProcessed': total_n,
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'Accuracy': round(acc, 4),
            'TotalLatency_s': round(total_lat, 4),
            'AvgLatency_us': round(avg_lat, 2),
        })
        print(f"   {total_n:,} fields  F1={f1:.4f}  Acc={acc:.4f}  Lat={avg_lat:.2f}µs")

    # Save CSVs
    df_a = pd.DataFrame(ablation_rows)
    df_a.to_csv(os.path.join(OUT, 'table_ablation_study.csv'), index=False)
    df_d = pd.DataFrame(detail_rows)
    df_d.to_csv(os.path.join(OUT, 'table_ablation_detail.csv'), index=False)
    print(f"\n  CSVs saved")

    # ── Contribution analysis ──
    lines = []
    lines.append("="*70)
    lines.append("ABLATION STUDY — MODULE CONTRIBUTION ANALYSIS")
    lines.append("="*70)
    lines.append("")
    lines.append(f"{'Module Added':30s} {'ΔPrec':8s} {'ΔRecall':8s} {'ΔF1':8s} {'ΔAcc':8s} {'ΔLat(µs)':10s}")
    lines.append("-"*70)
    prev = None
    contribs = []
    for _, r in df_a.iterrows():
        parts = r['Configuration'].split(' + ')
        module = parts[-1] if len(parts) > 1 else r['Configuration']
        if prev is not None:
            dp = r['Precision'] - prev['Precision']
            dr = r['Recall'] - prev['Recall']
            df1 = r['F1'] - prev['F1']
            da = r['Accuracy'] - prev['Accuracy']
            dl = r['AvgLatency_us'] - prev['AvgLatency_us']
            lines.append(f"{module:30s} {dp:+.4f}  {dr:+.4f}  {df1:+.4f}  {da:+.4f}  {dl:+8.2f}µs")
            contribs.append({'Module': module, 'ΔF1': round(df1, 4), 'ΔAccuracy': round(da, 4),
                             'ΔLatency_us': round(dl, 2)})
        prev = r
    lines.append("")
    lines.append("="*70)
    lines.append("MODULE ROLES:")
    lines.append("="*70)
    lines.append("")
    lines.append("1. Regex Baseline:     Fast pattern matching — limited to known formats")
    lines.append("2. ML Detector:        TF-IDF(ngram 2-5) + LogisticRegression — generalizes")
    lines.append("3. Risk Scoring:       Field heuristics + entropy. Adapts thresholds dynamically.")
    lines.append("4. Tokenization:       SHA3-256 + Fernet. Replaces sensitive values.")
    lines.append("5. Blockchain:         py-evm EVM. Immutable token hashes on-chain.")
    lines.append("")
    lines.append("="*70)
    lines.append("CUMULATIVE SYSTEM VIEW")
    lines.append("="*70)
    lines.append("")
    lines.append(f"{'Stage':30s} {'F1':8s} {'Latency':10s}")
    lines.append("-"*70)
    for _, r in df_a.iterrows():
        lines.append(f"{r['Configuration']:30s} {r['F1']:.4f}  {r['AvgLatency_us']:>8.2f}µs")
    lines.append("")

    with open(os.path.join(OUT, 'contribution_analysis.txt'), 'w') as f:
        f.write('\n'.join(lines))
    pd.DataFrame(contribs).to_csv(os.path.join(OUT, 'table_contributions.csv'), index=False)
    print("  Contribution analysis saved")

    # ── IEEE table ──
    lines2 = []
    lines2.append("="*80)
    lines2.append("TABLE: ABLATION STUDY — QUANTEDGE MODULE CONTRIBUTION")
    lines2.append("="*80)
    lines2.append(f"{'Configuration':30s} {'Prec':8s} {'Recall':8s} {'F1':8s} {'Acc':8s} {'Lat(µs)':10s}")
    lines2.append("-"*80)
    for _, r in df_a.iterrows():
        lines2.append(f"{r['Configuration']:30s} {r['Precision']:.4f}  {r['Recall']:.4f}  "
                      f"{r['F1']:.4f}  {r['Accuracy']:.4f}  {r['AvgLatency_us']:>8.2f}")
    lines2.append("")
    for c in contribs:
        lines2.append(f"  {c['Module']:25s}: ΔF1={c['ΔF1']:+.4f}  ΔAcc={c['ΔAccuracy']:+.4f}  ΔLat={c['ΔLatency_us']:+.1f}µs")
    with open(os.path.join(OUT, 'ieee_ablation_table.txt'), 'w') as f:
        f.write('\n'.join(lines2))

    print("  IEEE table saved")
    print('\n' + '\n'.join(lines))

    # ── Graphs ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
        FD = os.path.join(BASE, 'experiments', 'figures')
        os.makedirs(FD, exist_ok=True)
        FS, COLORS = (6.0, 4.0), ['#999999','#2166ac','#4daf4a','#ff7f00','#d6604d']
        configs_l = df_a['Configuration'].tolist()
        x = list(range(len(configs_l)))
        wb = 0.7

        # F1 + Accuracy
        fig, ax = plt.subplots(figsize=FS)
        ax.bar(x, df_a['F1'], wb, label='F1', color='#2166ac', edgecolor='white', alpha=0.9)
        ax.bar(x, df_a['Accuracy'], wb*0.6, label='Accuracy', color='#4daf4a', edgecolor='white', alpha=0.8)
        for _, r in df_a.iterrows():
            ax.text(list(df_a.index).index(_), r['F1']+0.015, f"{r['F1']:.3f}", ha='center', fontsize=8, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels([c[:12] for c in configs_l], rotation=20, fontsize=8)
        ax.set_ylabel('Score'); ax.set_title('Ablation: Detection Performance'); ax.set_ylim([0,1.1])
        ax.legend(fontsize=8, loc='lower right'); ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(FD, 'fig_ablation_performance.png')); plt.close(fig)

        # Precision + Recall
        fig, ax = plt.subplots(figsize=FS)
        ax.bar(x, df_a['Precision'], wb, label='Precision', color='#2166ac', edgecolor='white', alpha=0.7)
        ax.bar(x, df_a['Recall'], wb*0.6, label='Recall', color='#d6604d', edgecolor='white', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels([c[:12] for c in configs_l], rotation=20, fontsize=8)
        ax.set_ylabel('Score'); ax.set_title('Ablation: Precision vs Recall'); ax.set_ylim([0,1.1])
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(FD, 'fig_ablation_prec_recall.png')); plt.close(fig)

        # Latency
        fig, ax = plt.subplots(figsize=FS)
        bars = ax.bar(x, df_a['AvgLatency_us'], wb, color=[COLORS[i] for i in range(5)], edgecolor='white')
        max_lat = df_a['AvgLatency_us'].max()
        for bar, val in zip(bars, df_a['AvgLatency_us']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max_lat*0.02,
                    f'{val:.1f}µs', ha='center', fontsize=8, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels([c[:12] for c in configs_l], rotation=20, fontsize=8)
        ax.set_ylabel('Avg Latency (µs)'); ax.set_title('Ablation: Latency Breakdown')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(FD, 'fig_ablation_latency.png')); plt.close(fig)

        # Per-dataset F1
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        datasets = df_d['Dataset'].unique()
        configs_d = df_d['Configuration'].unique()
        xd = np.arange(len(datasets)); wd = 0.8 / len(configs_d)
        for i, conf in enumerate(configs_d):
            vals = [df_d[(df_d['Dataset']==d)&(df_d['Configuration']==conf)]['F1'].values[0] for d in datasets]
            ax.bar(xd+i*wd-0.4+wd/2, vals, wd, label=conf[:12], color=COLORS[i], edgecolor='white')
        ax.set_xticks(xd); ax.set_xticklabels([d[:15] for d in datasets], rotation=20, fontsize=7)
        ax.set_ylabel('F1 Score'); ax.set_title('Ablation: Per-Dataset F1'); ax.set_ylim([0,1.1])
        ax.legend(fontsize=7, loc='lower right', ncol=2); ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(FD, 'fig_ablation_per_dataset.png')); plt.close(fig)

        # Contribution bars
        if contribs:
            fig, ax = plt.subplots(figsize=FS)
            mods = [c['Module'] for c in contribs]; xc = np.arange(len(mods)); wc = 0.25
            ax.bar(xc-wc, [c['ΔF1'] for c in contribs], wc, label='ΔF1', color='#2166ac', edgecolor='white')
            ax.bar(xc, [c['ΔAccuracy'] for c in contribs], wc, label='ΔAccuracy', color='#4daf4a', edgecolor='white')
            max_lat_d = max(abs(c['ΔLatency_us']) for c in contribs) or 1
            lat_norm = [c['ΔLatency_us']/max_lat_d*0.1 for c in contribs]
            ax.bar(xc+wc, lat_norm, wc, label='ΔLat (norm)', color='#d6604d', edgecolor='white')
            ax.axhline(y=0, color='gray', lw=0.5)
            ax.set_xticks(xc); ax.set_xticklabels(mods, fontsize=8, rotation=15)
            ax.set_ylabel('Marginal Change'); ax.set_title('Module Contribution')
            ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout(); fig.savefig(os.path.join(FD, 'fig_ablation_contributions.png')); plt.close(fig)

        print("  All figures saved")
    except Exception as e:
        print(f"  [WARN] Figure generation: {e}")

    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY COMPLETE")
    print(f"  Output: {OUT}/")
    print(f"{'='*70}")

if __name__ == '__main__':
    run()
