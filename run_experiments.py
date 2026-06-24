"""
QuantEdge — Publication-Grade Experiment Runner
================================================
Comprehensive evaluation across 5 benchmark datasets with
3 method baselines (Regex, ML, Full QuantEdge).

Output: experiments/results/
"""
import os, sys, csv, json, time, math, re, gc, random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from itertools import combinations
from contextlib import contextmanager

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'experiments', 'results')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, BASE)
from backend.ai_detector import SensitivityDetector, _generate_synthetic_dataset

# Monkey-patch batch_predict onto SensitivityDetector for vectorized evaluation
def _batch_predict(self, texts):
    texts_clean = [str(t).strip() for t in texts]
    results = []
    chunk_size = 10000
    for i in range(0, len(texts_clean), chunk_size):
        chunk = texts_clean[i:i+chunk_size]
        probs = self.pipeline.predict_proba(chunk)[:, 1]
        for prob in probs:
            results.append((bool(prob >= 0.5), round(float(prob), 4)))
    return results
SensitivityDetector.batch_predict = _batch_predict

from quantum_tokenizer import QuantumTokenizer
from risk_scoring import RiskScorer
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

SEED = 4201
np.random.seed(SEED)
random.seed(SEED)
RNG = random.Random(SEED)

# ── Try psutil for memory/CPU tracking ──
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    print("  [INFO] psutil not installed — memory/CPU measurements disabled")

# ── Load benchmark metadata ──
with open(os.path.join(BASE, 'datasets', 'metadata', 'benchmark_metadata.json')) as f:
    META = json.load(f)

DATASET_NAMES = [
    'HR_Benchmark_20K.csv',
    'Healthcare_Benchmark_50K.csv',
    'Banking_Benchmark_100K.csv',
    'Adversarial_Benchmark_20K.csv',
    'Scalability_Benchmark_1M.csv',
]

SENSITIVE_COLS = {}  # dataset → set of sensitive column names
ALL_COLS = {}
for name in DATASET_NAMES:
    info = META['datasets'][name]
    SENSITIVE_COLS[name] = set(info['high_risk'])
    ALL_COLS[name] = info['columns']

# ── Sampling limits for AI evaluation ──
MAX_EVAL_ROWS = {
    'HR_Benchmark_20K.csv': 20000,
    'Healthcare_Benchmark_50K.csv': 50000,
    'Banking_Benchmark_100K.csv': 50000,   # sample 50K of 100K
    'Adversarial_Benchmark_20K.csv': 20000,
    'Scalability_Benchmark_1M.csv': 50000, # sample 50K of 1M
}

# ======================================================================
# REGEX BASELINE DETECTOR
# ======================================================================
class RegexDetector:
    """Pure regex pattern matching baseline."""
    PATTERNS = [
        (r'^[\w\.\+\-]+@[\w\.\-]+\.[a-zA-Z]{2,}$', 'email'),
        (r'^\+?\d[\d\-\(\)\s]{7,20}\d$', 'phone'),
        (r'^\d{3}-\d{2}-\d{4}$', 'ssn'),
        (r'^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$', 'credit_card'),
        (r'^\$\d{1,3}(,\d{3})*(\.\d{2})?$', 'salary'),
        (r'^\d{1,3}(,\d{3})*(\.\d{2})?$', 'numeric_amount'),
        (r'^\d+\s+\w+\s+(St|Ave|Blvd|Dr|Ln|Way|Ct|Rd|Cir|Pl)\b', 'address_street'),
        (r'PO Box \d+', 'po_box'),
    ]

    def predict(self, text):
        text_s = str(text).strip()
        for pat, _ in self.PATTERNS:
            if re.match(pat, text_s):
                return (True, 0.9)
        return (False, 0.1)

    def detect(self, data_dict):
        results = []
        for field, value in data_dict.items():
            if value is None:
                continue
            sens, conf = self.predict(str(value))
            results.append({'field': field, 'original': str(value)[:50],
                            'sensitive': sens, 'confidence': conf})
        return results

    def batch_predict(self, texts):
        return [self.predict(t) for t in texts]


class EnsembleDetector:
    """Full QuantEdge: regex patterns first, ML fallback for uncertainty."""
    def __init__(self, ml_detector):
        self.regex = RegexDetector()
        self.ml = ml_detector

    def predict(self, text):
        text_s = str(text).strip()
        # Regex first
        is_sens, conf = self.regex.predict(text_s)
        if is_sens:
            return (True, max(conf, 0.85))
        # ML fallback for values not matched by regex
        return self.ml.predict(text_s)

    def detect(self, data_dict):
        results = []
        for field, value in data_dict.items():
            if value is None:
                continue
            sens, conf = self.predict(str(value))
            results.append({'field': field, 'original': str(value)[:50],
                            'sensitive': sens, 'confidence': conf})
        return results

    def batch_predict(self, texts):
        # Regex first for all texts
        regex_results = [self.regex.predict(t) for t in texts]
        # Find which need ML fallback
        ml_texts = []
        ml_indices = []
        for i, (sens, conf) in enumerate(regex_results):
            if not sens:
                ml_texts.append(texts[i])
                ml_indices.append(i)

        if ml_texts:
            ml_results = self.ml.batch_predict(ml_texts)
            for idx, (sens, conf) in zip(ml_indices, ml_results):
                regex_results[idx] = (sens, conf)

        return regex_results


# ======================================================================
# DATA LOADERS
# ======================================================================
# ENSEMBLE DETECTOR (Regex → ML fallback)
# ======================================================================
# DATA LOADERS
# ======================================================================
def load_dataset(name, max_rows=None):
    path = os.path.join(BASE, 'datasets', 'benchmark', name)
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(row)
    return rows


# ======================================================================
# EVALUATION METRICS COMPUTATION
# ======================================================================
def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute all classification metrics."""
    if len(set(y_true)) < 2:
        # Single class — handle gracefully
        acc = accuracy_score(y_true, y_pred)
        return {
            'precision': 1.0 if acc == 1.0 else 0.0,
            'recall': 1.0 if acc == 1.0 else 0.0,
            'f1': 1.0 if acc == 1.0 else 0.0,
            'accuracy': acc,
            'auc': 1.0 if acc == 1.0 else 0.5,
        }
    try:
        auc_val = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    except:
        auc_val = 0.0

    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': round(auc_val, 4),
    }


@contextmanager
def track_resources():
    """Context manager to track time, memory, CPU."""
    start = time.perf_counter()
    mem_before = psutil.Process().memory_info().rss / (1024**2) if HAVE_PSUTIL else 0
    cpu_before = psutil.Process().cpu_percent(interval=None) if HAVE_PSUTIL else 0

    class Tracker:
        pass
    tracker = Tracker()
    tracker.start_time = start
    tracker.mem_before = mem_before
    tracker.cpu_before = cpu_before

    yield tracker

    elapsed = time.perf_counter() - start
    mem_after = psutil.Process().memory_info().rss / (1024**2) if HAVE_PSUTIL else 0
    cpu_after = psutil.Process().cpu_percent(interval=0.1) if HAVE_PSUTIL else 0

    tracker.elapsed = elapsed
    tracker.mem_delta = mem_after - mem_before if HAVE_PSUTIL else 0
    tracker.mem_peak = max(mem_before, mem_after) if HAVE_PSUTIL else 0
    tracker.cpu_avg = (cpu_before + cpu_after) / 2 if HAVE_PSUTIL else 0


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: AI DETECTOR EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_dataset(detector, rows, sens_fields, all_cols):
    """Evaluate detector on a dataset using batch prediction."""
    all_texts = []
    all_gts = []
    for row in rows:
        for col in all_cols:
            if col in ('RiskScore', 'RiskCategory'):
                continue
            val = row.get(col, '')
            if not val or str(val).strip() == '':
                continue
            all_texts.append(str(val))
            all_gts.append(1 if col in sens_fields else 0)
    predictions = detector.batch_predict(all_texts)
    y_pred = [1 if p[0] else 0 for p in predictions]
    y_prob = [p[1] for p in predictions]
    return all_gts, y_pred, y_prob


def run_experiment_1(detectors):
    """Run AI detector evaluation across all datasets."""
    print("\n" + "█"*65)
    print("  EXPERIMENT 1: AI DETECTOR EVALUATION")
    print("█"*65)

    all_results = []

    for ds_name in DATASET_NAMES:
        print(f"\n  ── Dataset: {ds_name} ──")
        max_rows = MAX_EVAL_ROWS[ds_name]
        rows = load_dataset(ds_name, max_rows=max_rows)
        sens_fields = SENSITIVE_COLS[ds_name]
        all_cols = ALL_COLS[ds_name]

        print(f"    Loaded {len(rows)} rows, {len(all_cols)} columns", end='', flush=True)

        for det_name, detector in detectors:
            t0 = time.perf_counter()
            y_true, y_pred, y_prob = evaluate_dataset(detector, rows, sens_fields, all_cols)
            elapsed = time.perf_counter() - t0

            metrics = compute_metrics(y_true, y_pred, y_prob)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            result = {
                'Dataset': ds_name.replace('_Benchmark_', ' ').replace('.csv',''),
                'Method': det_name,
                'Samples': len(y_true),
                'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
                'Precision': round(metrics['precision'], 4),
                'Recall': round(metrics['recall'], 4),
                'F1': round(metrics['f1'], 4),
                'Accuracy': round(metrics['accuracy'], 4),
                'AUC': round(metrics['auc'], 4),
            }
            all_results.append(result)
            print(f"\n    [{det_name:15s}] {len(y_true):6d} preds in {elapsed:.1f}s  "
                  f"F1={result['F1']:.4f}  Acc={result['Accuracy']:.4f}  "
                  f"AUC={result['AUC']:.4f}  TP={tp} FP={fp} FN={fn} TN={tn}")

    df = pd.DataFrame(all_results)
    path = os.path.join(OUT, 'table_ai_results.csv')
    df.to_csv(path, index=False)
    print(f"\n  -> Saved: {path}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: TOKENIZATION SCALABILITY
# ═══════════════════════════════════════════════════════════════════════

def run_experiment_2():
    """Tokenization scalability at 10K, 50K, 100K, 500K, 1M."""
    print("\n" + "█"*65)
    print("  EXPERIMENT 2: TOKENIZATION SCALABILITY")
    print("█"*65)

    tokenizer = QuantumTokenizer()
    sizes = [10000, 50000, 100000, 500000, 1000000]
    results = []

    for n in sizes:
        records = [f'user{i}@company.com_field_{i}' for i in range(n)]

        with track_resources() as tr:
            count = 0
            for val in records:
                ta, tb = tokenizer.generate_token_pair()
                count += 1

        elapsed = tr.elapsed
        tps = count / elapsed if elapsed > 0 else 0
        latency_us = (elapsed / count) * 1_000_000 if count > 0 else 0
        mem_mb = round(tr.mem_peak, 1)
        cpu = round(tr.cpu_avg, 1)

        results.append({
            'Records': n,
            'TokensGenerated': count,
            'Time_s': round(elapsed, 4),
            'Throughput_tps': round(tps, 1),
            'Latency_us': round(latency_us, 1),
            'Memory_MB': mem_mb,
            'CPU_Percent': cpu,
        })
        print(f"  {n:7d} records: {elapsed:8.4f}s | {tps:9.1f} t/s | "
              f"{latency_us:7.1f} µs | {mem_mb:6.1f} MB | {cpu:5.1f}% CPU")

    df = pd.DataFrame(results)
    path = os.path.join(OUT, 'table_tokenization_scalability.csv')
    df.to_csv(path, index=False)
    print(f"\n  -> Saved: {path}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: BLOCKCHAIN PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════

def run_experiment_3():
    """Blockchain operations at scale."""
    print("\n" + "█"*65)
    print("  EXPERIMENT 3: BLOCKCHAIN PERFORMANCE")
    print("█"*65)

    batch_sizes = [100, 500, 1000]
    all_results = []

    for batch_size in batch_sizes:
        print(f"\n  ── Batch size: {batch_size} ──")
        tokenizer = QuantumTokenizer()

        store_times = []
        verify_times = []
        revoke_times = []
        store_gas = []
        revoke_gas = []

        from backend.blockchain_manager import BlockchainManager

        for i in range(batch_size):
            ta, tb = tokenizer.generate_token_pair()
            uid = f'EXP3_{i:06d}_{ta[:8]}'

            bc = BlockchainManager()

            # Store
            t0 = time.perf_counter()
            store_res = bc.store_token(uid, f'Field_{i}', 50, 30, 'HR', 'SALES')
            t_store = time.perf_counter() - t0
            store_times.append(t_store)
            store_gas.append(store_res.get('gas_used', 0))

            # Verify
            t0 = time.perf_counter()
            verified = bc.validate_token(uid, f'Field_{i}')
            t_verify = time.perf_counter() - t0
            verify_times.append(t_verify)

            # Revoke
            t0 = time.perf_counter()
            revoke_res = bc.revoke_token(store_res.get('token_id', ''))
            t_revoke = time.perf_counter() - t0
            revoke_times.append(t_revoke)
            revoke_gas.append(revoke_res.get('gas_used', 0) if revoke_res else 0)

            if (i + 1) % 100 == 0:
                print(f"    {i+1:4d}/{batch_size} | "
                      f"Store: {np.mean(store_times[-100:]):.5f}s | "
                      f"Verify: {np.mean(verify_times[-100:]):.5f}s | "
                      f"Revoke: {np.mean(revoke_times[-100:]):.5f}s")

        all_results.append({
            'Operations': batch_size,
            'StoreTimeMean_s': round(np.mean(store_times), 5),
            'StoreTimeStd_s': round(np.std(store_times), 5),
            'VerifyTimeMean_s': round(np.mean(verify_times), 5),
            'VerifyTimeStd_s': round(np.std(verify_times), 5),
            'RevokeTimeMean_s': round(np.mean(revoke_times), 5),
            'RevokeTimeStd_s': round(np.std(revoke_times), 5),
            'GasStoreMean': round(np.mean(store_gas), 0),
            'GasStoreStd': round(np.std(store_gas), 0),
            'GasRevokeMean': round(np.mean(revoke_gas), 0),
        })

        print(f"  Summary [{batch_size:4d} ops]: "
              f"Store={np.mean(store_times):.5f}s±{np.std(store_times):.5f}s "
              f"Verify={np.mean(verify_times):.5f}s±{np.std(verify_times):.5f}s "
              f"Revoke={np.mean(revoke_times):.5f}s±{np.std(revoke_times):.5f}s")

    df = pd.DataFrame(all_results)
    path = os.path.join(OUT, 'table_blockchain_performance.csv')
    df.to_csv(path, index=False)
    print(f"\n  -> Saved: {path}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: RISK DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def run_experiment_4():
    """Compute risk distribution across all datasets."""
    print("\n" + "█"*65)
    print("  EXPERIMENT 4: RISK DISTRIBUTION ANALYSIS")
    print("█"*65)

    all_results = []

    for ds_name in DATASET_NAMES:
        info = META['datasets'][ds_name]
        rows = load_dataset(ds_name, max_rows=None)

        high_cols = set(info['high_risk'])
        med_cols = set(info['medium_risk'])
        low_cols = set(info['low_risk'])
        total_cols = len(info['columns'])

        # Count field-level risk
        field_counts = {'high': len(high_cols), 'medium': len(med_cols),
                        'low': len(low_cols)}

        # Count per-row risk scores
        row_high = row_med = row_low = 0
        for row in rows:
            try:
                rs = int(row.get('RiskScore', 50))
            except:
                rs = 50
            if rs >= 70:
                row_high += 1
            elif rs >= 40:
                row_med += 1
            else:
                row_low += 1

        result = {
            'Dataset': ds_name.replace('_Benchmark_', ' ').replace('.csv',''),
            'TotalRows': len(rows),
            'TotalFields': total_cols,
            'HighRiskFields': field_counts['high'],
            'MediumRiskFields': field_counts['medium'],
            'LowRiskFields': field_counts['low'],
            'HighRiskRows': row_high,
            'MediumRiskRows': row_med,
            'LowRiskRows': row_low,
            'HighRiskPct': round(100*row_high/len(rows), 1),
            'MediumRiskPct': round(100*row_med/len(rows), 1),
            'LowRiskPct': round(100*row_low/len(rows), 1),
        }
        all_results.append(result)

        print(f"  {result['Dataset']:25s} fields H/M/L: "
              f"{field_counts['high']}/{field_counts['medium']}/{field_counts['low']}  |  "
              f"rows: {row_high}H/{row_med}M/{row_low}L")

    df = pd.DataFrame(all_results)
    path = os.path.join(OUT, 'table_risk_distribution.csv')
    df.to_csv(path, index=False)
    print(f"\n  -> Saved: {path}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: ADVERSARIAL ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════

def run_experiment_5(detectors):
    """Compare detection on normal vs adversarial data."""
    print("\n" + "█"*65)
    print("  EXPERIMENT 5: ADVERSARIAL ROBUSTNESS")
    print("█"*65)

    # Build "normal" dataset from HR + Healthcare + Banking
    normal_rows = []
    for ds_name in ['HR_Benchmark_20K.csv', 'Healthcare_Benchmark_50K.csv',
                     'Banking_Benchmark_100K.csv']:
        rows = load_dataset(ds_name, max_rows=10000)
        normal_rows.extend(rows)
    RNG.shuffle(normal_rows)
    normal_rows = normal_rows[:20000]

    # Build all high-risk field sets
    normal_sens_fields = set()
    for ds_name in ['HR_Benchmark_20K.csv', 'Healthcare_Benchmark_50K.csv',
                     'Banking_Benchmark_100K.csv']:
        normal_sens_fields |= SENSITIVE_COLS[ds_name]

    # Load adversarial dataset
    adv_rows = load_dataset('Adversarial_Benchmark_20K.csv', max_rows=20000)
    adv_sens_fields = SENSITIVE_COLS['Adversarial_Benchmark_20K.csv']
    adv_all_cols = ALL_COLS['Adversarial_Benchmark_20K.csv']

    all_results = []

    for data_name, rows, sens_fields, all_cols in [
        ('Normal (HR+HC+BN)', normal_rows, normal_sens_fields,
         set(ALL_COLS['HR_Benchmark_20K.csv']) |
         set(ALL_COLS['Healthcare_Benchmark_50K.csv']) |
         set(ALL_COLS['Banking_Benchmark_100K.csv'])),
        ('Adversarial', adv_rows, adv_sens_fields, set(adv_all_cols)),
    ]:
        print(f"\n  ── {data_name} ({len(rows)} rows) ──")

        for det_name, detector in detectors:
            t0 = time.perf_counter()
            y_true, y_pred, y_prob = evaluate_dataset(detector, rows, sens_fields,
                                                       list(sens_fields | set(rows[0].keys())))
            elapsed = time.perf_counter() - t0

            metrics = compute_metrics(y_true, y_pred, y_prob)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            result = {
                'Dataset': data_name,
                'Method': det_name,
                'Samples': len(y_true),
                'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
                'Precision': round(metrics['precision'], 4),
                'Recall': round(metrics['recall'], 4),
                'F1': round(metrics['f1'], 4),
                'Accuracy': round(metrics['accuracy'], 4),
                'AUC': round(metrics['auc'], 4),
            }
            all_results.append(result)
            print(f"    [{det_name:15s}] {len(y_true):6d} preds in {elapsed:.1f}s  "
                  f"F1={result['F1']:.4f}  Acc={result['Accuracy']:.4f}  "
                  f"AUC={result['AUC']:.4f}  TP={tp} FP={fp} FN={fn} TN={tn}")

    # Compute accuracy drop for each method
    normal_results = {r['Method']: r for r in all_results if r['Dataset'] == 'Normal (HR+HC+BN)'}
    adv_results = {r['Method']: r for r in all_results if r['Dataset'] == 'Adversarial'}

    for method in normal_results:
        nr = normal_results[method]
        ar = adv_results[method]
        f1_drop = round(nr['F1'] - ar['F1'], 4)
        acc_drop = round(nr['Accuracy'] - ar['Accuracy'], 4)
        fp_change = ar['FP'] - nr['FP']
        fn_change = ar['FN'] - nr['FN']
        print(f"\n    [{method:15s}] Adversarial Impact:")
        print(f"      F1 drop:       {f1_drop:.4f}")
        print(f"      Accuracy drop: {acc_drop:.4f}")
        print(f"      FP change:     {fp_change:+d}")
        print(f"      FN change:     {fn_change:+d}")

    df = pd.DataFrame(all_results)
    path = os.path.join(OUT, 'table_adversarial_results.csv')
    df.to_csv(path, index=False)
    print(f"\n  -> Saved: {path}")

    # Also save adversarial impact summary
    impact_rows = []
    for method in normal_results:
        nr = normal_results[method]
        ar = adv_results[method]
        impact_rows.append({
            'Method': method,
            'Normal_F1': nr['F1'],
            'Adversarial_F1': ar['F1'],
            'F1_Drop': round(nr['F1'] - ar['F1'], 4),
            'Normal_Acc': nr['Accuracy'],
            'Adversarial_Acc': ar['Accuracy'],
            'Accuracy_Drop': round(nr['Accuracy'] - ar['Accuracy'], 4),
            'Normal_FP': nr['FP'],
            'Adversarial_FP': ar['FP'],
            'FP_Change': ar['FP'] - nr['FP'],
            'Normal_FN': nr['FN'],
            'Adversarial_FN': ar['FN'],
            'FN_Change': ar['FN'] - nr['FN'],
        })
    impact_df = pd.DataFrame(impact_rows)
    impact_path = os.path.join(OUT, 'table_adversarial_impact.csv')
    impact_df.to_csv(impact_path, index=False)
    print(f"  -> Saved: {impact_path}")

    return df, impact_df


# ═══════════════════════════════════════════════════════════════════════
# IEEE-STYLE TABLES GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_ieee_tables(exp1_df, exp2_df, exp3_df, exp4_df, exp5_impact_df):
    """Generate publication-ready IEEE-style tables."""
    print("\n" + "█"*65)
    print("  IEEE-STYLE TABLES")
    print("█"*65)

    lines = []

    # ── TABLE I: AI Detector Performance Summary ──
    lines.append("=" * 85)
    lines.append("TABLE I: AI DETECTOR PERFORMANCE — ALL DATASETS & METHODS")
    lines.append("=" * 85)
    lines.append(f"{'Dataset':30s} {'Method':15s} {'Prec':8s} {'Recall':8s} {'F1':8s} {'Acc':8s} {'AUC':8s}")
    lines.append("-" * 85)
    for _, r in exp1_df.iterrows():
        lines.append(f"{r['Dataset']:30s} {r['Method']:15s} "
                     f"{r['Precision']:.4f}  {r['Recall']:.4f}  "
                     f"{r['F1']:.4f}  {r['Accuracy']:.4f}  {r['AUC']:.4f}")
    lines.append("")

    # ── TABLE II: Tokenization Scalability ──
    lines.append("=" * 85)
    lines.append("TABLE II: TOKENIZATION SCALABILITY")
    lines.append("=" * 85)
    lines.append(f"{'Records':12s} {'Time (s)':12s} {'Throughput':14s} {'Latency':10s} {'Memory':10s} {'CPU':8s}")
    lines.append("-" * 85)
    for _, r in exp2_df.iterrows():
        lines.append(f"{int(r['Records']):<12d} {r['Time_s']:<12.4f} "
                     f"{r['Throughput_tps']:<14.1f} {r['Latency_us']:<10.1f} "
                     f"{r['Memory_MB']:<10.1f} {r['CPU_Percent']:<8.1f}")
    lines.append("")

    # ── TABLE III: Blockchain Performance ──
    lines.append("=" * 85)
    lines.append("TABLE III: BLOCKCHAIN TRANSACTION PERFORMANCE")
    lines.append("=" * 85)
    lines.append(f"{'Ops':8s} {'Store (s)':14s} {'Verify (s)':14s} "
                 f"{'Revoke (s)':14s} {'Gas Store':12s} {'Gas Revoke':12s}")
    lines.append("-" * 85)
    for _, r in exp3_df.iterrows():
        lines.append(f"{r['Operations']:<8d} "
                     f"{r['StoreTimeMean_s']:.5f}±{r['StoreTimeStd_s']:.5f}  "
                     f"{r['VerifyTimeMean_s']:.5f}±{r['VerifyTimeStd_s']:.5f}  "
                     f"{r['RevokeTimeMean_s']:.5f}±{r['RevokeTimeStd_s']:.5f}  "
                     f"{r['GasStoreMean']:<8.0f}    {r['GasRevokeMean']:<8.0f}")
    lines.append("")

    # ── TABLE IV: Risk Distribution ──
    lines.append("=" * 85)
    lines.append("TABLE IV: RISK DISTRIBUTION ANALYSIS")
    lines.append("=" * 85)
    lines.append(f"{'Dataset':28s} {'Rows':10s} {'H-Fields':10s} {'M-Fields':10s} "
                 f"{'L-Fields':10s} {'H-Rows%':8s}")
    lines.append("-" * 85)
    for _, r in exp4_df.iterrows():
        lines.append(f"{r['Dataset']:28s} {r['TotalRows']:<10,d} "
                     f"{r['HighRiskFields']:<10d} {r['MediumRiskFields']:<10d} "
                     f"{r['LowRiskFields']:<10d} {r['HighRiskPct']:<7.1f}%")
    lines.append("")

    # ── TABLE V: Adversarial Robustness ──
    lines.append("=" * 85)
    lines.append("TABLE V: ADVERSARIAL ROBUSTNESS — ACCURACY IMPACT")
    lines.append("=" * 85)
    lines.append(f"{'Method':15s} {'Norm F1':8s} {'Adv F1':8s} {'F1 Drop':8s} "
                 f"{'Norm Acc':8s} {'Adv Acc':8s} {'Acc Drop':8s} {'FPΔ':6s} {'FNΔ':6s}")
    lines.append("-" * 85)
    for _, r in exp5_impact_df.iterrows():
        lines.append(f"{r['Method']:15s} {r['Normal_F1']:.4f}  {r['Adversarial_F1']:.4f}  "
                     f"{r['F1_Drop']:.4f}  {r['Normal_Acc']:.4f}  "
                     f"{r['Adversarial_Acc']:.4f}  {r['Accuracy_Drop']:.4f}  "
                     f"{r['FP_Change']:+d}   {r['FN_Change']:+d}")
    lines.append("")

    # ── Summary Statistics ──
    lines.append("=" * 85)
    lines.append("EXPERIMENTAL SUMMARY")
    lines.append("=" * 85)

    # Best F1 per dataset
    for ds in exp1_df['Dataset'].unique():
        ds_df = exp1_df[exp1_df['Dataset'] == ds]
        best = ds_df.loc[ds_df['F1'].idxmax()]
        lines.append(f"  {ds:30s} Best F1: {best['F1']:.4f} ({best['Method']})")

    lines.append("")
    lines.append(f"  Tokenization:  {exp2_df['Throughput_tps'].max():.0f} tokens/s peak throughput")
    lines.append(f"  Tokenization:  {exp2_df['Latency_us'].min():.1f} µs min latency")
    lines.append(f"  Blockchain:    {exp3_df['StoreTimeMean_s'].min():.5f}s min store time")
    lines.append(f"  Blockchain:    {exp3_df['GasStoreMean'].max():.0f} max gas per store")
    lines.append(f"  Risk Coverage: {len(exp4_df)} datasets, "
                 f"{exp4_df['TotalRows'].sum():,} total rows analyzed")
    lines.append("=" * 85)

    report = '\n'.join(lines)
    path = os.path.join(OUT, 'ieee_tables.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n  -> Saved: {path}")
    print("\n" + report)


# ═══════════════════════════════════════════════════════════════════════
# PUBLICATION GRAPHS
# ═══════════════════════════════════════════════════════════════════════

def generate_graphs(exp1_df, exp2_df, exp3_df, exp4_df, exp5_impact_df):
    """Generate publication-quality graphs."""
    print("\n" + "█"*65)
    print("  GENERATING PUBLICATION GRAPHS")
    print("█"*65)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np

        matplotlib.rc('font', family='serif', serif=['Times', 'Computer Modern Roman'])
        rc = {
            'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
            'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
            'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
        }
        plt.rcParams.update(rc)
        FIGSIZE = (5.5, 4.2)
        COLORS = ['#2166ac', '#d6604d', '#4daf4a', '#ff7f00', '#984ea3']
    except Exception as e:
        print(f"  [WARN] Cannot generate graphs: {e}")
        return

    fig_dir = os.path.join(BASE, 'experiments', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # ── FIG 1: F1 Score Comparison (bar chart) ──
    fig, ax = plt.subplots(figsize=(FIGSIZE))
    datasets = exp1_df['Dataset'].unique()
    methods = exp1_df['Method'].unique()
    x = np.arange(len(datasets))
    w = 0.8 / len(methods)

    for i, method in enumerate(methods):
        vals = []
        for ds in datasets:
            row = exp1_df[(exp1_df['Dataset'] == ds) & (exp1_df['Method'] == method)]
            vals.append(row['F1'].values[0] if len(row) > 0 else 0)
        ax.bar(x + i*w - 0.4 + w/2, vals, w, label=method,
               color=COLORS[i % len(COLORS)], edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('AI Detector F1 Comparison', fontsize=12, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([d[:12] for d in datasets], fontsize=8, rotation=15)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='gray', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig_f1_comparison.png'))
    plt.close(fig)
    print(f"  -> Saved: fig_f1_comparison.png")

    # ── FIG 2: Accuracy Comparison ──
    fig, ax = plt.subplots(figsize=(FIGSIZE))
    for i, method in enumerate(methods):
        vals = []
        for ds in datasets:
            row = exp1_df[(exp1_df['Dataset'] == ds) & (exp1_df['Method'] == method)]
            vals.append(row['Accuracy'].values[0] if len(row) > 0 else 0)
        ax.bar(x + i*w - 0.4 + w/2, vals, w, label=method,
               color=COLORS[i % len(COLORS)], edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Detection Accuracy by Dataset & Method', fontsize=12, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([d[:12] for d in datasets], fontsize=8, rotation=15)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='gray', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig_accuracy_comparison.png'))
    plt.close(fig)
    print(f"  -> Saved: fig_accuracy_comparison.png")

    # ── FIG 3: AUC Comparison ──
    fig, ax = plt.subplots(figsize=(FIGSIZE))
    for i, method in enumerate(methods):
        vals = []
        for ds in datasets:
            row = exp1_df[(exp1_df['Dataset'] == ds) & (exp1_df['Method'] == method)]
            vals.append(row['AUC'].values[0] if len(row) > 0 else 0)
        ax.bar(x + i*w - 0.4 + w/2, vals, w, label=method,
               color=COLORS[i % len(COLORS)], edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('AUC-ROC', fontsize=11)
    ax.set_title('AUC-ROC by Dataset & Method', fontsize=12, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([d[:12] for d in datasets], fontsize=8, rotation=15)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='gray', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig_auc_comparison.png'))
    plt.close(fig)
    print(f"  -> Saved: fig_auc_comparison.png")

    # ── FIG 4: Tokenization Scalability (dual axis) ──
    fig, ax1 = plt.subplots(figsize=(FIGSIZE))
    color1, color2 = '#2166ac', '#d6604d'

    ax1.plot(exp2_df['Records'], exp2_df['Throughput_tps'], 'o-', color=color1,
             lw=2, markersize=5, label='Throughput')
    ax1.set_xlabel('Records', fontsize=11)
    ax1.set_ylabel('Throughput (tokens/s)', fontsize=11, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')

    ax2 = ax1.twinx()
    ax2.plot(exp2_df['Records'], exp2_df['Latency_us'], 's--', color=color2,
             lw=2, markersize=5, label='Latency')
    ax2.set_ylabel('Latency (µs/token)', fontsize=11, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', frameon=True,
               facecolor='white', edgecolor='gray')
    ax1.set_title('Tokenization Scalability', fontsize=12, pad=8)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig_tokenization_scalability.png'))
    plt.close(fig)
    print(f"  -> Saved: fig_tokenization_scalability.png")

    # ── FIG 5: Blockchain Performance ──
    fig, ax = plt.subplots(figsize=(FIGSIZE))
    x = np.arange(len(exp3_df))
    w = 0.25

    store = exp3_df['StoreTimeMean_s'].values
    verify = exp3_df['VerifyTimeMean_s'].values
    revoke = exp3_df['RevokeTimeMean_s'].values

    ax.bar(x - w, store, w, label='Store', color='#2166ac', edgecolor='white', linewidth=0.5)
    ax.bar(x, verify, w, label='Verify', color='#4daf4a', edgecolor='white', linewidth=0.5)
    ax.bar(x + w, revoke, w, label='Revoke', color='#d6604d', edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Operations', fontsize=11)
    ax.set_ylabel('Mean Time (s)', fontsize=11)
    ax.set_title('Blockchain Transaction Performance', fontsize=12, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(o)}' for o in exp3_df['Operations']], fontsize=9)
    ax.legend(frameon=True, facecolor='white', edgecolor='gray')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig_blockchain_performance.png'))
    plt.close(fig)
    print(f"  -> Saved: fig_blockchain_performance.png")

    # ── FIG 6: Risk Distribution ──
    fig, ax = plt.subplots(figsize=(FIGSIZE))
    x = np.arange(len(exp4_df))
    w = 0.25

    ax.bar(x - w, exp4_df['HighRiskFields'].values, w, label='High Risk',
           color='#d6604d', edgecolor='white', linewidth=0.5)
    ax.bar(x, exp4_df['MediumRiskFields'].values, w, label='Medium Risk',
           color='#ffa500', edgecolor='white', linewidth=0.5)
    ax.bar(x + w, exp4_df['LowRiskFields'].values, w, label='Low Risk',
           color='#4daf4a', edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Fields', fontsize=11)
    ax.set_title('Risk Distribution by Dataset', fontsize=12, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r['Dataset'][:15] for _, r in exp4_df.iterrows()], fontsize=8, rotation=15)
    ax.legend(frameon=True, facecolor='white', edgecolor='gray')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig_risk_distribution.png'))
    plt.close(fig)
    print(f"  -> Saved: fig_risk_distribution.png")

    # ── FIG 7: Adversarial Robustness ──
    fig, ax = plt.subplots(figsize=(FIGSIZE))
    methods = exp5_impact_df['Method'].values
    x = np.arange(len(methods))
    w = 0.35

    normal_f1 = exp5_impact_df['Normal_F1'].values
    adv_f1 = exp5_impact_df['Adversarial_F1'].values

    ax.bar(x - w/2, normal_f1, w, label='Normal Data', color='#2166ac', edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, adv_f1, w, label='Adversarial', color='#d6604d', edgecolor='white', linewidth=0.5)

    # Annotate drops
    for i, (_, r) in enumerate(exp5_impact_df.iterrows()):
        drop = r['F1_Drop']
        ax.annotate(f'↓{drop:.3f}', xy=(i, min(normal_f1[i], adv_f1[i])),
                    xytext=(i, min(normal_f1[i], adv_f1[i]) - 0.08),
                    ha='center', fontsize=8, color='#d6604d', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('Adversarial Robustness: F1 Comparison', fontsize=12, pad=8)
    ax.set_ylim([0, 1.15])
    ax.legend(frameon=True, facecolor='white', edgecolor='gray')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig_adversarial_robustness.png'))
    plt.close(fig)
    print(f"  -> Saved: fig_adversarial_robustness.png")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  QUANTEDGE — PUBLICATION-GRADE EXPERIMENT RUNNER")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Output:  {OUT}/")
    print("=" * 65)
    print()
    print("  Experiments:")
    print("    1. AI Detector (Regex / ML / Full QuantEdge)")
    print("    2. Tokenization Scalability (10K–1M)")
    print("    3. Blockchain Performance (100–5000 ops)")
    print("    4. Risk Distribution Analysis")
    print("    5. Adversarial Robustness")
    print("    6. IEEE Tables + Publication Graphs")
    print()

    # ── Initialize detectors ──
    print("◆ Initializing detectors...")
    t0 = time.perf_counter()

    # Regex detector
    regex_det = RegexDetector()
    print(f"  RegexDetector ready")

    # ML detector (train on synthetic data)
    ml_det = SensitivityDetector()
    print("  Training ML detector (synthetic data)...")
    ml_metrics = ml_det.train(n_per_class=3000)
    print(f"  ML trained: F1={ml_metrics['f1_score']:.4f}")

    # Ensemble detector
    ensemble_det = EnsembleDetector(ml_det)
    print(f"  EnsembleDetector ready")

    detectors = [('Regex', regex_det), ('ML-Only', ml_det), ('Full QuantEdge', ensemble_det)]
    print(f"  Initialization: {time.perf_counter()-t0:.2f}s")

    # ── Run all experiments ──
    exp1_df = run_experiment_1(detectors)

    # Clear memory before exp2
    gc.collect()

    exp2_df = run_experiment_2()

    gc.collect()
    exp3_df = run_experiment_3()

    gc.collect()
    exp4_df = run_experiment_4()

    gc.collect()
    exp5_df, exp5_impact_df = run_experiment_5(detectors)

    # ── Generate outputs ──
    generate_ieee_tables(exp1_df, exp2_df, exp3_df, exp4_df, exp5_impact_df)

    generate_graphs(exp1_df, exp2_df, exp3_df, exp4_df, exp5_impact_df)

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  EXPERIMENTS COMPLETE")
    print(f"  All outputs in: {OUT}/")
    print("=" * 65)
    print(f"\n  Output files:")
    print(f"    {OUT}/table_ai_results.csv")
    print(f"    {OUT}/table_tokenization_scalability.csv")
    print(f"    {OUT}/table_blockchain_performance.csv")
    print(f"    {OUT}/table_risk_distribution.csv")
    print(f"    {OUT}/table_adversarial_results.csv")
    print(f"    {OUT}/table_adversarial_impact.csv")
    print(f"    {OUT}/ieee_tables.txt")
    print(f"    {os.path.join(BASE, 'experiments', 'figures')}/ (7 publication-quality PNGs)")
