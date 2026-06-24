"""
QuantEdge — Publication-Grade Experimental Framework
=====================================================
IEEE Conference-style evaluation with:
  - AI detector audit (leakage, cross-val, confusion matrix, overfitting)
  - Blockchain performance (store/verify/revoke times + gas)
  - Tokenization performance (scalability)
  - Risk scoring evaluation
  - Statistical summaries + CSV output

Output: research_output/ directory with tables, CSVs, and summaries.
"""
import os, sys, time, json, math, hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

# ── Configure paths ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'experiments', 'results')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, BASE)
from backend.ai_detector import SensitivityDetector, _generate_synthetic_dataset
from quantum_tokenizer import QuantumTokenizer
from risk_scoring import RiskScorer
from backend.blockchain_manager import BlockchainManager

SEED = 42
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════
# 1. AI DETECTOR AUDIT
# ═══════════════════════════════════════════════════════════════════════

def audit_ml_pipeline():
    """Complete ML audit: leakage, duplicates, bias, overfitting, cross-val, confusion matrix."""
    print("\n" + "█"*60)
    print("SECTION 1: AI DETECTOR AUDIT")
    print("█"*60)

    report = {}
    ts = datetime.now().isoformat()

    # 1.1 Generate full dataset for audit
    texts, labels = _generate_synthetic_dataset(n_per_class=5000)
    df = pd.DataFrame({'text': texts, 'label': labels})
    n_sens = sum(labels)
    n_nons = len(labels) - n_sens
    print(f"\n1.1 Dataset: {len(df)} samples ({n_sens} sensitive, {n_nons} non-sensitive)")

    # 1.2 Check for duplicates
    dupes = df.duplicated(subset='text').sum()
    print(f"1.2 Duplicate samples: {dupes} ({100*dupes/len(df):.2f}%)")
    report['duplicate_pct'] = round(100*dupes/len(df), 2)

    # 1.3 Check train-test leakage (exact string overlap)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    overlap = len(set(X_tr) & set(X_te))
    print(f"1.3 Train-test overlap (exact strings): {overlap} samples")
    report['train_test_overlap'] = overlap

    # 1.4 Character diversity analysis (synthetic data bias)
    sens_texts = [t for t, l in zip(texts, labels) if l == 1]
    nons_texts = [t for t, l in zip(texts, labels) if l == 0]
    sens_chars = set(''.join(sens_texts))
    nons_chars = set(''.join(nons_texts))
    unique_to_sens = sens_chars - nons_chars
    unique_to_nons = nons_chars - sens_chars
    print(f"1.4 Character set overlap: {len(sens_chars & nons_chars)}/{len(sens_chars | nons_chars)} shared")
    print(f"    Unique to sensitive: {unique_to_sens}")
    print(f"    Unique to non-sensitive: {unique_to_nons}")
    report['chars_shared'] = len(sens_chars & nons_chars)
    report['chars_sens_unique'] = str(unique_to_sens)
    report['chars_nons_unique'] = str(unique_to_nons)
    report['total_samples'] = len(df)

    # Average length analysis
    avg_len_sens = np.mean([len(t) for t in sens_texts])
    avg_len_nons = np.mean([len(t) for t in nons_texts])
    print(f"    Avg length sensitive: {avg_len_sens:.1f}, non-sensitive: {avg_len_nons:.1f}")
    report['avg_len_sens'] = round(avg_len_sens, 1)
    report['avg_len_nons'] = round(avg_len_nons, 1)

    # 1.5 Cross-validation (5-fold)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix, classification_report as sk_report
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
    all_y_true = []
    all_y_pred = []

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), max_features=5000, sublinear_tf=True)),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=SEED)),
    ])

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        X_tr_f = [texts[i] for i in train_idx]
        y_tr_f = [labels[i] for i in train_idx]
        X_te_f = [texts[i] for i in test_idx]
        y_te_f = [labels[i] for i in test_idx]

        pipeline.fit(X_tr_f, y_tr_f)
        y_pred_f = pipeline.predict(X_te_f)

        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        cv_scores['precision'].append(precision_score(y_te_f, y_pred_f))
        cv_scores['recall'].append(recall_score(y_te_f, y_pred_f))
        cv_scores['f1'].append(f1_score(y_te_f, y_pred_f))
        cv_scores['accuracy'].append(accuracy_score(y_te_f, y_pred_f))
        all_y_true.extend(y_te_f)
        all_y_pred.extend(y_pred_f)

    # Aggregate confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n1.5 5-Fold Cross-Validation Results:")
    print(f"    Precision: {np.mean(cv_scores['precision']):.4f} ± {np.std(cv_scores['precision']):.4f}")
    print(f"    Recall:    {np.mean(cv_scores['recall']):.4f} ± {np.std(cv_scores['recall']):.4f}")
    print(f"    F1 Score:  {np.mean(cv_scores['f1']):.4f} ± {np.std(cv_scores['f1']):.4f}")
    print(f"    Accuracy:  {np.mean(cv_scores['accuracy']):.4f} ± {np.std(cv_scores['accuracy']):.4f}")

    report['cv_precision_mean'] = round(np.mean(cv_scores['precision']), 4)
    report['cv_precision_std'] = round(np.std(cv_scores['precision']), 4)
    report['cv_recall_mean'] = round(np.mean(cv_scores['recall']), 4)
    report['cv_recall_std'] = round(np.std(cv_scores['recall']), 4)
    report['cv_f1_mean'] = round(np.mean(cv_scores['f1']), 4)
    report['cv_f1_std'] = round(np.std(cv_scores['f1']), 4)
    report['cv_accuracy_mean'] = round(np.mean(cv_scores['accuracy']), 4)
    report['cv_accuracy_std'] = round(np.std(cv_scores['accuracy']), 4)

    # Confusion matrix
    print(f"\n1.6 Confusion Matrix (aggregated across 5 folds):")
    print(f"                 Predicted Neg   Predicted Pos")
    print(f"    Actual Neg   {tn:6d}          {fp:6d}")
    print(f"    Actual Pos   {fn:6d}          {tp:6d}")
    print(f"\n    True Positives:  {tp}")
    print(f"    True Negatives:  {tn}")
    print(f"    False Positives: {fp}")
    print(f"    False Negatives: {fn}")

    # Derived metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0

    report['tpr'] = round(tpr, 4)
    report['tnr'] = round(tnr, 4)
    report['ppv'] = round(ppv, 4)
    report['npv'] = round(npv, 4)
    report['fpr'] = round(fpr_val, 4)
    report['fnr'] = round(fnr_val, 4)

    # 1.7 Overfitting test: compare train vs test on same fold
    print(f"\n1.7 Overfitting Check (fold 0):")
    pipeline.fit(X_tr_f, y_tr_f)
    train_acc = pipeline.score(X_tr_f, y_tr_f)
    test_acc = pipeline.score(X_te_f, y_te_f)
    gap = train_acc - test_acc
    print(f"    Train accuracy: {train_acc:.4f}")
    print(f"    Test accuracy:  {test_acc:.4f}")
    print(f"    Gap:            {gap:.4f} {'(OVERFITTING ⚠️)' if gap > 0.05 else '(OK)'}")
    report['overfit_gap'] = round(gap, 4)

    # 1.8 AUC-ROC
    from sklearn.metrics import roc_auc_score
    y_prob_all = []
    pipeline2 = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), max_features=5000, sublinear_tf=True)),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=SEED)),
    ])
    pipeline2.fit(X_tr, y_tr)
    y_prob = pipeline2.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    print(f"\n1.8 AUC-ROC: {auc:.4f}")
    report['auc_roc'] = round(auc, 4)

    # Save CV results
    cv_df = pd.DataFrame({
        'Fold': list(range(1, 6)),
        'Precision': cv_scores['precision'],
        'Recall': cv_scores['recall'],
        'F1': cv_scores['f1'],
        'Accuracy': cv_scores['accuracy'],
    })
    cv_df.to_csv(os.path.join(OUT, 'table_cross_validation.csv'), index=False)

    # Save confusion matrix
    cm_df = pd.DataFrame({
        '': ['Actual Negative', 'Actual Positive'],
        'Predicted Negative': [tn, fn],
        'Predicted Positive': [fp, tp],
    })
    cm_df.to_csv(os.path.join(OUT, 'table_confusion_matrix.csv'), index=False)

    # Save full classification report
    with open(os.path.join(OUT, 'ai_audit_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    return report, cv_df


# ═══════════════════════════════════════════════════════════════════════
# 2. BLOCKCHAIN BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════

def benchmark_blockchain():
    """Store/Verify/Revoke token benchmarks with gas usage."""
    print("\n" + "█"*60)
    print("SECTION 2: BLOCKCHAIN BENCHMARKS")
    print("█"*60)

    tokenizer = QuantumTokenizer()
    results = []

    for trial in range(10):
        ta, tb = tokenizer.generate_token_pair()
        uid = f'QT_{ta[:12]}_bench_{trial}'

        # Fresh blockchain for each trial to avoid hash collisions
        bc = BlockchainManager()

        # ── STORE ──
        t0 = time.perf_counter()
        store_res = bc.store_token(uid, f'Field_Bench', 50, 30, 'HR', 'SALES')
        t_store = time.perf_counter() - t0

        # ── VERIFY ──
        t0 = time.perf_counter()
        verified = bc.validate_token(uid, f'Field_Bench')
        t_verify = time.perf_counter() - t0

        # ── REVOKE ──
        t0 = time.perf_counter()
        revoke_res = bc.revoke_token(store_res['token_id'])
        t_revoke = time.perf_counter() - t0

        results.append({
            'Trial': trial + 1,
            'StoreTime_s': round(t_store, 5),
            'VerifyTime_s': round(t_verify, 5),
            'RevokeTime_s': round(t_revoke, 5),
            'GasStore': store_res['gas_used'],
            'GasRevoke': revoke_res['gas_used'] if revoke_res else 0,
            'BlockNumber': store_res['block_number'],
            'TokenID': store_res['token_id'],
        })
        print(f"  Trial {trial+1:2d}: Store={t_store:.5f}s (gas={store_res['gas_used']:>6d}) "
              f"Verify={t_verify:.5f}s Revoke={t_revoke:.5f}s", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, 'table_blockchain_benchmark.csv'), index=False)

    # Summary statistics
    summary = {
        'store_time_mean': round(df['StoreTime_s'].mean(), 5),
        'store_time_std': round(df['StoreTime_s'].std(), 5),
        'verify_time_mean': round(df['VerifyTime_s'].mean(), 5),
        'verify_time_std': round(df['VerifyTime_s'].std(), 5),
        'revoke_time_mean': round(df['RevokeTime_s'].mean(), 5),
        'revoke_time_std': round(df['RevokeTime_s'].std(), 5),
        'gas_store_mean': round(df['GasStore'].mean(), 0),
        'gas_store_std': round(df['GasStore'].std(), 0),
        'gas_revoke_mean': round(df['GasRevoke'].mean(), 0),
        'gas_revoke_std': round(df['GasRevoke'].std(), 0),
    }
    with open(os.path.join(OUT, 'blockchain_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary:")
    print(f"    Store:  {summary['store_time_mean']:.5f}s ± {summary['store_time_std']:.5f}s (gas: {summary['gas_store_mean']:.0f})")
    print(f"    Verify: {summary['verify_time_mean']:.5f}s ± {summary['verify_time_std']:.5f}s")
    print(f"    Revoke: {summary['revoke_time_mean']:.5f}s ± {summary['revoke_time_std']:.5f}s (gas: {summary['gas_revoke_mean']:.0f})")

    return df, summary


# ═══════════════════════════════════════════════════════════════════════
# 3. TOKENIZATION SCALABILITY
# ═══════════════════════════════════════════════════════════════════════

def benchmark_scalability():
    """Test tokenization performance across dataset sizes."""
    print("\n" + "█"*60)
    print("SECTION 3: TOKENIZATION SCALABILITY")
    print("█"*60)

    tokenizer = QuantumTokenizer()
    sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    results = []

    for n in sizes:
        values = [f'user{i}@company.com' for i in range(n)]

        t0 = time.perf_counter()
        count = 0
        for v in values:
            ta, tb = tokenizer.generate_token_pair()
            count += 1
        elapsed = time.perf_counter() - t0

        tps = count / elapsed if elapsed > 0 else 0
        results.append({
            'Records': n,
            'TokensGenerated': count,
            'Time_s': round(elapsed, 5),
            'Throughput_tps': round(tps, 1),
            'Latency_per_token_us': round(elapsed / count * 1_000_000, 1) if count > 0 else 0,
        })
        print(f"  {n:5d} records: {elapsed:.5f}s | {tps:>8.1f} tokens/s | {results[-1]['Latency_per_token_us']:>8.1f} µs/token")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, 'table_scalability.csv'), index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 4. RISK SCORING EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def benchmark_risk_scoring():
    """Evaluate risk scoring across field types and departments."""
    print("\n" + "█"*60)
    print("SECTION 4: RISK SCORING EVALUATION")
    print("█"*60)

    test_cases = [
        # (field_name, value, source_dept)
        ('Name', 'John Michael Smith', 'HR'),
        ('Email', 'john.smith@example.com', 'HR'),
        ('Phone', '+1-212-555-0198', 'SALES'),
        ('SSN', '123-45-6789', 'FINANCE'),
        ('Salary', '150000', 'HR'),
        ('CreditCard', '4111-1111-1111-1111', 'FINANCE'),
        ('Address', '742 Evergreen Terrace, Springfield, IL 62701', 'HR'),
        ('Department', 'Engineering', 'HR'),
        ('Position', 'Senior Manager', 'SALES'),
        ('Status', 'Active', 'IT'),
        ('EmployeeID', 'EMP-0042', 'HR'),
        ('Diagnosis', 'Type 2 Diabetes', 'HR'),
        ('BloodType', 'A+', 'HR'),
        ('City', 'New York', 'SALES'),
        ('Age', '34', 'HR'),
    ]

    results = []
    for field, value, dept in test_cases:
        risk = RiskScorer.compute_risk(field, value, dept)
        results.append({
            'Field': field,
            'Value': value[:40],
            'Department': dept,
            'RiskScore': risk['score'],
            'Tier': risk['tier'],
            'ExpiryMin': risk['expiry_minutes'],
            'Entropy': risk['entropy'],
            'FieldScore': risk['field_score'],
            'PatternContrib': risk['pattern_contribution'],
            'DeptModifier': risk['department_modifier'],
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, 'table_risk_scoring.csv'), index=False)

    print(f"\n  {'Field':15s} {'Risk':5s} {'Tier':8s} {'Expiry':6s} {'Entropy':8s}")
    print(f"  {'-'*45}")
    for _, r in df.iterrows():
        print(f"  {r['Field']:15s} {r['RiskScore']:3d}   {r['Tier']:8s} {r['ExpiryMin']:2d}m    {r['Entropy']:.2f}")

    return df


# ═══════════════════════════════════════════════════════════════════════
# 5. IEEE-STYLE TABLE GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_ieee_tables(ml_report, cv_df, bc_df, bc_summary, scal_df, risk_df):
    """Generate publication-ready IEEE-style tables as text."""
    print("\n" + "█"*60)
    print("SECTION 5: IEEE-STYLE TABLES")
    print("█"*60)

    lines = []

    # ── Table I: AI Detector Performance ──
    lines.append("="*75)
    lines.append("TABLE I: AI SENSITIVITY DETECTOR — CLASSIFICATION PERFORMANCE")
    lines.append("="*75)
    lines.append(f"{'Metric':25s} {'Value':12s} {'Std Dev':12s}")
    lines.append("-"*75)
    lines.append(f"{'Precision':25s} {ml_report['cv_precision_mean']:.4f}      ±{ml_report['cv_precision_std']:.4f}")
    lines.append(f"{'Recall (TPR)':25s} {ml_report['cv_recall_mean']:.4f}      ±{ml_report['cv_recall_std']:.4f}")
    lines.append(f"{'F1 Score':25s} {ml_report['cv_f1_mean']:.4f}      ±{ml_report['cv_f1_std']:.4f}")
    lines.append(f"{'Accuracy':25s} {ml_report['cv_accuracy_mean']:.4f}      ±{ml_report['cv_accuracy_std']:.4f}")
    lines.append(f"{'AUC-ROC':25s} {ml_report['auc_roc']:.4f}")
    lines.append(f"{'True Positive Rate':25s} {ml_report['tpr']:.4f}")
    lines.append(f"{'True Negative Rate':25s} {ml_report['tnr']:.4f}")
    lines.append(f"{'False Positive Rate':25s} {ml_report['fpr']:.4f}")
    lines.append(f"{'False Negative Rate':25s} {ml_report['fnr']:.4f}")
    lines.append(f"{'Overfitting Gap':25s} {ml_report['overfit_gap']:.4f}")
    lines.append("-"*75)
    lines.append(f"Classifier: LogisticRegression with TF-IDF (char_wb, ngram 2-5)")
    lines.append(f"Cross-validation: 5-fold stratified | Dataset: {ml_report.get('total_samples', 10000)} synthetic samples")
    lines.append("")

    # ── Table II: Confusion Matrix ──
    lines.append("="*75)
    lines.append("TABLE II: CONFUSION MATRIX (AGGREGATED 5-FOLD)")
    lines.append("="*75)
    lines.append(f"{'':25s} {'Predicted Negative':20s} {'Predicted Positive':20s}")
    lines.append("-"*75)
    cm_df2 = pd.read_csv(os.path.join(OUT, 'table_confusion_matrix.csv'), index_col=0)
    for label, row in cm_df2.iterrows():
        lines.append(f"{label:25s} {int(row['Predicted Negative']):10d}              {int(row['Predicted Positive']):10d}")
    lines.append("")

    # ── Table III: Cross-Validation Detail ──
    lines.append("="*75)
    lines.append("TABLE III: 5-FOLD CROSS-VALIDATION DETAIL")
    lines.append("="*75)
    lines.append(f"{'Fold':8s} {'Precision':12s} {'Recall':12s} {'F1':12s} {'Accuracy':12s}")
    lines.append("-"*75)
    for _, r in cv_df.iterrows():
        lines.append(f"{int(r['Fold']):3d}     {r['Precision']:.4f}      {r['Recall']:.4f}      {r['F1']:.4f}      {r['Accuracy']:.4f}")
    lines.append("")

    # ── Table IV: Blockchain Performance ──
    lines.append("="*75)
    lines.append("TABLE IV: BLOCKCHAIN TRANSACTION PERFORMANCE (LOCAL EVM)")
    lines.append("="*75)
    lines.append(f"{'Operation':15s} {'Mean Time (s)':15s} {'Std Dev (s)':15s} {'Mean Gas':12s}")
    lines.append("-"*75)
    lines.append(f"{'Store Token':15s} {bc_summary['store_time_mean']:<15.5f} {bc_summary['store_time_std']:<15.5f} {bc_summary['gas_store_mean']:<12.0f}")
    lines.append(f"{'Verify Token':15s} {bc_summary['verify_time_mean']:<15.5f} {bc_summary['verify_time_std']:<15.5f} {'N/A':>12s}")
    lines.append(f"{'Revoke Token':15s} {bc_summary['revoke_time_mean']:<15.5f} {bc_summary['revoke_time_std']:<15.5f} {bc_summary['gas_revoke_mean']:<12.0f}")
    lines.append("-"*75)
    lines.append(f"Smart contract: TokenStore.sol | EVM: py-evm (local) | Consensus: PoA")
    lines.append("")

    # ── Table V: Blockchain raw data ──
    lines.append("="*75)
    lines.append("TABLE V: BLOCKCHAIN RAW TRIAL DATA (10 RUNS)")
    lines.append("="*75)
    lines.append(f"{'Trial':6s} {'Store (s)':12s} {'Verify (s)':12s} {'Revoke (s)':12s} {'Gas Store':10s} {'Gas Revoke':10s}")
    lines.append("-"*75)
    for _, r in bc_df.iterrows():
        lines.append(f"{int(r['Trial']):3d}    {r['StoreTime_s']:.5f}    {r['VerifyTime_s']:.5f}    {r['RevokeTime_s']:.5f}    {int(r['GasStore']):>6d}      {int(r['GasRevoke']):>6d}")
    lines.append("")

    # ── Table VI: Scalability ──
    lines.append("="*75)
    lines.append("TABLE VI: TOKENIZATION SCALABILITY")
    lines.append("="*75)
    lines.append(f"{'Records':10s} {'Time (s)':12s} {'Throughput (t/s)':18s} {'Latency (µs)':14s}")
    lines.append("-"*75)
    for _, r in scal_df.iterrows():
        lines.append(f"{int(r['Records']):<10d} {r['Time_s']:<12.5f} {r['Throughput_tps']:<18.1f} {r['Latency_per_token_us']:<14.1f}")
    lines.append("")

    # ── Table VII: Risk Scoring ──
    lines.append("="*75)
    lines.append("TABLE VII: RISK SCORING — FIELD-LEVEL BREAKDOWN")
    lines.append("="*75)
    lines.append(f"{'Field':15s} {'Value':30s} {'Risk':6s} {'Tier':8s} {'Exp':5s} {'Entropy':8s}")
    lines.append("-"*75)
    for _, r in risk_df.iterrows():
        lines.append(f"{r['Field']:15s} {str(r['Value'])[:30]:30s} {r['RiskScore']:3d}   {r['Tier']:8s} {r['ExpiryMin']:2d}m  {r['Entropy']:.2f}")
    lines.append("")

    # ── Statistical Summary ──
    lines.append("="*75)
    lines.append("STATISTICAL SUMMARY")
    lines.append("="*75)
    lines.append(f"AI Detector:           {ml_report['cv_f1_mean']:.4f} ± {ml_report['cv_f1_std']:.4f} F1 (5-fold CV)")
    lines.append(f"Blockchain Store:      {bc_summary['store_time_mean']:.5f}s ± {bc_summary['store_time_std']:.5f}s")
    lines.append(f"Blockchain Verify:     {bc_summary['verify_time_mean']:.5f}s ± {bc_summary['verify_time_std']:.5f}s")
    lines.append(f"Blockchain Revoke:     {bc_summary['revoke_time_mean']:.5f}s ± {bc_summary['revoke_time_std']:.5f}s")
    lines.append(f"Token Throughput:      {scal_df[scal_df['Records']==10000]['Throughput_tps'].values[0]:.0f} tokens/s at 10,000 records")
    lines.append(f"Risk Score Range:      {risk_df['RiskScore'].min()}–{risk_df['RiskScore'].max()} across {len(risk_df)} field types")
    lines.append(f"Total Blockchain TX:   {len(bc_df)} transactions")
    lines.append("="*75)

    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(OUT, 'ieee_tables.txt'), 'w') as f:
        f.write(report)

    print(f"\nAll tables saved to: {OUT}/")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*75)
    print("  QUANTEDGE — PUBLICATION-GRADE EXPERIMENTAL FRAMEWORK")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Output:  {OUT}/")
    print("="*75)
    print("  Framework: IEEE Conference Evaluation")
    print("  AI: LogisticRegression + TF-IDF | 5-fold CV | Confusion Matrix | AUC-ROC")
    print("  Blockchain: py-evm EVM | TokenStore.sol | Store/Verify/Revoke + Gas")
    print("  Scalability: 10–10,000 records | Throughput & Latency")
    print("  Risk: 15 field types | Entropy | Department Modifiers")
    print("="*75)

    # 1. ML Audit
    ml_report, cv_df = audit_ml_pipeline()

    # 2. Blockchain Benchmarks
    bc_df, bc_summary = benchmark_blockchain()

    # 3. Scalability
    scal_df = benchmark_scalability()

    # 4. Risk Scoring
    risk_df = benchmark_risk_scoring()

    # 5. IEEE Tables
    generate_ieee_tables(ml_report, cv_df, bc_df, bc_summary, scal_df, risk_df)

    print("\n" + "="*75)
    print("  EVALUATION COMPLETE")
    print(f"  All outputs in: {OUT}/")
    print("="*75)
