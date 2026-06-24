"""
Real ML-based Sensitivity Detector for QuantEdge.

Architecture:
  - Feature extraction: TF-IDF vectorizer (character n-grams, word n-grams)
  - Classifier: LogisticRegression (interpretable, probability-calibrated)
  - Training: Synthetic datasets for names, emails, phones, SSNs, salaries, addresses, IDs
  - Output: Per-field confidence scores + precision/recall/F1 metrics

This is a real ML pipeline, not regex rules.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import re
import os
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Synthetic training data generators
# ---------------------------------------------------------------------------

def _generate_synthetic_dataset(n_per_class: int = 2000) -> Tuple[List[str], List[int]]:
    """Generate labeled synthetic data for sensitive field detection.
    
    Returns:
        texts: list of string values
        labels: 1 = sensitive, 0 = not sensitive
    """
    np.random.seed(42)
    texts = []
    labels = []

    first_names = ["John","Jane","Alex","Sarah","Mike","Emily","David","Emma","Chris","Olivia"]
    last_names  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez"]
    domains     = ["gmail.com","yahoo.com","outlook.com","company.org","mail.net","business.co","test.edu"]
    companies   = ["Acme","Globex","Initech","Umbrella","Cyberdyne","Wonka","Stark","Wayne","Oscorp","Massive"]
    streets     = ["Main","Oak","Elm","Maple","Cedar","Pine","Birch","Walnut","Cherry","Spruce"]
    street_types = ["St","Ave","Blvd","Rd","Dr","Ln","Way","Ct","Pl","Cir"]
    cities      = ["New York","Los Angeles","Chicago","Houston","Phoenix","Philadelphia","San Antonio","San Diego","Dallas","Austin"]
    states      = ["CA","TX","NY","FL","IL","PA","OH","GA","NC","MI"]
    departments = ["Engineering","Marketing","Sales","Finance","HR","Operations","Legal","R&D"]
    positions   = ["Manager","Analyst","Engineer","Director","Coordinator","Specialist","Lead","Associate"]
    diagnosis   = ["Diabetes","Hypertension","Asthma","Cancer","Arthritis","Depression","Anxiety","Migraine","Allergy","Thyroid"]
    blood_types = ["A+","A-","B+","B-","O+","O-","AB+","AB-"]

    # --- Sensitive class examples (label = 1) ---

    for _ in range(n_per_class):
        choice = np.random.randint(0, 6)
        if choice == 0:  # Full names
            texts.append(f"{np.random.choice(first_names)} {np.random.choice(last_names)}")
        elif choice == 1:  # Emails
            fn = np.random.choice(first_names).lower()
            ln = np.random.choice(last_names).lower()
            texts.append(f"{fn}.{ln}@{np.random.choice(domains)}")
        elif choice == 2:  # Phone numbers
            area = np.random.randint(200, 999)
            exch = np.random.randint(200, 999)
            line = np.random.randint(1000, 9999)
            texts.append(f"+1-{area}-{exch}-{line}")
        elif choice == 3:  # SSNs
            texts.append(f"{np.random.randint(100,999)}-{np.random.randint(10,99)}-{np.random.randint(1000,9999)}")
        elif choice == 4:  # Salary / income
            texts.append(f"${np.random.randint(3, 25)}0,{np.random.randint(0, 99):02d}")
        else:  # Street addresses
            texts.append(f"{np.random.randint(100, 9999)} {np.random.choice(streets)} {np.random.choice(street_types)}, {np.random.choice(cities)}, {np.random.choice(states)}")
        labels.append(1)

    # --- Non-sensitive class examples (label = 0) ---

    for _ in range(n_per_class):
        choice = np.random.randint(0, 6)
        if choice == 0:  # Department
            texts.append(np.random.choice(departments))
        elif choice == 1:  # Position
            texts.append(f"{np.random.choice(positions)}")
        elif choice == 2:  # Company names
            texts.append(f"{np.random.choice(companies)} Inc.")
        elif choice == 3:  # Generic text
            texts.append(f"Project {np.random.choice(['Alpha','Beta','Gamma','Delta','Omega'])} phase {np.random.randint(1,5)}")
        elif choice == 4:  # Status
            texts.append(np.random.choice(["Active","Inactive","Pending","Completed","Approved","Rejected"]))
        else:  # Simple codes
            texts.append(f"REF-{np.random.randint(10000, 99999)}")
        labels.append(0)

    return texts, labels


# ---------------------------------------------------------------------------
# 2. The ML Detector
# ---------------------------------------------------------------------------

class SensitivityDetector:
    """Real ML-based sensitive data detector using TF-IDF + LogisticRegression.
    
    Usage:
        detector = SensitivityDetector()
        detector.train()                          # train on synthetic data
        result = detector.detect({"name": "John Smith", "note": "Hello"})
        # result -> [{"field": "name", "sensitive": True, "confidence": 0.97}, ...]
    """

    def __init__(self, model_path: str = None):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 5),
                max_features=5000,
                sublinear_tf=True,
            )),
            ('clf', LogisticRegression(
                C=1.0,
                class_weight='balanced',
                solver='lbfgs',
                max_iter=1000,
                random_state=42,
            )),
        ])
        self._trained = False
        self._metrics = {}

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, texts: List[str] = None, labels: List[int] = None,
              n_per_class: int = 3000) -> Dict[str, float]:
        """Train the model on synthetic data (or provided data).
        
        Returns classification metrics.
        """
        if texts is None or labels is None:
            texts, labels = _generate_synthetic_dataset(n_per_class=n_per_class)

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        self.pipeline.fit(X_train, y_train)
        self._trained = True

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        self._metrics = {
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
        }

        logger.info(f"ML model trained. Metrics: {self._metrics}")
        return self._metrics

    def predict(self, text: str) -> Tuple[bool, float]:
        """Predict if a single text value is sensitive.
        
        Returns:
            (is_sensitive, confidence) where confidence is probability [0,1]
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        text_clean = str(text).strip()
        if not text_clean:
            return (False, 0.0)

        prob = float(self.pipeline.predict_proba([text_clean])[0, 1])
        return (prob >= 0.5, round(prob, 4))

    def detect(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect sensitive fields in a dictionary of field: value pairs.
        
        Returns list of dicts with keys: field, original, sensitive, confidence
        """
        results = []
        for field, value in data.items():
            if value is None:
                continue
            is_sensitive, confidence = self.predict(str(value))
            results.append({
                'field': field,
                'original': str(value)[:50],
                'sensitive': is_sensitive,
                'confidence': confidence,
            })
        return results

    def detect_dataframe(self, df: 'pd.DataFrame') -> Dict[str, Dict]:
        """Analyze all columns of a DataFrame and return per-field sensitivity."""
        import pandas as pd
        results = {}
        for col in df.columns:
            values = df[col].dropna().astype(str).tolist()
            if not values:
                results[col] = {'sensitive': False, 'confidence': 0.0, 'risk_score': 0}
                continue
            # Average confidence across all non-null values
            confs = [self.predict(v)[1] for v in values[:100]]  # sample first 100
            avg_conf = float(np.mean(confs)) if confs else 0.0
            sensitive = avg_conf >= 0.5
            risk_score = int(round(avg_conf * 100))
            results[col] = {
                'sensitive': sensitive,
                'confidence': round(avg_conf, 4),
                'risk_score': min(risk_score, 100),
            }
        return results

    def get_metrics(self) -> Dict[str, float]:
        """Return training metrics."""
        return dict(self._metrics)

    def save(self, path: str):
        """Save trained model to disk."""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'pipeline': self.pipeline, 'metrics': self._metrics}, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load trained model from disk."""
        import joblib
        data = joblib.load(path)
        self.pipeline = data['pipeline']
        self._metrics = data.get('metrics', {})
        self._trained = True
        logger.info(f"Model loaded from {path}")

    def generate_classification_report(self) -> str:
        """Generate a text report suitable for inclusion in a research paper."""
        if not self._metrics:
            return "Model not trained yet."
        lines = [
            "=" * 55,
            "SENSITIVITY DETECTOR — CLASSIFICATION REPORT",
            "=" * 55,
            f"  Classifier:    LogisticRegression (TF-IDF char_wb ngram 2-5)",
            f"  Train samples: {self._metrics.get('train_samples', 'N/A')}",
            f"  Test samples:  {self._metrics.get('test_samples', 'N/A')}",
            "",
            f"  Precision:     {self._metrics.get('precision', 0):.4f}",
            f"  Recall:        {self._metrics.get('recall', 0):.4f}",
            f"  F1 Score:      {self._metrics.get('f1_score', 0):.4f}",
            f"  Accuracy:      {self._metrics.get('accuracy', 0):.4f}",
            "-" * 55,
            "The model detects: names, emails, phone numbers, SSNs,",
            "salaries, and addresses as sensitive (label=1).",
            "Non-sensitive: departments, positions, status codes, etc.",
            "=" * 55,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Quick test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    detector = SensitivityDetector()
    metrics = detector.train(n_per_class=3000)

    print(detector.generate_classification_report())

    # Quick inference demo
    test_cases = [
        ("John Smith", "name"),
        ("john.smith@gmail.com", "email"),
        ("+1-555-123-4567", "phone"),
        ("123-45-6789", "SSN"),
        ("$120,000", "salary"),
        ("742 Evergreen Terrace, Springfield, IL", "address"),
        ("Engineering", "department"),
        ("Manager", "position"),
        ("Active", "status"),
    ]
    print("\n--- Inference demo ---")
    print(f"{'Value':40s} {'Field':15s} {'Sensitive':10s} {'Confidence':10s}")
    print("-" * 75)
    for val, fname in test_cases:
        sens, conf = detector.predict(val)
        print(f"{val:40s} {fname:15s} {'YES' if sens else 'NO':10s} {conf:.4f}")
