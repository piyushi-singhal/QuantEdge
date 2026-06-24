# QuantEdge: AI-Driven Dynamic Tokenization with Blockchain Auditability

**QuantEdge** is a research framework for secure enterprise data sharing that combines **AI-powered sensitive data detection**, **dynamic risk-adaptive tokenization**, **blockchain-based auditability**, and **role-based access control**. It is designed for publication-quality evaluation in IEEE conferences and journals.

---

## Architecture Overview

```
Enterprise Dataset → AI Detection → Risk Scoring → Tokenization → Blockchain Audit → RBAC → Secure Access
```

QuantEdge operates in 8 modular layers:

| Layer | Component | Technology |
|-------|-----------|------------|
| 1 | AI Sensitivity Detection | TF-IDF Vectorizer + Logistic Regression |
| 2 | Risk Scoring Engine | Field heuristics + Shannon entropy + department modifiers |
| 3 | Dynamic Tokenization | CSPRNG → Dual Token → SHA3-256 → Fernet encryption |
| 4 | Blockchain Audit | Solidity Smart Contract + Local EVM (py-evm) |
| 5 | Secure Storage | SQLite token mappings |
| 6 | Access Control | RBAC with dynamic expiry (5/15/30 min per risk level) |
| 7 | API Layer | FastAPI REST endpoints |
| 8 | Dashboard | Jinja2 templates + Material UI styling |

See `docs/architecture/` for the full architecture diagram.

---

## Features

- **AI Sensitive Data Detection**: ML-based detector (TF-IDF + Logistic Regression) that identifies sensitive fields from content patterns, not just field names
- **Risk-Adaptive Tokenization**: Per-field risk scores drive dynamic token expiry and selective encryption
- **Blockchain Audit Trail**: Every tokenization event is hashed (SHA3-256) and stored on a local EVM via Solidity smart contracts
- **Role-Based Access Control**: JWT-authenticated roles with automatic token expiry enforcement
- **5 Publication-Grade Benchmark Datasets**: HR (20K), Healthcare (50K), Banking (100K), Adversarial (20K), Scalability (1M) — zero leakage, realistic data
- **Full Evaluation Suite**: 5 experiments (AI detection, tokenization, blockchain, risk, adversarial) + 5-configuration ablation study
- **Publication-Ready Outputs**: IEEE tables, CSV results, 300 DPI PNG figures

---

## Repository Structure

```
QuantEdge/
├── app.py                          # Main FastAPI application
├── models.py                       # SQLAlchemy models (User, TokenMapping, ProcessedFile)
├── quantum_tokenizer.py            # Core tokenizer (CSPRNG + SHA3-256 + Fernet)
├── process_file.py                 # CSV processing pipeline
├── file_watcher.py                 # Watchdog-based file system observer
├── tokenization_rules.py           # Department-pair field rules
├── risk_scoring.py                 # Field heuristics + entropy + risk scoring
├── events.py                       # SSE token update events
├── websocket_manager.py            # WebSocket connection manager
├── backend/
│   ├── ai_detector.py              # ML detector (TF-IDF + LogisticRegression)
│   ├── blockchain_manager.py       # Local EVM / Solidity contract manager
│   └── database.py                 # SQLAlchemy database setup
├── contracts/
│   ├── TokenStore.sol              # Solidity smart contract
│   ├── deployment.json             # Contract deployment config
│   └── build/                      # Compiled ABI + BIN
├── datasets/
│   ├── raw/                        # Source datasets (Kaggle + QuantEdge)
│   ├── benchmark/                  # 5 publication-grade benchmark CSVs
│   ├── splits/                     # Train/validation/test splits (70/15/15)
│   └── metadata/                   # Dataset stats, validation reports
├── experiments/
│   ├── results/                    # Experiment CSVs + IEEE tables
│   ├── ablation/                   # Ablation study outputs
│   └── figures/                    # 12 publication-quality PNG graphs
├── docs/
│   ├── architecture/               # System architecture diagrams
│   ├── methodology/                # Workflow methodology diagrams
│   └── figures/                    # Additional figures
├── requirements/
│   ├── base.txt                    # Runtime dependencies
│   ├── dev.txt                     # Development tools
│   └── research.txt                # Research/experiment dependencies
├── scripts/                        # Utility scripts
├── templates/                      # Jinja2 HTML templates
│   ├── index.html                  # Dashboard
│   ├── login.html                  # Login page
│   └── register.html               # Registration page
├── static/
│   ├── css/styles.css              # Dashboard styling
│   ├── css/auth.css                # Login/Register styling
│   └── js/main.js                  # Tab switching, upload, polling
├── data/
│   ├── input/                      # Upload folder
│   ├── processed/                  # Processed output files
│   ├── stats.json                  # Runtime statistics
│   └── token_mappings.json         # Token mapping records
├── archive/                        # Archived obsolete/duplicate files
├── VALIDATION.md                   # Comprehensive validation report
└── README.md                       # This file
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/QuantEdge.git
cd QuantEdge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install base dependencies
pip install -r requirements/base.txt

# For running experiments
pip install -r requirements/research.txt
```

---

## Running QuantEdge

### Start the application

```bash
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000) to access the dashboard.

### Key API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Main dashboard |
| `GET /login` | Login page |
| `GET /register` | Registration page |
| `GET /stats` | System-wide tokenization statistics |
| `GET /api/mappings/{field}` | Token mappings for a specific field |
| `GET /api/fields` | Available fields for tokenization |
| `POST /upload` | Upload a CSV file for processing |

### Authentication

Default users: `admin` / `admin123` (admin role) and `user` / `user123` (user role). Create new accounts via `/register`.

---

## Running Experiments

### Prerequisites

```bash
pip install -r requirements/research.txt
```

### Benchmark Datasets

First generate the 5 benchmark datasets:

```bash
python generate_datasets.py
```

Output: `datasets/benchmark/` (5 CSVs), `datasets/splits/` (15 train/val/test splits), `datasets/metadata/` (stats + validation report).

### Experiments

Run the 5-experiment evaluation suite:

```bash
python run_experiments.py
```

Output: `experiments/results/` (6 CSVs + IEEE tables) and `experiments/figures/` (7 PNG graphs).

### Ablation Study

Run the 5-configuration ablation study:

```bash
python run_ablation.py
```

Output: `experiments/ablation/` (CSVs + contribution analysis + IEEE table) and `experiments/figures/` (5 additional PNGs).

### Research Evaluation

Run the full ML audit + blockchain benchmarks:

```bash
python research_evaluation.py
python generate_figures.py
```

---

## Benchmark Datasets

| Dataset | Rows | Source | Sensitive Fields |
|---------|------|--------|------------------|
| HR Benchmark (20K) | 20,000 | IBM HR Kaggle + QuantEdge_HR | SSN, Email, Phone, DOB, Salary |
| Healthcare (50K) | 50,000 | Kaggle Healthcare + QuantEdge_Healthcare | SSN, Email, Phone, DOB, Diagnosis |
| Banking (100K) | 100,000 | Kaggle Bank + QuantEdge_Customer | SSN, Email, Phone, DOB, Account |
| Adversarial (20K) | 20,000 | Field-renamed HR/HC/Bank + QuantEdge_Adv | SSN, Email, Phone, DOB, Account |
| Scalability (1M) | 1,000,000 | Expanded Banking + synthetic | SSN, Email, Phone, DOB, Account |

All datasets use 70/15/15 train/validation/test splits with zero leakage verified.

---

## Research Contributions

1. **AI-Driven Detection**: Demonstrated that ML (TF-IDF + LogisticRegression) generalizes beyond regex — F1 of 0.7453 vs 0.5491 for regex baseline, with strong generalization to adversarially renamed fields (+0.0868 F1 gain).

2. **Risk-Adaptive Tokenization**: Risk scoring provides +0.0191 F1 gain by dynamically adjusting thresholds per field, with 5/15/30 minute expiry enforced at the access control layer.

3. **Blockchain Audit Trail**: Local EVM (py-evm) with Solidity smart contract provides immutable token storage at ~38 ms per operation — feasible for batch processing but not real-time.

4. **Evaluation Framework**: Publication-grade evaluation across 5 benchmark datasets (190K+ fields) with 5 experiments, 5-configuration ablation, and full ML audit.

---

## Results Summary

### Ablation Study (116,000 fields)

| Configuration | F1 | ΔF1 | Latency |
|---|---|---|---|
| Regex Baseline | 0.5491 | — | 1.25 µs |
| ML Detector Only | **0.7453** | +0.1962 | 7.89 µs |
| ML + Risk Scoring | **0.7644** | +0.0191 | 17.36 µs |
| ML + Risk + Token | 0.7644 | 0.0000 | 37.48 µs |
| Full QuantEdge | 0.7644 | 0.0000 | 34,774 µs |

**Key finding**: ML provides the largest detection gain (+0.1962 F1). Risk scoring fine-tunes (+0.0191). Tokenization and blockchain add protection/immutability at latency cost.

See `experiments/ablation/ieee_ablation_table.txt` and `experiments/results/ieee_tables.txt` for full publication-ready tables.

---

## Publication Assets

All figures and tables are ready for inclusion in IEEE/ACM conference papers:

| Asset | Location | Format |
|-------|----------|--------|
| System Architecture | `docs/architecture/fig1_architecture.mmd` | Mermaid + Draw.io |
| Methodology Workflow | `docs/methodology/fig2_workflow.mmd` | Mermaid + Draw.io |
| Experiment Results | `experiments/results/ieee_tables.txt` | LaTeX |
| Ablation Results | `experiments/ablation/ieee_ablation_table.txt` | LaTeX |
| Experiment Figures | `experiments/figures/*.png` | 300 DPI PNG |
| ML Audit Report | `experiments/results/` | CSVs + tables |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@inproceedings{quantedge2025,
  title={QuantEdge: An AI-Driven Dynamic Tokenization Framework with Blockchain-Based Auditability and Risk-Adaptive Access Control for Secure Enterprise Data Sharing},
  author={[Author Names]},
  booktitle={Proceedings of the [Conference Name]},
  year={2025}
}
```
