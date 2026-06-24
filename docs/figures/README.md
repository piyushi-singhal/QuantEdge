# QuantEdge: Publication-Ready Figure Descriptions
# =================================================
# Intended for IEEE conference paper (e.g., IEEE BigData, S&P, CCS)
# Use these captions and descriptions verbatim in your manuscript.

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: System Architecture
# ═══════════════════════════════════════════════════════════════════════
# Files:
#   docs/figures/fig1_architecture.mmd       — Mermaid (renderable)
#   docs/figures/fig1_architecture.drawio    — Draw.io (editable)
#
# Suggested figure placement: Full-page width, Section II (System Overview)
#

Figure 1 Caption:
    Fig. 1. QuantEdge system architecture. The framework comprises eight
    modular layers spanning data ingestion to secure access delivery. Solid
    arrows indicate data flow; dashed arrows represent metadata or
    configuration flow. Layer 1 performs AI-driven sensitive data detection
    using TF-IDF vectorization and logistic regression classification with
    confidence scoring. Layer 2 computes a risk score based on field
    patterns, Shannon entropy, and department-level modifiers. Layer 3
    generates cryptographically secure tokens (Token A for identification,
    Token B for blockchain anchoring) through CSPRNG seeding and SHA3-256
    hashing. Layer 4 stores token hashes on a local EVM via Solidity smart
    contract invocation and records transaction receipts for auditability.
    Layer 5 persists token-field mappings in a SQLite database. Layer 6
    enforces dynamic token expiry (5 min for high risk, 15 min for medium,
    30 min for low) based on the risk classification from Layer 2. Layer 7
    exposes FastAPI REST endpoints consumed by the React-based web
    dashboard.

Figure 1 Description (for accessibility / alt-text):
    A layered block diagram showing eight horizontal layers stacked
    vertically. Layer 0 (Enterprise Data Source) feeds into Layer 1 (AI
    Sensitivity Detection) containing TF-IDF, Logistic Regression, and
    Confidence Score modules. Layer 2 (Risk Scoring Engine) branches into
    Field Pattern Analysis, Shannon Entropy Analysis, and Department-Based
    Risk Modifier converging on Risk Classification. Layer 3 (Dynamic
    Tokenization) proceeds from CSPRNG through Dual Token Generation and
    SHA3-256 Hashing to Fernet Encryption. Layer 4 (Blockchain Audit
    Layer) includes Solidity Smart Contract, Local EVM, On-Chain Token
    Storage, and Transaction Hash Generation. Layer 5 (Secure Storage)
    contains SQLite Database. Layer 6 (Access Control) enforces RBAC with
    dynamic expiry. Layer 7 (Presentation) comprises FastAPI REST
    Endpoints and React Dashboard. Data flows top to bottom; risk
    configuration flows from Layer 2 to Layer 6 via a dashed arrow.

Rendering instructions:
    mermaid fig1_architecture.mmd -o fig1_architecture.png
    # Or use the Mermaid live editor at https://mermaid.live


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Methodology Workflow
# ═══════════════════════════════════════════════════════════════════════
# Files:
#   docs/figures/fig2_workflow.mmd          — Mermaid (renderable)
#   docs/figures/fig2_workflow.drawio       — Draw.io (editable)
#
# Suggested figure placement: Full-page width, Section III (Methodology)
#

Figure 2 Caption:
    Fig. 2. QuantEdge methodology workflow. The pipeline processes
    enterprise data through ten sequential stages with branching at risk
    classification (Step 5). High-risk and medium-risk fields undergo
    dynamic tokenization (Step 6) followed by blockchain auditability
    (Step 7). Low-risk fields bypass tokenization via a pass-through path
    directly to the access control layer (Step 8). All fields converge at
    Step 8 where
    dynamic token expiry is enforced. Authorized users view sensitive data
    per role permissions (Step 9). A parallel audit trail records all
    transactions for verification (Step 10). The dashed arrow indicates
    the optional blockchain verification path where transaction hashes are
    cross-referenced against the on-chain ledger.

Figure 2 Description (for accessibility / alt-text):
    A flowchart with ten numbered steps connected by arrows. Steps 1-4
    proceed linearly: Dataset Upload, Data Preprocessing, Sensitive Data
    Detection (TF-IDF → Logistic Regression → Confidence Score), and Risk
    Assessment (Field + Entropy + Department → Risk Score 0-100). Step 5
    is a diamond decision node classifying into three branches: High Risk
    (score ≥ 70), Medium Risk (50 ≤ score < 70), and Low Risk (score <
    50). High and Medium branches converge into Step 6 (Dynamic
    Tokenization: CSPRNG → Dual Token → SHA3-256) and Step 7 (Blockchain
    Audit: Smart Contract → Hash → Transaction Receipt). The Low branch
    follows a pass-through path directly to Step 8 (Access Control: JWT +
    Permissions + Dynamic Expiry), bypassing tokenization and blockchain.
    All branches converge at Step 8 and proceed to Step 9 (Secure Data
    Presentation). Step 10 (Blockchain Verification & Audit Logging)
    receives input from Step 9 and via a dashed verification arrow from
    Step 7 for on-chain hash cross-referencing.

Rendering instructions:
    mermaid fig2_workflow.mmd -o fig2_workflow.png
    # Or use the Mermaid live editor at https://mermaid.live
