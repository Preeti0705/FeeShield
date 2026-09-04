# 🛡️ FeeShield — AI Finance Controller

A **deterministic-first, LLM-augmented** financial audit system that detects payment gateway fee overcharges, reconciles settlements against bank postings, and computes real-time cash-impact positions — built as a 10-day progressive engineering project.

---

## What it does

Payment gateways charge MDR (Merchant Discount Rate) fees based on a signed contract. In practice, they routinely apply:

- Wrong MDR rates (wrong payment method category)
- Stale contract rates after a rate renegotiation
- GST on the gross transaction amount instead of on the MDR fee
- Duplicate fee deductions in the same settlement batch
- Missed volume-tier discounts when GMV crosses a threshold

FeeShield detects all of these **deterministically**, then uses a **Gemini-powered multi-agent pipeline** (Investigator → Reviewer → Orchestrator) to classify findings, cross-check the AI's labels against the deterministic engine's labels, and produce a formal dispute letter and batch routing report.

Finally, it computes a **treasury cash-position verdict** — *SAFE / AT_RISK / CRITICAL* — comparing cleared cash against 7-day obligations.

---

## Benchmark results (ground truth: 13 cases)

| Metric | Score |
|---|---|
| **Precision** | **100%** |
| **Recall** | **100%** |
| **False Positive Rate** | **0%** |
| **F1 Score** | **1.0** |
| True Positives | 9 / 9 |
| True Negatives | 4 / 4 |
| False Positives | 0 |
| False Negatives | 0 |

> Run `python benchmark.py` to reproduce these numbers (requires `run_audit.py` to have been run first).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER (CSVs)                            │
│  payments · settlements · bank_feed · contracts · contract_rules    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────▼──────────────────────┐
        │          DETERMINISTIC ENGINE (Days 1-5)      │
        │                                               │
        │  Contract Resolver   ← contracts.csv          │
        │       ↓                                       │
        │  Fee Calculator      ← expected MDR + tax     │
        │       ↓                                       │
        │  Fee Variance Auditor ← expected vs charged   │
        │       ↓                                       │
        │  Bank Matcher        ← settlement vs bank     │
        │       ↓                                       │
        │  Root Cause Classifier                        │
        │       ↓                                       │
        │  Evidence Builder    → audit_claims.csv       │
        │       ↓                                       │
        │  Aggregator          → AuditBatchSummary      │
        └───────────────────────┬──────────────────────┘
                                │
        ┌───────────────────────▼──────────────────────┐
        │        AI AGENT PIPELINE (Days 5-7)          │
        │                                              │
        │  Investigator (Gemini) → structured JSON     │
        │       ↓                                      │
        │  Reviewer (deterministic) → label agreement  │
        │       ↓ conditional routing                  │
        │  ┌────┴────────────┐                         │
        │  Orchestrator(LLM) │  Human Queue            │
        │  dispute letter    │  escalation             │
        │  └────────────────┘                          │
        │       ↓                                      │
        │  BatchRoutingReport (deterministic math)     │
        └───────────────────────┬──────────────────────┘
                                │
        ┌───────────────────────▼──────────────────────┐
        │        CASH POSITION (Day 8)                 │
        │                                              │
        │  cleared_cash + expected_inflows − at_risk   │
        │  vs 7-day obligations                        │
        │  → SAFE / AT_RISK / CRITICAL                 │
        └───────────────────────┬──────────────────────┘
                                │
        ┌───────────────────────▼──────────────────────┐
        │        DASHBOARD (Day 9)                     │
        │                                              │
        │  Streamlit · 4 tabs · No sidebar             │
        │  Executive Summary / Fee Analysis /          │
        │  Bank Reconciliation / Cash Position         │
        └──────────────────────────────────────────────┘
```

---

## Project structure

```
feeshield/
├── src/
│   ├── contracts/          # Contract resolver + rule models
│   ├── fees/               # Fee calculator (Decimal-strict)
│   ├── audit/              # Fee variance engine
│   ├── reconciliation/     # Settlement-to-bank matcher (5 gap types)
│   ├── rootcause/          # Root cause classifier
│   ├── confidence/         # Confidence scorer
│   ├── evidence/           # Claim item builder + CSV export
│   ├── leakage/            # Batch aggregator (AuditBatchSummary)
│   ├── agent/              # LangGraph multi-agent pipeline
│   │   ├── state.py        # AuditState typed dict
│   │   ├── nodes.py        # Investigator + Reviewer + Orchestrator nodes
│   │   ├── graph.py        # LangGraph graph + conditional routing
│   │   └── orchestrator.py # BatchRoutingReport + invariant validation
│   └── cash_impact/        # Cash-position calculator (Day 8)
│       └── position_calculator.py
├── tests/
│   ├── test_contracts.py   # 18 tests — contract date/tier/method resolution
│   ├── test_fees.py        # 17 tests — fee calculation + rounding + leakage
│   ├── test_bank_matching.py # 27 tests — all 5 gap types + integration
│   ├── test_agents.py      # 53 tests — agent nodes + graph + orchestrator
│   └── test_cash_position.py # 30 tests — verdict + math + gap categorisation
├── data/
│   ├── generate_data.py    # Synthetic data generator (238 payments)
│   ├── ground_truth.csv    # 13 hand-labelled cases for benchmarking
│   └── *.csv               # payments, settlements, bank_feed, contracts…
├── dashboard.py            # Streamlit UI (4 tabs, no sidebar)
├── run_audit.py            # Deterministic pipeline runner
├── run_agents.py           # LLM agent pipeline runner
├── benchmark.py            # Precision/recall evaluator vs ground truth
├── GUIDE.md                # Day-by-day build log + conceptual explanations
└── requirements.txt
```

---

## Setup

**Prerequisites:** Python 3.12+ (tested on 3.14.2), Windows/macOS/Linux

```powershell
# 1. Clone and enter the project
cd C:\Users\PREETI\Desktop\feeshield

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data
$env:PYTHONUTF8 = "1"
python data/generate_data.py
```

---

## Usage

### Run the deterministic audit (no API key needed)

```powershell
$env:PYTHONUTF8 = "1"
python run_audit.py
```

Outputs:
- Executive summary with fee leakage + bank gaps
- Cash position report (SAFE / AT_RISK / CRITICAL)
- `data/audit_claims.csv` — all actionable claims

### Run the benchmark against ground truth

```powershell
python benchmark.py
```

Prints precision, recall, F1, and recovery rate against 13 hand-labelled cases.

### Run the AI agent pipeline (requires Gemini API key)

```powershell
$env:GOOGLE_API_KEY = "your-key-here"
python run_agents.py
```

Outputs:
- Investigator → Reviewer → Orchestrator verdicts
- Batch routing report (auto-matched / investigated / claim-ready / escalated)
- Formal dispute letter

### Launch the dashboard

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\streamlit.exe run dashboard.py
# Open http://localhost:8501
```

### Run all tests

```powershell
python -m pytest tests/ -v
# 140 passed — zero live API calls
```

---

## Key design rules (enforced throughout)

| Rule | Why |
|---|---|
| **All money is `Decimal`, never `float`** | `0.1 + 0.2 != 0.3` in float; Rs 1 paisa wrong in a million-Rs settlement claim is a legal problem |
| **`ROUND_HALF_UP` everywhere** | Matches standard banking rounding; Python default is `ROUND_HALF_EVEN` |
| **Deterministic core, LLM at the edge** | The engine can be audited, replayed, and unit-tested; LLM adds language not arithmetic |
| **Mock LLM in all unit tests** | `pytest` never touches the API; every test is offline and deterministic |
| **Invariants enforced at runtime** | `BatchRoutingReport.validate_invariants()` and `CashPosition.validate()` raise `AssertionError` if the numbers don't sum — the terminal output is proven correct |
| **No Docker, no Redis, no React** | Runs anywhere with Python + pip; demo-able on a laptop with no infra |

---

## The 5 gap types detected

| Gap type | What it means | Cash impact |
|---|---|---|
| `SETTLEMENT_NOT_POSTED` | Gateway settled; bank has no record | Full settled amount at risk |
| `SETTLEMENT_POSTED_LATE` | Bank posted > 3 days after settlement | Delay risk (working capital) |
| `POSTED_AMOUNT_MISMATCH` | Bank posted less than the settled net amount | Difference amount lost |
| `POSTING_REVERSED` | Bank posting was reversed — net credit = 0 | Full posting amount lost |
| `HOLD_PLACED_THEN_CLEARED` | Hold placed but cleared within SLA | Monitoring only — no claim |

---

## The 5 root causes detected

| Root cause | Description |
|---|---|
| `wrong_mdr` | Gateway applied wrong MDR rate for the payment method |
| `missed_volume_tier` | GMV crossed a tier threshold; gateway used previous (higher) rate |
| `wrong_tax_base` | GST computed on gross transaction amount instead of on MDR fee |
| `duplicate_fee` | MDR + GST deducted twice in same settlement |
| `contract_version_violation` | Gateway used rates from an expired contract version |

---

## Build log

See [`GUIDE.md`](./GUIDE.md) for the complete day-by-day build narrative — every conceptual decision explained, every command shown, and every test result recorded.

| Day | What was built | Tests |
|---|---|---|
| 1-2 | Data generator, contract resolver, fee calculator | 35 |
| 3A | Root cause classifier, confidence scorer, evidence builder | — |
| 3B | Settlement-to-bank matcher (5 gap types) | 27 |
| 4 | Leakage aggregator, audit batch summary | — |
| 5 | LangGraph agent graph, Investigator node | — |
| 6 | Reviewer node (deterministic), human queue, conditional routing | 53 |
| 7 | Orchestrator routing, BatchRoutingReport + invariants | 53 |
| 8 | Cash position calculator, SAFE/AT_RISK/CRITICAL verdict | 30 |
| 9 | Streamlit dashboard (4 tabs) | — |
| 10 | Benchmark (precision/recall), README | 140 total |

---

## Requirements

```
langchain-google-genai
langgraph
google-generativeai
google-genai
streamlit
pandas
pytest
```

> The deterministic pipeline and dashboard have **no API key requirement**. Only `run_agents.py` calls Gemini.
