"""
run_audit.py — AI Finance Controller Core Audit Engine
======================================================
CLI runner for the deterministic audit and reconciliation pipeline.

Executes:
  1. Load CSV data (contracts, contract_rules, payments, settlements, bank_feed).
  2. Fee variance detection (contract rules vs settlement deductions).
  3. Bank reconciliation (settlements vs bank statement postings).
  4. Root cause classification & confidence scoring.
  5. Evidence claim pack generation.
  6. Batch financial rollups.
"""

import csv
import sys
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.contracts.models import Contract, ContractRule
from src.contracts.resolver import load_contracts, load_contract_rules
from src.audit.fee_variance import audit_fee_variances
from src.reconciliation.bank_matching import load_settlements, load_bank_feed, match_settlements_to_bank
from src.rootcause.classifier import classify_fee_root_cause, classify_lifecycle_anomaly
from src.confidence.scorer import compute_confidence_score
from src.evidence.builder import build_fee_claim_item, build_bank_claim_item, ClaimItem
from src.leakage.aggregator import aggregate_audit_results, AuditBatchSummary

DATA_DIR = Path(__file__).parent / "data"
TWO_DP = Decimal("0.01")


def _f(val: Decimal) -> str:
    return f"{val.quantize(TWO_DP, rounding=ROUND_HALF_UP):,.2f}"


def load_raw_csv(filename: str) -> list[dict]:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_full_audit(verbose: bool = True) -> tuple[AuditBatchSummary, list[ClaimItem]]:
    """
    Run the end-to-end deterministic audit pipeline.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("  AI FINANCE CONTROLLER — DETERMINISTIC AUDIT PIPELINE")
        print("=" * 70)

    # 1. Load Data
    contracts = load_contracts(DATA_DIR / "contracts.csv")
    rules = load_contract_rules(DATA_DIR / "contract_rules.csv")
    payments_raw = load_raw_csv("payments.csv")
    settlements_raw = load_raw_csv("settlements.csv")
    settlements_typed = load_settlements(DATA_DIR / "settlements.csv")
    bank_feed_typed = load_bank_feed(DATA_DIR / "bank_feed.csv")

    if verbose:
        print(f"\n[1/5] Loaded data sources:")
        print(f"      - Contracts:    {len(contracts)} contracts, {len(rules)} fee rules")
        print(f"      - Payments:     {len(payments_raw)} transactions")
        print(f"      - Settlements:  {len(settlements_raw)} records")
        print(f"      - Bank Feed:    {len(bank_feed_typed)} bank postings")

    # 2. Run Fee Variance Engine
    variance_records = audit_fee_variances(
        payments=payments_raw,
        settlements=settlements_raw,
        contracts=contracts,
        rules=rules,
    )

    # 3. Run Bank Reconciliation Engine
    bank_gaps = match_settlements_to_bank(
        settlements=settlements_typed,
        bank_feed=bank_feed_typed,
    )

    # 4. Root Cause Classification & Confidence Scoring
    classifications = []
    claim_items: list[ClaimItem] = []
    claim_idx = 1

    # Classify Fee Variances
    for rec in variance_records:
        pay_row = next((p for p in payments_raw if p["payment_id"] == rec.payment_id), {})
        gmv = Decimal(pay_row.get("monthly_gmv", "0"))
        diag = classify_fee_root_cause(rec, contracts, rules, monthly_gmv=gmv)
        classifications.append(diag)

        if rec.has_variance and rec.fee_variance_inr > Decimal("0"):
            conf = compute_confidence_score(diag.case_type, rec.fee_variance_inr)
            claim = build_fee_claim_item(
                claim_id=f"CLM-FEE-{claim_idx:04d}",
                record=rec,
                classification=diag,
                confidence=conf,
            )
            claim_items.append(claim)
            claim_idx += 1

    # Classify Bank Gaps
    for gap in bank_gaps:
        if gap.should_escalate:
            conf = compute_confidence_score(gap.gap_type, gap.cash_impact_inr)
            # Find merchant & gateway
            pay_match = next((p for p in payments_raw if p["payment_id"] == gap.payment_id), {})
            merch = pay_match.get("merchant_id", "MER001")
            gw = pay_match.get("gateway_id", "GW_ALPHA")
            
            bank_claim = build_bank_claim_item(
                claim_id=f"CLM-BNK-{claim_idx:04d}",
                gap=gap,
                merchant_id=merch,
                gateway_id=gw,
                confidence=conf,
            )
            claim_items.append(bank_claim)
            claim_idx += 1

    # 5. Financial Aggregation
    summary = aggregate_audit_results(
        variance_records=variance_records,
        classifications=classifications,
        bank_gaps=bank_gaps,
    )

    # 6. Save Claim Packet to CSV
    claim_csv_path = DATA_DIR / "audit_claims.csv"
    if claim_items:
        with open(claim_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=claim_items[0].to_dict().keys())
            writer.writeheader()
            for c in claim_items:
                writer.writerow(c.to_dict())

    # 7. Print Executive Report
    if verbose:
        print("\n" + "-" * 70)
        print("  EXECUTIVE AUDIT SUMMARY")
        print("-" * 70)
        print(f"  Total Processed Volume:      Rs {_f(summary.total_volume_inr):>15}")
        print(f"  Total Fee Leakage Detected:  Rs {_f(summary.total_fee_leakage_inr):>15}  ({summary.fee_leakage_count} discrepancies)")
        print(f"  Total Bank Cash at Risk:     Rs {_f(summary.total_bank_cash_at_risk_inr):>15}  ({summary.bank_gap_count} gaps)")
        print(f"  ------------------------------------------------------------")
        print(f"  TOTAL FINANCIAL IMPACT:      Rs {_f(summary.total_combined_impact_inr):>15}")
        print("-" * 70)

        print("\n[Leakage by Merchant]")
        for m, amt in summary.leakage_by_merchant.items():
            print(f"  - {m:10s} : Rs {_f(amt):>12}")

        print("\n[Leakage by Root Cause]")
        for rc, amt in summary.leakage_by_root_cause.items():
            print(f"  - {rc:30s} : Rs {_f(amt):>12}")

        print("\n[Bank Gaps by Type]")
        for gt, cnt in summary.bank_gaps_by_type.items():
            print(f"  - {gt:30s} : {cnt} events")

        print(f"\n[Generated Claims] Saved {len(claim_items)} actionable claims -> {claim_csv_path}")
        print("=" * 70 + "\n")

    return summary, claim_items


if __name__ == "__main__":
    run_full_audit(verbose=True)
