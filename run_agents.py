"""
run_agents.py — AI Finance Controller Agent Layer Runner
=========================================================
Runs the LangGraph multi-agent pipeline on top of the deterministic audit output.

Pre-requisite:
  1. run_audit.py must have already run (produces data/audit_claims.csv)
  2. GOOGLE_API_KEY must be set in the environment

Usage:
  $env:GOOGLE_API_KEY = "your-key"
  py -3.11 run_agents.py

Output:
  - Prints investigation summaries and reviews to terminal
  - Writes data/dispute_letter.md  — formal dispute letter to gateway
  - Prints routing decision and batch summary
"""

import csv
import os
import sys
import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.contracts.resolver import load_contracts, load_contract_rules
from src.audit.fee_variance import audit_fee_variances
from src.reconciliation.bank_matching import load_settlements, load_bank_feed, match_settlements_to_bank
from src.rootcause.classifier import classify_fee_root_cause
from src.confidence.scorer import compute_confidence_score
from src.evidence.builder import build_fee_claim_item, build_bank_claim_item
from src.leakage.aggregator import aggregate_audit_results
from src.agent.graph import run_agent_audit
from src.agent.orchestrator import route_batch_outcomes

DATA_DIR = Path(__file__).parent / "data"


def _load_raw(filename: str) -> list[dict]:
    path = DATA_DIR / filename
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    print("\n" + "=" * 70)
    print("  AI FINANCE CONTROLLER -- AGENT LAYER")
    print("  Investigator -> Reviewer -[conditional]-> Orchestrator | HumanQueue")
    print("=" * 70)

    # Check API key early
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\n[ERROR] GOOGLE_API_KEY is not set.")
        print("  Run: $env:GOOGLE_API_KEY = 'your-key-here'")
        sys.exit(1)

    # ── 1. Re-run deterministic pipeline to get typed objects ─────────────────
    print("\n[1/3] Loading deterministic audit results...")

    contracts = load_contracts(DATA_DIR / "contracts.csv")
    rules     = load_contract_rules(DATA_DIR / "contract_rules.csv")
    payments_raw    = _load_raw("payments.csv")
    settlements_raw = _load_raw("settlements.csv")
    settlements_typed = load_settlements(DATA_DIR / "settlements.csv")
    bank_feed_typed   = load_bank_feed(DATA_DIR / "bank_feed.csv")

    variance_records = audit_fee_variances(
        payments=payments_raw,
        settlements=settlements_raw,
        contracts=contracts,
        rules=rules,
    )
    bank_gaps = match_settlements_to_bank(settlements_typed, bank_feed_typed)

    # Serialise to dicts for agent tools
    variance_dicts = [r.to_dict() for r in variance_records]
    bank_gap_dicts = [g.to_dict() for g in bank_gaps]

    # Build claims
    classifications = []
    claim_items = []
    claim_idx = 1
    for rec in variance_records:
        pay_row = next((p for p in payments_raw if p["payment_id"] == rec.payment_id), {})
        gmv = Decimal(pay_row.get("monthly_gmv", "0"))
        diag = classify_fee_root_cause(rec, contracts, rules, monthly_gmv=gmv)
        classifications.append(diag)
        if rec.has_variance and rec.fee_variance_inr > Decimal("0"):
            conf = compute_confidence_score(diag.case_type, rec.fee_variance_inr)
            claim = build_fee_claim_item(f"CLM-FEE-{claim_idx:04d}", rec, diag, conf)
            claim_items.append(claim)
            claim_idx += 1
    for gap in bank_gaps:
        if gap.should_escalate:
            conf = compute_confidence_score(gap.gap_type, gap.cash_impact_inr)
            pay_match = next((p for p in payments_raw if p["payment_id"] == gap.payment_id), {})
            claim = build_bank_claim_item(f"CLM-BNK-{claim_idx:04d}", gap,
                                          pay_match.get("merchant_id", "MER001"),
                                          pay_match.get("gateway_id", "GW_ALPHA"), conf)
            claim_items.append(claim)
            claim_idx += 1

    claim_dicts = [c.to_dict() for c in claim_items]

    # Batch summary stats
    summary = aggregate_audit_results(variance_records, classifications, bank_gaps)
    batch_stats = {
        "total_transactions_audited": summary.total_transactions_audited,
        "total_volume_inr": str(summary.total_volume_inr),
        "total_fee_leakage_inr": str(summary.total_fee_leakage_inr),
        "total_bank_cash_at_risk_inr": str(summary.total_bank_cash_at_risk_inr),
        "total_combined_impact_inr": str(summary.total_combined_impact_inr),
        "fee_leakage_count": summary.fee_leakage_count,
        "bank_gap_count": summary.bank_gap_count,
        "leakage_by_root_cause": {k: str(v) for k, v in summary.leakage_by_root_cause.items()},
        "bank_gaps_by_type": summary.bank_gaps_by_type,
    }

    # Contract rules as plain dicts for Investigator tool
    rules_dicts = [
        {
            "rule_id": r.rule_id, "contract_id": r.contract_id,
            "payment_method": r.payment_method, "card_category": r.card_category,
            "volume_tier_min_gmv": str(r.volume_tier_min_gmv),
            "mdr_rate": str(r.mdr_rate), "fixed_fee": str(r.fixed_fee),
            "tax_rate": str(r.tax_rate),
        }
        for r in rules
    ]

    flagged_fee_pids = [r["payment_id"] for r in variance_dicts if r.get("has_variance") == "True" or r.get("has_variance") is True]
    flagged_bank_pids = [g["payment_id"] for g in bank_gap_dicts if g.get("should_escalate") is True]
    all_flagged = list(dict.fromkeys(flagged_fee_pids + flagged_bank_pids))  # dedupe, preserve order

    print(f"    Flagged for investigation: {len(all_flagged)} payments")
    print(f"    Fee discrepancies: {summary.fee_leakage_count}  |  Bank gaps: {summary.bank_gap_count}")

    # ── 2. Run Agent Graph ─────────────────────────────────────────────────────
    print("\n[2/3] Running LangGraph agents (this calls Gemini)...")
    print("      Agent chain: Investigator → Reviewer → Orchestrator\n")

    final_state = run_agent_audit(
        variance_records=variance_dicts,
        bank_gaps=bank_gap_dicts,
        claim_items=claim_dicts,
        contract_rules_dicts=rules_dicts,
        batch_summary_stats=batch_stats,
        payment_ids=all_flagged[:9],  # process the 9 ground-truth flagged payments
    )

    # ── 3. Output ─────────────────────────────────────────────────────────────
    print("\n[3/4] Agent verdicts:\n")

    reviews = final_state.get("reviews", [])
    routing = final_state.get("routing_decision", "unknown")
    human_pids = final_state.get("human_review_pids", [])

    print(f"  Reviews completed: {len(reviews)}")
    for rev in reviews:
        if rev["approved"]:
            status = "[APPROVED]   "
        else:
            status = "[HUMAN REVIEW]"
        cause = rev.get('llm_root_cause') or rev.get('det_root_cause') or '?'
        agr = rev.get('reviewer_agreement', '?')
        print(f"    {status}  {rev['payment_id']:8s}  {cause:35s}  agreement={agr:8s}  conf={rev['confidence_score']}")

    if human_pids:
        print(f"\n  [!] Human review queue: {', '.join(human_pids)}")
        print("      These findings need analyst inspection before a claim is filed.")

    print(f"\n  Graph routing_decision : {routing.upper()}")

    # ── 4. Batch routing report ───────────────────────────────────────────────
    print("\n[4/4] Computing batch routing report...")

    batch_report = route_batch_outcomes(
        variance_records=variance_dicts,
        bank_gaps=bank_gap_dicts,
        reviews=reviews,
        routing_decision=routing,
        total_volume_inr=summary.total_volume_inr,
        total_fee_leakage_inr=summary.total_fee_leakage_inr,
        total_bank_cash_at_risk_inr=summary.total_bank_cash_at_risk_inr,
    )

    print(batch_report.as_text_report())

    # Validate invariants loudly so we know the numbers add up
    violations = batch_report.validate_invariants()
    if violations:
        print("\n[WARN] Batch report invariant violations:")
        for v in violations:
            print(f"  - {v}")
    else:
        print("  [OK] Batch invariants verified: auto_matched + investigated == total_processed")
        print(f"  [OK] Sub-buckets sum:           {batch_report.claim_ready} + {batch_report.escalated} + {batch_report.monitoring} + {batch_report.human_review} == {batch_report.investigated}")

    # ── 5. Dispute letter ────────────────────────────────────────────────────
    letter = final_state.get("dispute_letter", "")
    if letter:
        letter_path = DATA_DIR / "dispute_letter.md"
        with open(letter_path, "w", encoding="utf-8") as f:
            f.write(letter)
        print(f"\n  Dispute letter saved -> {letter_path}")
    elif routing == "human_review":
        print("\n  No dispute letter filed (all findings in human review queue).")

    batch_summary = final_state.get("batch_summary", "")
    if batch_summary:
        print(f"\n  Batch narrative:\n  {batch_summary[:400]}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
