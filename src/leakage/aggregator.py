"""
src/leakage/aggregator.py
=========================
Batch-level rollup and metrics aggregator for financial audits.

Capabilities:
  - Aggregate fee leakage by merchant, gateway, and root cause.
  - Roll up bank gaps (settlement-to-bank cash at risk).
  - Compute total recoverable amounts and audit summary statistics.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Any

from src.audit.fee_variance import FeeVarianceRecord
from src.reconciliation.bank_matching import BankGap
from src.rootcause.classifier import RootCauseClassification

PRECISION = Decimal("0.000001")


@dataclass(frozen=True)
class AuditBatchSummary:
    """
    High-level metrics rollup across the entire audit batch.
    """
    total_transactions_audited: int
    total_volume_inr: Decimal
    total_fee_leakage_inr: Decimal
    total_bank_cash_at_risk_inr: Decimal
    total_combined_impact_inr: Decimal
    fee_leakage_count: int
    bank_gap_count: int
    leakage_by_merchant: Dict[str, Decimal]
    leakage_by_gateway: Dict[str, Decimal]
    leakage_by_root_cause: Dict[str, Decimal]
    bank_gaps_by_type: Dict[str, int]


def aggregate_audit_results(
    variance_records: list[FeeVarianceRecord],
    classifications: list[RootCauseClassification],
    bank_gaps: list[BankGap],
) -> AuditBatchSummary:
    """
    Aggregate all audit findings into a comprehensive summary.
    """
    total_txns = len(variance_records)
    total_vol = sum((r.amount for r in variance_records), Decimal("0")).quantize(PRECISION, rounding=ROUND_HALF_UP)
    
    # Fee leakage rollups
    fee_leakage_records = [r for r in variance_records if r.has_variance and r.fee_variance_inr > Decimal("0")]
    total_fee_leakage = sum((r.fee_variance_inr for r in fee_leakage_records), Decimal("0")).quantize(PRECISION, rounding=ROUND_HALF_UP)
    
    leakage_by_merch: Dict[str, Decimal] = {}
    leakage_by_gw: Dict[str, Decimal] = {}
    for r in fee_leakage_records:
        leakage_by_merch[r.merchant_id] = (leakage_by_merch.get(r.merchant_id, Decimal("0")) + r.fee_variance_inr).quantize(PRECISION, rounding=ROUND_HALF_UP)
        leakage_by_gw[r.gateway_id] = (leakage_by_gw.get(r.gateway_id, Decimal("0")) + r.fee_variance_inr).quantize(PRECISION, rounding=ROUND_HALF_UP)
        
    leakage_by_cause: Dict[str, Decimal] = {}
    class_map = {c.payment_id: c for c in classifications}
    for r in fee_leakage_records:
        cause = class_map.get(r.payment_id, None)
        cause_name = cause.case_type if cause else "unknown"
        leakage_by_cause[cause_name] = (leakage_by_cause.get(cause_name, Decimal("0")) + r.fee_variance_inr).quantize(PRECISION, rounding=ROUND_HALF_UP)

    # Bank gap rollups
    escalated_bank_gaps = [g for g in bank_gaps if g.should_escalate]
    total_bank_risk = sum((g.cash_impact_inr for g in escalated_bank_gaps), Decimal("0")).quantize(PRECISION, rounding=ROUND_HALF_UP)
    
    bank_by_type: Dict[str, int] = {}
    for g in bank_gaps:
        bank_by_type[g.gap_type] = bank_by_type.get(g.gap_type, 0) + 1

    total_combined = (total_fee_leakage + total_bank_risk).quantize(PRECISION, rounding=ROUND_HALF_UP)

    return AuditBatchSummary(
        total_transactions_audited=total_txns,
        total_volume_inr=total_vol,
        total_fee_leakage_inr=total_fee_leakage,
        total_bank_cash_at_risk_inr=total_bank_risk,
        total_combined_impact_inr=total_combined,
        fee_leakage_count=len(fee_leakage_records),
        bank_gap_count=len(escalated_bank_gaps),
        leakage_by_merchant=leakage_by_merch,
        leakage_by_gateway=leakage_by_gw,
        leakage_by_root_cause=leakage_by_cause,
        bank_gaps_by_type=bank_by_type,
    )
