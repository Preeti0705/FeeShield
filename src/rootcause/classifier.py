"""
src/rootcause/classifier.py
===========================
Deterministic root-cause classifier for fee variances and payment anomalies.

Categories:
  - wrong_mdr
  - missed_volume_tier
  - wrong_tax_base
  - duplicate_fee
  - contract_version_violation
  - legitimate_refund
  - legitimate_chargeback
  - timing_difference
  - no_variance

Design Rules:
  - Purely deterministic logic based on mathematical pattern matching, contracts, and lifecycle status.
  - Generates clear, audit-ready explanations and citations.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from datetime import date
from typing import Optional, List

from src.contracts.models import Contract, ContractRule
from src.audit.fee_variance import FeeVarianceRecord

PRECISION = Decimal("0.000001")
TOLERANCE = Decimal("0.01")


@dataclass(frozen=True)
class RootCauseClassification:
    """
    Root cause diagnosis for a transaction or variance.
    """
    payment_id: str
    case_type: str
    is_anomaly: bool
    description: str
    evidence_summary: str
    suggested_action: str


def classify_fee_root_cause(
    record: FeeVarianceRecord,
    contracts: list[Contract],
    rules: list[ContractRule],
    monthly_gmv: Decimal = Decimal("0"),
) -> RootCauseClassification:
    """
    Classify the root cause of a fee variance record.
    """
    pid = record.payment_id
    amt = record.amount
    var = record.fee_variance_inr

    # 1. Duplicate settlement deduction
    if record.is_duplicate_settlement:
        return RootCauseClassification(
            payment_id=pid,
            case_type="duplicate_fee",
            is_anomaly=True,
            description=f"Duplicate fee deduction of Rs {record.actual_total_fee} in settlement {record.settlement_id}.",
            evidence_summary=f"Payment {pid} had multiple settlement rows. Duplicate deduction amounted to Rs {record.actual_total_fee}.",
            suggested_action="Request reversal and refund of duplicate fee deduction from payment gateway.",
        )

    # 2. No actionable variance
    if not record.has_variance:
        return RootCauseClassification(
            payment_id=pid,
            case_type="no_variance",
            is_anomaly=False,
            description="Fee matched contractual expectations within tolerance.",
            evidence_summary=f"Expected: Rs {record.expected_total_fee}, Actual: Rs {record.actual_total_fee}, Variance: Rs {var}.",
            suggested_action="No action required.",
        )

    # 3. Wrong Tax Base Check
    # In Case 3: GST (18%) charged on gross transaction amount rather than MDR fee
    # Gross GST = amt * 18% = amt * 0.18
    # If actual_tax matches gross GST within tolerance:
    gross_tax_calc = (amt * Decimal("18.00") / Decimal("100")).quantize(PRECISION, rounding=ROUND_HALF_UP)
    if abs(record.actual_tax - gross_tax_calc) <= TOLERANCE and record.actual_tax > record.expected_tax:
        return RootCauseClassification(
            payment_id=pid,
            case_type="wrong_tax_base",
            is_anomaly=True,
            description=f"GST charged on gross transaction amount (Rs {amt}) instead of MDR fee (Rs {record.actual_mdr_fee}).",
            evidence_summary=f"Expected GST on MDR: Rs {record.expected_tax}. Actual GST charged: Rs {record.actual_tax} (~18% of gross amount).",
            suggested_action="Issue dispute notice for improper GST tax base calculation and claim overcharge.",
        )

    # 4. Volume Tier Check
    # In Case 2: GMV threshold exceeded but base rate was applied
    if monthly_gmv >= Decimal("500000"):
        # Check all rules for this contract/method
        # If expected rule is a discounted tier, but actual MDR rate matches base tier
        active_contract_rules = [r for r in rules if r.contract_id in [c.contract_id for c in contracts if c.merchant_id == record.merchant_id]]
        base_rules = [r for r in active_contract_rules if r.payment_method == record.payment_method and r.card_category == record.card_category and r.volume_tier_min_gmv == Decimal("0")]
        if base_rules:
            base_rule = base_rules[0]
            base_mdr_fee = (amt * base_rule.mdr_rate / Decimal("100")).quantize(PRECISION, rounding=ROUND_HALF_UP)
            if abs(record.actual_mdr_fee - base_mdr_fee) <= TOLERANCE and record.expected_mdr_fee < base_mdr_fee:
                return RootCauseClassification(
                    payment_id=pid,
                    case_type="missed_volume_tier",
                    is_anomaly=True,
                    description=f"Monthly GMV reached Rs {monthly_gmv} (qualifying for tier discount), but gateway billed at base rate ({base_rule.mdr_rate}%).",
                    evidence_summary=f"Monthly GMV: Rs {monthly_gmv} >= Rs 500,000 threshold. Applied MDR: Rs {record.actual_mdr_fee}, Expected Tier MDR: Rs {record.expected_mdr_fee}.",
                    suggested_action="Claim tier volume rebate and request gateway configuration update.",
                )

    # 5. Contract Version Violation Check
    # In Case 8: Transaction date falls in V2, but gateway billed at expired V1 rate
    # Look for expired contracts for this merchant
    expired_contracts = [
        c for c in contracts
        if c.merchant_id == record.merchant_id
        and c.effective_to is not None
        and c.effective_to < record.txn_date
    ]
    for exp_c in expired_contracts:
        exp_rules = [r for r in rules if r.contract_id == exp_c.contract_id and r.payment_method == record.payment_method and r.card_category == record.card_category]
        if exp_rules:
            exp_rule = exp_rules[0]
            exp_mdr_calc = (amt * exp_rule.mdr_rate / Decimal("100")).quantize(PRECISION, rounding=ROUND_HALF_UP)
            if abs(record.actual_mdr_fee - exp_mdr_calc) <= TOLERANCE and abs(record.actual_mdr_fee - record.expected_mdr_fee) > TOLERANCE:
                return RootCauseClassification(
                    payment_id=pid,
                    case_type="contract_version_violation",
                    is_anomaly=True,
                    description=f"Gateway applied superseded contract {exp_c.contract_id} (V{exp_c.version}, expired {exp_c.effective_to}) rate {exp_rule.mdr_rate}% instead of active contract rate.",
                    evidence_summary=f"Txn date {record.txn_date} governed by active contract. Gateway used expired rate from {exp_c.contract_id}.",
                    suggested_action="File claim for application of superseded contract terms.",
                )

    # 6. Wrong MDR (General Rate Mismatch)
    if abs(record.actual_mdr_fee - record.expected_mdr_fee) > TOLERANCE:
        return RootCauseClassification(
            payment_id=pid,
            case_type="wrong_mdr",
            is_anomaly=True,
            description=f"Gateway applied incorrect MDR fee of Rs {record.actual_mdr_fee} (expected Rs {record.expected_mdr_fee}).",
            evidence_summary=f"Contract rule {record.rule_id} specifies expected MDR Rs {record.expected_mdr_fee}; gateway billed Rs {record.actual_mdr_fee}.",
            suggested_action="Claim refund for MDR overbilling.",
        )

    # Fallback
    return RootCauseClassification(
        payment_id=pid,
        case_type="fee_variance_other",
        is_anomaly=True,
        description=f"Unclassified fee variance of Rs {var}.",
        evidence_summary=f"Expected total: Rs {record.expected_total_fee}, Actual total: Rs {record.actual_total_fee}.",
        suggested_action="Manual review required.",
    )


def classify_lifecycle_anomaly(payment: dict) -> RootCauseClassification:
    """
    Classify non-settled payment lifecycle cases (refunds, chargebacks, timing lags).
    """
    pid = payment["payment_id"]
    status = payment["status"]

    if status == "refunded":
        return RootCauseClassification(
            payment_id=pid,
            case_type="legitimate_refund",
            is_anomaly=False,
            description="Legitimate customer refund processed properly.",
            evidence_summary=f"Payment status is 'refunded'. Net settlement adjusted accordingly.",
            suggested_action="No claim needed (legitimate lifecycle event).",
        )
    elif status == "disputed":
        return RootCauseClassification(
            payment_id=pid,
            case_type="legitimate_chargeback",
            is_anomaly=False,
            description="Legitimate dispute/chargeback with contractually permitted dispute fee.",
            evidence_summary=f"Payment status is 'disputed'. Dispute fee permitted under contract terms.",
            suggested_action="No claim needed (valid contractual fee).",
        )
    elif status == "pending_settlement":
        return RootCauseClassification(
            payment_id=pid,
            case_type="timing_difference",
            is_anomaly=False,
            description="Payment settlement pending across period boundary (timing lag).",
            evidence_summary=f"Payment date {payment.get('txn_date')} pending settlement in subsequent cycle.",
            suggested_action="Monitor in subsequent reconciliation cycle.",
        )
    else:
        return RootCauseClassification(
            payment_id=pid,
            case_type="normal_settled",
            is_anomaly=False,
            description="Standard payment transaction.",
            evidence_summary=f"Status: {status}",
            suggested_action="No action required.",
        )
