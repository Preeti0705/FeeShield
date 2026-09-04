"""
src/evidence/builder.py
=======================
Generates audit-ready evidence claim packets for merchant recovery and PSP disputes.

Features:
  - Formats financial values to standard 2 decimal places for formal claims while preserving 6 d.p. in data.
  - Generates dispute letters with contract clauses, mathematical breakdown, and bank statements citation.
  - Exports claim items as structured dicts / CSV format.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from datetime import date
from typing import Optional, List, Dict

from src.audit.fee_variance import FeeVarianceRecord
from src.rootcause.classifier import RootCauseClassification
from src.reconciliation.bank_matching import BankGap


TWO_DP = Decimal("0.01")


def _f2(val: Decimal) -> str:
    """Format decimal to 2 decimal places for business display."""
    return f"{val.quantize(TWO_DP, rounding=ROUND_HALF_UP):,.2f}"


@dataclass(frozen=True)
class ClaimItem:
    """
    A single claim item ready for submission to payment gateway or bank.
    """
    claim_id: str
    item_type: str                  # 'FEE_OVERCHARGE' | 'BANK_POSTING_GAP'
    payment_id: str
    settlement_id: str
    merchant_id: str
    gateway_id: str
    txn_date: str
    amount_inr: str
    claim_amount_inr: str
    root_cause: str
    confidence_score: int
    rule_id: str
    evidence_text: str
    suggested_action: str

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "item_type": self.item_type,
            "payment_id": self.payment_id,
            "settlement_id": self.settlement_id,
            "merchant_id": self.merchant_id,
            "gateway_id": self.gateway_id,
            "txn_date": self.txn_date,
            "amount_inr": self.amount_inr,
            "claim_amount_inr": self.claim_amount_inr,
            "root_cause": self.root_cause,
            "confidence_score": self.confidence_score,
            "rule_id": self.rule_id,
            "evidence_text": self.evidence_text,
            "suggested_action": self.suggested_action,
        }


def build_fee_claim_item(
    claim_id: str,
    record: FeeVarianceRecord,
    classification: RootCauseClassification,
    confidence: int,
) -> ClaimItem:
    """
    Build a claim item from a fee variance finding.
    """
    evidence = (
        f"Contract Rule {record.rule_id}: Expected MDR fee Rs {_f2(record.expected_mdr_fee)}, "
        f"Expected Tax Rs {_f2(record.expected_tax)}, Expected Total Rs {_f2(record.expected_total_fee)}. "
        f"Gateway Actual Deducted: MDR Rs {_f2(record.actual_mdr_fee)}, Tax Rs {_f2(record.actual_tax)}, "
        f"Total Rs {_f2(record.actual_total_fee)}. Overcharge: Rs {_f2(record.fee_variance_inr)}. "
        f"Details: {classification.evidence_summary}"
    )

    return ClaimItem(
        claim_id=claim_id,
        item_type="FEE_OVERCHARGE",
        payment_id=record.payment_id,
        settlement_id=record.settlement_id,
        merchant_id=record.merchant_id,
        gateway_id=record.gateway_id,
        txn_date=str(record.txn_date),
        amount_inr=_f2(record.amount),
        claim_amount_inr=_f2(record.fee_variance_inr),
        root_cause=classification.case_type,
        confidence_score=confidence,
        rule_id=record.rule_id,
        evidence_text=evidence,
        suggested_action=classification.suggested_action,
    )


def build_bank_claim_item(
    claim_id: str,
    gap: BankGap,
    merchant_id: str = "MER001",
    gateway_id: str = "GW_ALPHA",
    confidence: int = 95,
) -> ClaimItem:
    """
    Build a claim item from a bank gap finding.
    """
    evidence = (
        f"Settlement {gap.settlement_id} (Date: {gap.settlement_date}): Net Settled Rs {_f2(gap.net_settled_amount)}. "
        f"Bank Posting (Date: {gap.posting_date or 'NONE'}): Net Posted Rs {_f2(gap.net_posted_amount)}. "
        f"Cash Shortfall / Impact: Rs {_f2(gap.cash_impact_inr)}. Delay: {gap.delay_days} days. Notes: {gap.notes}"
    )

    action = "File settlement disbursement inquiry with payment gateway operations."
    if gap.gap_type == "POSTING_REVERSED":
        action = "Escalate reversed settlement to bank and gateway partner for urgent re-credit."
    elif gap.gap_type == "SETTLEMENT_POSTED_LATE":
        action = "Issue SLA breach warning and request float interest credit."

    return ClaimItem(
        claim_id=claim_id,
        item_type="BANK_POSTING_GAP",
        payment_id=gap.payment_id,
        settlement_id=gap.settlement_id,
        merchant_id=merchant_id,
        gateway_id=gateway_id,
        txn_date=str(gap.settlement_date),
        amount_inr=_f2(gap.net_settled_amount),
        claim_amount_inr=_f2(gap.cash_impact_inr),
        root_cause=gap.gap_type,
        confidence_score=confidence,
        rule_id="SLA_COMPLIANCE",
        evidence_text=evidence,
        suggested_action=action,
    )
