"""
src/confidence/scorer.py
========================
Deterministic confidence scoring engine for fee leakage and bank gap findings.

Design Principles:
  - Financial auditing requires calibrated confidence scores based on objective mathematical proof.
  - Scores range from 0 to 100:
    * 100: Duplicate deduction with identical settlement amounts, or exact contract date violation.
    * 99: Exact tax base calculation match (GST on gross amount).
    * 98: Exact rate match to superseded contract version or missed GMV volume tier.
    * 95: Standard rate mismatch with valid active contract rule.
    * 85-90: Bank posting delay / mismatch with standard SLA.
    * 0: No variance or legitimate operational lifecycle event.
"""

from decimal import Decimal
from typing import Optional


def compute_confidence_score(case_type: str, variance_amount: Decimal = Decimal("0")) -> int:
    """
    Return a calibrated confidence score (0-100) for a detected root cause category.
    """
    case_scores = {
        # Fee-side anomalies
        "duplicate_fee": 100,
        "wrong_tax_base": 99,
        "contract_version_violation": 98,
        "missed_volume_tier": 98,
        "wrong_mdr": 95,
        "fee_variance_other": 85,
        
        # Bank-side anomalies
        "SETTLEMENT_NOT_POSTED": 99,
        "POSTING_REVERSED": 99,
        "POSTED_AMOUNT_MISMATCH": 95,
        "SETTLEMENT_POSTED_LATE": 90,
        "HOLD_PLACED_THEN_CLEARED": 40,  # low confidence of anomaly (monitoring only)
        
        # Legitimate lifecycle states (not anomalies)
        "legitimate_refund": 0,
        "legitimate_chargeback": 0,
        "timing_difference": 0,
        "no_variance": 0,
        "normal_settled": 0,
    }
    
    return case_scores.get(case_type, 75)
