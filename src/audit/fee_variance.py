"""
src/audit/fee_variance.py
=========================
Calculates variance between expected contractual fees and actual settlement fee deductions.

Reconciliation Flow:
  1. For each payment, determine expected fee via resolver + calculator.
  2. Match payment against settlements.csv records.
  3. Detect:
     - Single settlement fee discrepancy (actual_total_fee - expected_total_fee)
     - Duplicate settlement deductions (e.g. Case 4: PAY004 has multiple settlement records)
     - Non-settled states (refunded, disputed, pending)
  4. Emit FeeVarianceRecord for every evaluated settlement.

Financial Math Rules:
  - Strict Decimal arithmetic, ROUND_HALF_UP, 6 d.p.
  - Tolerance: Decimal("0.01"). Variance <= 0.01 is considered rounding noise (has_variance=False).
"""

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from src.contracts.models import Contract, ContractRule, FeeBreakdown
from src.contracts.resolver import (
    resolve_rule,
    ContractNotFoundError,
    AmbiguousContractError,
    RuleNotFoundError,
)
from src.fees.calculator import compute_expected_fee


PRECISION = Decimal("0.000001")
VARIANCE_TOLERANCE = Decimal("0.01")


def _d(val: str | int | Decimal) -> Decimal:
    if isinstance(val, float):
        raise TypeError(f"STOP: float {val!r} passed to _d(). Use string/Decimal.")
    return Decimal(str(val))


@dataclass(frozen=True)
class FeeVarianceRecord:
    """
    Detailed audit result for a payment-settlement pair.
    """
    payment_id: str
    settlement_id: str
    merchant_id: str
    gateway_id: str
    txn_date: date
    amount: Decimal
    payment_method: str
    card_category: str
    status: str
    rule_id: str
    expected_mdr_fee: Decimal
    expected_tax: Decimal
    expected_fixed_fee: Decimal
    expected_total_fee: Decimal
    actual_mdr_fee: Decimal
    actual_tax: Decimal
    actual_fixed_fee: Decimal
    actual_total_fee: Decimal
    fee_variance_inr: Decimal       # actual_total_fee - expected_total_fee
    has_variance: bool              # abs(fee_variance_inr) > VARIANCE_TOLERANCE
    is_duplicate_settlement: bool   # True if this is an extra deduction for the same payment
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "payment_id": self.payment_id,
            "settlement_id": self.settlement_id,
            "merchant_id": self.merchant_id,
            "gateway_id": self.gateway_id,
            "txn_date": str(self.txn_date),
            "amount": str(self.amount),
            "payment_method": self.payment_method,
            "card_category": self.card_category,
            "status": self.status,
            "rule_id": self.rule_id,
            "expected_mdr_fee": str(self.expected_mdr_fee),
            "expected_tax": str(self.expected_tax),
            "expected_fixed_fee": str(self.expected_fixed_fee),
            "expected_total_fee": str(self.expected_total_fee),
            "actual_mdr_fee": str(self.actual_mdr_fee),
            "actual_tax": str(self.actual_tax),
            "actual_fixed_fee": str(self.actual_fixed_fee),
            "actual_total_fee": str(self.actual_total_fee),
            "fee_variance_inr": str(self.fee_variance_inr),
            "has_variance": self.has_variance,
            "is_duplicate_settlement": self.is_duplicate_settlement,
            "notes": self.notes,
        }


def audit_fee_variances(
    payments: list[dict],
    settlements: list[dict],
    contracts: list[Contract],
    rules: list[ContractRule],
    variance_tolerance: Decimal = VARIANCE_TOLERANCE,
) -> list[FeeVarianceRecord]:
    """
    Run the fee variance audit across all payment and settlement records.

    Parameters
    ----------
    payments: list of payment dicts (from payments.csv)
    settlements: list of settlement dicts (from settlements.csv)
    contracts: list of Contract domain models
    rules: list of ContractRule domain models
    variance_tolerance: threshold in INR to consider variance actionable

    Returns
    -------
    List of FeeVarianceRecord for each settlement examined.
    """
    # Group settlements by payment_id
    settlements_by_pay: dict[str, list[dict]] = {}
    for s in settlements:
        pid = s["payment_id"]
        settlements_by_pay.setdefault(pid, []).append(s)

    records: list[FeeVarianceRecord] = []

    for pay in payments:
        pid = pay["payment_id"]
        mid = pay["merchant_id"]
        gid = pay["gateway_id"]
        txn_dt = date.fromisoformat(pay["txn_date"]) if isinstance(pay["txn_date"], str) else pay["txn_date"]
        amt = _d(pay["amount"])
        pm = pay["payment_method"]
        cc = pay["card_category"]
        status = pay["status"]
        monthly_gmv = _d(pay.get("monthly_gmv", "0"))
        pay_notes = pay.get("notes", "")

        # Skip payments that never should settle
        if status in ("refunded", "disputed", "pending_settlement"):
            continue

        sel_rows = settlements_by_pay.get(pid, [])
        if not sel_rows:
            continue

        # Resolve expected rule and compute fee
        rule = resolve_rule(
            contracts=contracts,
            rules=rules,
            merchant_id=mid,
            gateway_id=gid,
            txn_date=txn_dt,
            payment_method=pm,
            card_category=cc,
            monthly_gmv=monthly_gmv,
        )
        expected_breakdown = compute_expected_fee(amt, rule)

        # Evaluate each settlement row for this payment
        for idx, sel in enumerate(sel_rows):
            is_dup = (idx > 0) or sel["settlement_id"].endswith("_DUP") or ("DUPLICATE" in sel.get("notes", ""))
            
            act_mdr = _d(sel["actual_mdr_fee"])
            act_tax = _d(sel["actual_tax"])
            act_fix = _d(sel["actual_fixed_fee"])
            act_tot = (act_mdr + act_tax + act_fix).quantize(PRECISION, rounding=ROUND_HALF_UP)

            if is_dup:
                # In a duplicate deduction row, the expected fee is 0 because fee was already deducted
                exp_mdr = Decimal("0.000000")
                exp_tax = Decimal("0.000000")
                exp_fix = Decimal("0.000000")
                exp_tot = Decimal("0.000000")
                var_inr = act_tot
            else:
                exp_mdr = expected_breakdown.mdr_fee
                exp_tax = expected_breakdown.tax_fee
                exp_fix = expected_breakdown.fixed_fee
                exp_tot = expected_breakdown.total_fee
                var_inr = (act_tot - exp_tot).quantize(PRECISION, rounding=ROUND_HALF_UP)

            has_var = abs(var_inr) > variance_tolerance

            combined_notes = sel.get("notes", "")
            if pay_notes and pay_notes not in combined_notes:
                combined_notes = f"{pay_notes} | {combined_notes}" if combined_notes else pay_notes

            record = FeeVarianceRecord(
                payment_id=pid,
                settlement_id=sel["settlement_id"],
                merchant_id=mid,
                gateway_id=gid,
                txn_date=txn_dt,
                amount=amt,
                payment_method=pm,
                card_category=cc,
                status=status,
                rule_id=rule.rule_id,
                expected_mdr_fee=exp_mdr,
                expected_tax=exp_tax,
                expected_fixed_fee=exp_fix,
                expected_total_fee=exp_tot,
                actual_mdr_fee=act_mdr,
                actual_tax=act_tax,
                actual_fixed_fee=act_fix,
                actual_total_fee=act_tot,
                fee_variance_inr=var_inr,
                has_variance=has_var,
                is_duplicate_settlement=is_dup,
                notes=combined_notes,
            )
            records.append(record)

    return records
