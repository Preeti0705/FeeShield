"""
src/fees/calculator.py
======================
Compute the expected fee for a transaction given a ContractRule.

This is the ONLY place in the entire codebase where fee math happens.
All other modules that need an expected fee call this function — they never
re-implement the formula themselves.

Formula:
    mdr_fee   = ROUND_HALF_UP(amount * mdr_rate / 100, 6 d.p.)
    fixed_fee = rule.fixed_fee  (already Decimal, no rounding needed)
    tax_fee   = ROUND_HALF_UP(mdr_fee * tax_rate / 100, 6 d.p.)
    total_fee = mdr_fee + fixed_fee + tax_fee
    net_amount = amount - total_fee

Rounding rule: ROUND_HALF_UP to 6 decimal places for all intermediate
computations. The 6-decimal precision avoids accumulated rounding error
across a batch of thousands of transactions. Display/claim amounts are
rounded to 2 d.p. by the evidence generator (Day 4), not here.

FINANCIAL MATH CHECKPOINT (confirmed):
  - `amount` parameter is accepted as Decimal (caller must not pass float).
  - All arithmetic is Decimal arithmetic.
  - ROUND_HALF_UP is applied at each multiplication step, not just at the end.
    This matches how banks typically compute: each component rounded separately,
    then summed — avoiding one large compound rounding error.

Why round each step rather than round only at the end?
  Gateway processors typically round each fee component individually on each
  transaction and then sum. If we round only at the end, we'd compute a
  theoretically purer number but one that diverges from what the gateway
  actually charges, generating false-positive variances. Component-level
  rounding matches reality.
"""

from decimal import Decimal, ROUND_HALF_UP

from src.contracts.models import ContractRule, FeeBreakdown

# Single rounding constant — referenced everywhere so it can't drift.
PRECISION = Decimal("0.000001")   # 6 decimal places


def compute_expected_fee(amount: Decimal, rule: ContractRule) -> FeeBreakdown:
    """
    Compute the fully expected fee breakdown for a transaction.

    Parameters
    ----------
    amount : Decimal — the transaction amount in rupees. MUST be Decimal.
    rule   : ContractRule — the resolved rule from resolver.resolve_rule().

    Returns
    -------
    FeeBreakdown with all components filled in.

    Raises
    ------
    TypeError  — if amount is a float (guard against accidental float usage).
    ValueError — if amount is negative (refund amounts should be handled by
                 the caller before passing here).
    """
    if isinstance(amount, float):
        raise TypeError(
            f"STOP: float {amount!r} passed to compute_expected_fee(). "
            f"Convert to Decimal first: Decimal(str({amount!r}))"
        )
    if amount < Decimal("0"):
        raise ValueError(
            f"Transaction amount must be non-negative. Got: {amount}. "
            f"If this is a refund, handle it in lifecycle.py before calling the fee calculator."
        )

    # ── MDR fee ───────────────────────────────────────────────────────────────
    # amount * (mdr_rate / 100), rounded to PRECISION
    mdr_fee = (amount * rule.mdr_rate / Decimal("100")).quantize(
        PRECISION, rounding=ROUND_HALF_UP
    )

    # ── Fixed fee ─────────────────────────────────────────────────────────────
    # Already a Decimal from the ContractRule; just ensure precision alignment.
    fixed_fee = rule.fixed_fee.quantize(PRECISION, rounding=ROUND_HALF_UP)

    # ── Tax fee ───────────────────────────────────────────────────────────────
    # IMPORTANT: Tax base is the MDR fee, NOT the transaction amount.
    # GST in India is charged on the service fee (MDR), not on the goods value.
    # A gateway charging GST on the gross amount is Case Type 3 (wrong_tax_base).
    tax_fee = (mdr_fee * rule.tax_rate / Decimal("100")).quantize(
        PRECISION, rounding=ROUND_HALF_UP
    )

    # ── Total and net ─────────────────────────────────────────────────────────
    total_fee  = mdr_fee + fixed_fee + tax_fee
    net_amount = (amount - total_fee).quantize(PRECISION, rounding=ROUND_HALF_UP)

    return FeeBreakdown(
        rule_id    = rule.rule_id,
        mdr_fee    = mdr_fee,
        fixed_fee  = fixed_fee,
        tax_fee    = tax_fee,
        total_fee  = total_fee,
        net_amount = net_amount,
    )
