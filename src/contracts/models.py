"""
src/contracts/models.py
=======================
Dataclass definitions for the contract domain.

Design decisions:
  - Using Python dataclasses (not dicts) so that field access is typed and
    typos like rule["mrd_rate"] raise AttributeError immediately rather than
    silently returning None.
  - All monetary/rate fields are stored as Decimal, never float.
    They are constructed from strings when loaded from CSV.
  - `effective_to` is Optional because a contract with no end date is common
    in practice (open-ended). We treat None as "forever valid".
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class Contract:
    """
    A merchant-gateway contract, valid between effective_from and effective_to.
    frozen=True means instances are immutable and hashable (safe to use as dict keys).
    """
    contract_id: str
    merchant_id: str
    gateway_id: str
    version: int
    effective_from: date
    effective_to: Optional[date]   # None = no expiry
    status: str                    # 'active' | 'superseded'


@dataclass(frozen=True)
class ContractRule:
    """
    A single fee rule within a contract, keyed by payment_method + card_category
    + volume tier. There can be multiple rules for the same method/category with
    different volume_tier_min_gmv values (the resolver picks the highest qualifying one).

    FINANCIAL MATH CHECKPOINT: mdr_rate, fixed_fee, tax_rate are ALL Decimal.
    They represent percentages: e.g., mdr_rate=Decimal("1.50") means 1.50%.
    """
    rule_id: str
    contract_id: str
    payment_method: str            # 'card' | 'upi'
    card_category: str             # 'credit' | 'debit' | 'na'
    volume_tier_min_gmv: Decimal   # 0 = base tier, no minimum
    mdr_rate: Decimal              # percentage, e.g. Decimal("1.50")
    fixed_fee: Decimal             # flat rupees per transaction
    tax_rate: Decimal              # percentage, e.g. Decimal("18.00")


@dataclass
class FeeBreakdown:
    """
    Result of calculator.compute_expected_fee().
    Stores every component so that variance.py can pinpoint exactly which
    component the gateway got wrong (MDR vs tax vs fixed fee).

    FINANCIAL MATH CHECKPOINT: all amounts are Decimal.
    """
    rule_id: str
    mdr_fee: Decimal       # MDR component = amount * mdr_rate / 100
    fixed_fee: Decimal     # flat fee component
    tax_fee: Decimal       # GST/tax = mdr_fee * tax_rate / 100
    total_fee: Decimal     # mdr_fee + fixed_fee + tax_fee
    net_amount: Decimal    # transaction_amount - total_fee
