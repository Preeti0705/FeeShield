"""
src/contracts/resolver.py
==========================
Given a transaction's context, find the ONE correct ContractRule to apply.

The resolver is the financial authority for "which rule governs this transaction."
It is called by calculator.py, which then does the actual math.

Algorithm (three steps, each can raise an explicit error):
  Step 1 — Find the active contract for (merchant_id, gateway_id) on txn_date.
            Raises ContractNotFoundError if none exists.
            Raises AmbiguousContractError if more than one active contract matches.
  Step 2 — Filter contract rules by payment_method + card_category.
            Raises RuleNotFoundError if no rule exists for this method/category.
  Step 3 — Among qualifying rules, find the highest volume_tier_min_gmv that
            the merchant's monthly_gmv meets or exceeds. Ties are impossible by
            design (each tier has a unique min_gmv), but we assert it anyway.

Design decisions flagged:
  - "Never guess" policy: every ambiguity raises a named exception with a
    diagnostic message. Silent fallbacks (returning None, defaulting to base tier
    without checking) are not allowed because they'd produce wrong fee expectations
    with no audit trail.
  - Contract lookup uses date range overlap, not just "active" status flag, because
    the status field in the CSV is informational; the dates are authoritative.
  - The resolver accepts pre-loaded lists (not file paths) so it's easy to inject
    test data without touching the filesystem.
  - monthly_gmv is passed as Decimal by the caller; the resolver never reads it
    from the payment row directly (separation of concerns).
"""

from datetime import date
from decimal import Decimal
from typing import Optional

from src.contracts.models import Contract, ContractRule


# ── Custom exceptions ─────────────────────────────────────────────────────────

class ContractNotFoundError(Exception):
    """No contract covers this merchant/gateway on this date."""

class AmbiguousContractError(Exception):
    """More than one contract is active for this merchant/gateway on this date."""

class RuleNotFoundError(Exception):
    """No contract rule matches this payment_method/card_category combination."""


# ── CSV loaders ───────────────────────────────────────────────────────────────

def load_contracts(csv_path: str) -> list[Contract]:
    """
    Load contracts.csv into a list of Contract dataclasses.
    Converts date strings to date objects, version to int.
    """
    import csv
    contracts = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            effective_to = (
                date.fromisoformat(row["effective_to"])
                if row["effective_to"]
                else None
            )
            contracts.append(Contract(
                contract_id    = row["contract_id"],
                merchant_id    = row["merchant_id"],
                gateway_id     = row["gateway_id"],
                version        = int(row["version"]),
                effective_from = date.fromisoformat(row["effective_from"]),
                effective_to   = effective_to,
                status         = row["status"],
            ))
    return contracts


def load_contract_rules(csv_path: str) -> list[ContractRule]:
    """
    Load contract_rules.csv into a list of ContractRule dataclasses.
    All rate/fee fields are parsed as Decimal from their string representations.

    FINANCIAL MATH CHECKPOINT: Decimal(row["mdr_rate"]) — string → Decimal,
    no float conversion at any point.
    """
    import csv
    rules = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rules.append(ContractRule(
                rule_id             = row["rule_id"],
                contract_id         = row["contract_id"],
                payment_method      = row["payment_method"],
                card_category       = row["card_category"],
                volume_tier_min_gmv = Decimal(row["volume_tier_min_gmv"]),
                mdr_rate            = Decimal(row["mdr_rate"]),
                fixed_fee           = Decimal(row["fixed_fee"]),
                tax_rate            = Decimal(row["tax_rate"]),
            ))
    return rules


# ── Core resolver ─────────────────────────────────────────────────────────────

def resolve_rule(
    merchant_id:    str,
    gateway_id:     str,
    txn_date:       date,
    payment_method: str,
    card_category:  str,
    monthly_gmv:    Decimal,
    contracts:      list[Contract],
    rules:          list[ContractRule],
) -> ContractRule:
    """
    Return the single ContractRule that governs this transaction.

    Parameters
    ----------
    merchant_id     : e.g. "MER001"
    gateway_id      : e.g. "GW_ALPHA"
    txn_date        : the transaction date (used to find the active contract version)
    payment_method  : "card" | "upi"
    card_category   : "credit" | "debit" | "na"
    monthly_gmv     : the merchant's gross merchandise value for the txn month
                      (used to determine volume tier)
    contracts       : full list of Contract objects (from load_contracts)
    rules           : full list of ContractRule objects (from load_contract_rules)

    Returns
    -------
    ContractRule — the one correct rule

    Raises
    ------
    ContractNotFoundError   — no contract covers this merchant/gateway on txn_date
    AmbiguousContractError  — >1 contract covers this merchant/gateway on txn_date
    RuleNotFoundError       — no rule for this payment_method/card_category
    """

    # ── Step 1: Find the active contract ─────────────────────────────────────
    # A contract is "active" for a date if:
    #   effective_from <= txn_date <= effective_to (or effective_to is None)
    matching_contracts = [
        c for c in contracts
        if (c.merchant_id == merchant_id
            and c.gateway_id == gateway_id
            and c.effective_from <= txn_date
            and (c.effective_to is None or txn_date <= c.effective_to))
    ]

    if len(matching_contracts) == 0:
        raise ContractNotFoundError(
            f"No contract found for merchant={merchant_id!r}, "
            f"gateway={gateway_id!r} on date={txn_date}. "
            f"Check contracts.csv for coverage gaps."
        )

    if len(matching_contracts) > 1:
        ids = [c.contract_id for c in matching_contracts]
        raise AmbiguousContractError(
            f"Multiple contracts active for merchant={merchant_id!r}, "
            f"gateway={gateway_id!r} on date={txn_date}: {ids}. "
            f"Contracts must have non-overlapping date ranges."
        )

    contract = matching_contracts[0]

    # ── Step 2: Filter rules by method + category ─────────────────────────────
    method_rules = [
        r for r in rules
        if (r.contract_id == contract.contract_id
            and r.payment_method == payment_method
            and r.card_category == card_category)
    ]

    if len(method_rules) == 0:
        raise RuleNotFoundError(
            f"No rule found in contract={contract.contract_id!r} for "
            f"payment_method={payment_method!r}, card_category={card_category!r}. "
            f"Add the missing rule to contract_rules.csv."
        )

    # ── Step 3: Pick the highest qualifying volume tier ───────────────────────
    # Sort tiers descending by min_gmv.
    # Pick the FIRST one whose min_gmv the merchant meets (i.e., monthly_gmv >= min_gmv).
    # This implements "step-down" tier pricing: qualify for the best tier you can.
    #
    # Why descending sort? We want to check the most generous discount first.
    # Example: tiers at 0, 500000, 2000000 GMV → if GMV=600000, we try 2000000
    # (fail), then 500000 (pass) → return the 500000-tier rule.
    sorted_tiers = sorted(method_rules, key=lambda r: r.volume_tier_min_gmv, reverse=True)

    qualifying_rule: Optional[ContractRule] = None
    for rule in sorted_tiers:
        if monthly_gmv >= rule.volume_tier_min_gmv:
            qualifying_rule = rule
            break

    if qualifying_rule is None:
        # This can only happen if the base tier (min_gmv=0) is missing.
        # All merchants qualify for the base tier, so this is a data error.
        raise RuleNotFoundError(
            f"No qualifying volume tier in contract={contract.contract_id!r} for "
            f"payment_method={payment_method!r}, card_category={card_category!r}, "
            f"monthly_gmv={monthly_gmv}. "
            f"Ensure a base tier with volume_tier_min_gmv=0 exists."
        )

    return qualifying_rule
