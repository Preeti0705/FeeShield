"""
tests/test_contracts.py
=======================
Unit tests for src/contracts/resolver.py.

EVERY expected value in this file was computed by hand BEFORE writing the test.
Hand calculations are shown in comments so they can be verified without running code.

Test strategy:
  - We inject Contract and ContractRule objects directly (no CSV files) so
    tests are hermetic and fast. CSV loading is tested separately (end-to-end).
  - Each test covers exactly one behaviour. Naming: test_<what>_<condition>.
"""

import pytest
from datetime import date
from decimal import Decimal

from src.contracts.models import Contract, ContractRule
from src.contracts.resolver import (
    resolve_rule,
    ContractNotFoundError,
    AmbiguousContractError,
    RuleNotFoundError,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

# A simple two-contract setup: V1 expired Jan-31, V2 starts Feb-1.
# This mirrors the MER001 setup in generate_data.py.

CONTRACT_V1 = Contract(
    contract_id="CON001", merchant_id="MER001", gateway_id="GW_ALPHA",
    version=1,
    effective_from=date(2024, 1, 1),
    effective_to=date(2024, 1, 31),
    status="superseded",
)

CONTRACT_V2 = Contract(
    contract_id="CON002", merchant_id="MER001", gateway_id="GW_ALPHA",
    version=2,
    effective_from=date(2024, 2, 1),
    effective_to=date(2024, 12, 31),
    status="active",
)

ALL_CONTRACTS = [CONTRACT_V1, CONTRACT_V2]

# Rules for CON002: credit card with three volume tiers, plus debit + UPI.
RULE_CREDIT_BASE  = ContractRule("RUL004", "CON002", "card", "credit",
                                  Decimal("0"),       Decimal("1.50"), Decimal("0"), Decimal("18.00"))
RULE_CREDIT_TIER1 = ContractRule("RUL005", "CON002", "card", "credit",
                                  Decimal("500000"),  Decimal("1.20"), Decimal("0"), Decimal("18.00"))
RULE_CREDIT_TIER2 = ContractRule("RUL006", "CON002", "card", "credit",
                                  Decimal("2000000"), Decimal("1.00"), Decimal("0"), Decimal("18.00"))
RULE_DEBIT        = ContractRule("RUL007", "CON002", "card", "debit",
                                  Decimal("0"),       Decimal("0.75"), Decimal("0"), Decimal("18.00"))
RULE_UPI          = ContractRule("RUL008", "CON002", "upi",  "na",
                                  Decimal("0"),       Decimal("0.00"), Decimal("0"), Decimal("0.00"))

# Rules for CON001: only base tier, higher rates
RULE_V1_CREDIT = ContractRule("RUL001", "CON001", "card", "credit",
                               Decimal("0"), Decimal("1.80"), Decimal("0"), Decimal("18.00"))
RULE_V1_DEBIT  = ContractRule("RUL002", "CON001", "card", "debit",
                               Decimal("0"), Decimal("0.90"), Decimal("0"), Decimal("18.00"))

ALL_RULES = [
    RULE_V1_CREDIT, RULE_V1_DEBIT,
    RULE_CREDIT_BASE, RULE_CREDIT_TIER1, RULE_CREDIT_TIER2,
    RULE_DEBIT, RULE_UPI,
]


# ── Tests: contract selection by date ────────────────────────────────────────

class TestContractDateSelection:

    def test_jan_date_selects_v1_contract(self):
        """A Jan-15 transaction must resolve to CON001, not CON002."""
        rule = resolve_rule(
            "MER001", "GW_ALPHA", date(2024, 1, 15),
            "card", "credit", Decimal("0"),
            ALL_CONTRACTS, ALL_RULES,
        )
        assert rule.contract_id == "CON001", (
            "Jan-15 is within CON001 (Jan-1 to Jan-31). Got: " + rule.contract_id
        )
        assert rule.rule_id == "RUL001"

    def test_feb_date_selects_v2_contract(self):
        """A Feb-10 transaction must resolve to CON002."""
        rule = resolve_rule(
            "MER001", "GW_ALPHA", date(2024, 2, 10),
            "card", "credit", Decimal("0"),
            ALL_CONTRACTS, ALL_RULES,
        )
        assert rule.contract_id == "CON002"

    def test_jan31_boundary_selects_v1(self):
        """Boundary: Jan-31 is the LAST day of CON001 — must resolve to V1."""
        rule = resolve_rule(
            "MER001", "GW_ALPHA", date(2024, 1, 31),
            "card", "credit", Decimal("0"),
            ALL_CONTRACTS, ALL_RULES,
        )
        assert rule.contract_id == "CON001"

    def test_feb1_boundary_selects_v2(self):
        """Boundary: Feb-1 is the FIRST day of CON002 — must resolve to V2."""
        rule = resolve_rule(
            "MER001", "GW_ALPHA", date(2024, 2, 1),
            "card", "credit", Decimal("0"),
            ALL_CONTRACTS, ALL_RULES,
        )
        assert rule.contract_id == "CON002"

    def test_no_contract_raises_error(self):
        """A date with no covering contract must raise ContractNotFoundError, not return None."""
        with pytest.raises(ContractNotFoundError) as exc_info:
            resolve_rule(
                "MER001", "GW_ALPHA", date(2025, 6, 1),  # far in the future, no contract
                "card", "credit", Decimal("0"),
                ALL_CONTRACTS, ALL_RULES,
            )
        assert "MER001" in str(exc_info.value)

    def test_ambiguous_contracts_raises_error(self):
        """Overlapping contracts (bad data) must raise AmbiguousContractError."""
        # Create a contract that overlaps with V1
        duplicate = Contract(
            contract_id="CON_DUP", merchant_id="MER001", gateway_id="GW_ALPHA",
            version=99,
            effective_from=date(2024, 1, 1),
            effective_to=date(2024, 1, 31),
            status="active",
        )
        dup_rule = ContractRule("RUL_DUP", "CON_DUP", "card", "credit",
                                Decimal("0"), Decimal("2.00"), Decimal("0"), Decimal("18.00"))
        with pytest.raises(AmbiguousContractError):
            resolve_rule(
                "MER001", "GW_ALPHA", date(2024, 1, 15),
                "card", "credit", Decimal("0"),
                ALL_CONTRACTS + [duplicate],
                ALL_RULES + [dup_rule],
            )

    def test_wrong_merchant_raises_error(self):
        """A merchant with no contract raises ContractNotFoundError."""
        with pytest.raises(ContractNotFoundError):
            resolve_rule(
                "MER999", "GW_ALPHA", date(2024, 2, 1),
                "card", "credit", Decimal("0"),
                ALL_CONTRACTS, ALL_RULES,
            )


# ── Tests: volume tier selection ──────────────────────────────────────────────

class TestVolumeTierSelection:
    """
    Hand-calculated tier expectations for CON002 credit card:
      Tier 0:       GMV < 500000   → mdr_rate = 1.50%  (RUL004)
      Tier 500000:  GMV >= 500000  → mdr_rate = 1.20%  (RUL005)
      Tier 2000000: GMV >= 2000000 → mdr_rate = 1.00%  (RUL006)
    """

    def test_gmv_zero_uses_base_tier(self):
        """GMV=0 → base tier (1.50%)."""
        rule = resolve_rule(
            "MER001", "GW_ALPHA", date(2024, 2, 10),
            "card", "credit", Decimal("0"),
            ALL_CONTRACTS, ALL_RULES,
        )
        assert rule.rule_id == "RUL004"
        assert rule.mdr_rate == Decimal("1.50")

    def test_gmv_below_tier1_uses_base(self):
        """GMV=499999 → still base tier (1.50%)."""
        rule = resolve_rule(
            "MER001", "GW_ALPHA", date(2024, 2, 10),
            "card", "credit", Decimal("499999"),
            ALL_CONTRACTS, ALL_RULES,
        )
        assert rule.rule_id == "RUL004"
        assert rule.mdr_rate == Decimal("1.50")

    def test_gmv_exactly_at_tier1_boundary(self):
        """GMV=500000 → exactly at tier 1 boundary → must qualify (1.20%)."""
        rule = resolve_rule(
            "MER001", "GW_ALPHA", date(2024, 2, 10),
            "card", "credit", Decimal("500000"),
            ALL_CONTRACTS, ALL_RULES,
        )
        assert rule.rule_id == "RUL005"
        assert rule.mdr_rate == Decimal("1.20")

    def test_gmv_between_tiers_uses_tier1(self):
        """GMV=550000 → above tier 1, below tier 2 → 1.20%."""
        rule = resolve_rule(
            "MER001", "GW_ALPHA", date(2024, 2, 10),
            "card", "credit", Decimal("550000"),
            ALL_CONTRACTS, ALL_RULES,
        )
        assert rule.rule_id == "RUL005"

    def test_gmv_at_tier2_boundary(self):
        """GMV=2000000 → exactly at tier 2 boundary → 1.00%."""
        rule = resolve_rule(
            "MER001", "GW_ALPHA", date(2024, 2, 10),
            "card", "credit", Decimal("2000000"),
            ALL_CONTRACTS, ALL_RULES,
        )
        assert rule.rule_id == "RUL006"
        assert rule.mdr_rate == Decimal("1.00")

    def test_gmv_above_tier2(self):
        """GMV=5000000 → above all tiers → best tier (1.00%)."""
        rule = resolve_rule(
            "MER001", "GW_ALPHA", date(2024, 2, 10),
            "card", "credit", Decimal("5000000"),
            ALL_CONTRACTS, ALL_RULES,
        )
        assert rule.rule_id == "RUL006"


# ── Tests: payment method / card category ────────────────────────────────────

class TestPaymentMethodCategory:

    def test_debit_card_resolves_correctly(self):
        """Debit card uses RUL007 (0.75%) in CON002."""
        rule = resolve_rule(
            "MER001", "GW_ALPHA", date(2024, 2, 10),
            "card", "debit", Decimal("0"),
            ALL_CONTRACTS, ALL_RULES,
        )
        assert rule.rule_id == "RUL007"
        assert rule.mdr_rate == Decimal("0.75")

    def test_upi_resolves_to_zero_mdr(self):
        """UPI in CON002 has 0% MDR and 0% tax."""
        rule = resolve_rule(
            "MER001", "GW_ALPHA", date(2024, 2, 10),
            "upi", "na", Decimal("0"),
            ALL_CONTRACTS, ALL_RULES,
        )
        assert rule.rule_id == "RUL008"
        assert rule.mdr_rate == Decimal("0.00")
        assert rule.tax_rate == Decimal("0.00")

    def test_missing_payment_method_raises_error(self):
        """A payment method with no rule must raise RuleNotFoundError."""
        with pytest.raises(RuleNotFoundError) as exc_info:
            resolve_rule(
                "MER001", "GW_ALPHA", date(2024, 2, 10),
                "netbanking", "na", Decimal("0"),  # no rule for this
                ALL_CONTRACTS, ALL_RULES,
            )
        assert "netbanking" in str(exc_info.value)

    def test_missing_base_tier_raises_error(self):
        """
        If ALL rules for a method have min_gmv > 0 (no base tier),
        a merchant with GMV=0 must get RuleNotFoundError, not a silent default.
        """
        # Remove the base tier rule, keep only the higher tiers
        rules_no_base = [r for r in ALL_RULES if r.rule_id != "RUL004"]
        with pytest.raises(RuleNotFoundError) as exc_info:
            resolve_rule(
                "MER001", "GW_ALPHA", date(2024, 2, 10),
                "card", "credit", Decimal("0"),
                ALL_CONTRACTS, rules_no_base,
            )
        assert "base tier" in str(exc_info.value).lower() or "qualifying" in str(exc_info.value).lower()


# ── Tests: CSV loading ────────────────────────────────────────────────────────

class TestCSVLoading:
    """Smoke test: can we load the actual generated CSVs and resolve a real rule?"""

    def test_load_and_resolve_real_data(self):
        """
        Load contracts.csv + contract_rules.csv and resolve a known rule.
        PAY001 is MER001, GW_ALPHA, Feb-10, credit card, GMV=0
        → should resolve to RUL004 (CON002 base credit tier, 1.50%)
        """
        from src.contracts.resolver import load_contracts, load_contract_rules
        contracts = load_contracts("data/contracts.csv")
        rules     = load_contract_rules("data/contract_rules.csv")

        resolved = resolve_rule(
            "MER001", "GW_ALPHA", date(2024, 2, 10),
            "card", "credit", Decimal("0"),
            contracts, rules,
        )
        assert resolved.rule_id == "RUL004"
        assert resolved.mdr_rate == Decimal("1.50")
        assert resolved.tax_rate == Decimal("18.00")

    def test_load_rates_are_decimal_not_float(self):
        """
        Confirm that loaded rates are Decimal instances, not float.
        This guards against anyone changing the loader to use float().
        """
        from src.contracts.resolver import load_contract_rules
        rules = load_contract_rules("data/contract_rules.csv")
        for rule in rules:
            assert isinstance(rule.mdr_rate, Decimal), (
                f"mdr_rate in {rule.rule_id} is {type(rule.mdr_rate)}, not Decimal"
            )
            assert isinstance(rule.tax_rate, Decimal), (
                f"tax_rate in {rule.rule_id} is {type(rule.tax_rate)}, not Decimal"
            )
