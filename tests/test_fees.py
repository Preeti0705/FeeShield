"""
tests/test_fees.py
==================
Unit tests for src/fees/calculator.py.

ALL expected values were computed by hand before writing these tests.
The hand calculations are shown in comments below each test so you
can verify them with a calculator independently.

Test strategy:
  - Test every meaningful combination: normal rate, zero rate (UPI),
    fixed fee, rounding edge cases, negative amount guard, float guard.
  - Use hard-coded Decimal expected values, not re-running the formula.
    (If you test by running the formula to compare with itself, you'll
    never catch a wrong formula.)
"""

import pytest
from decimal import Decimal

from src.contracts.models import ContractRule, FeeBreakdown
from src.fees.calculator import compute_expected_fee, PRECISION


# ── Shared test fixtures (inline, not pytest fixtures, for clarity) ───────────

def make_rule(rule_id="TEST", contract_id="CON_TEST",
              payment_method="card", card_category="credit",
              volume_tier_min_gmv="0",
              mdr_rate="1.50", fixed_fee="0", tax_rate="18.00") -> ContractRule:
    """Helper: build a ContractRule from string arguments → Decimal."""
    return ContractRule(
        rule_id=rule_id, contract_id=contract_id,
        payment_method=payment_method, card_category=card_category,
        volume_tier_min_gmv=Decimal(volume_tier_min_gmv),
        mdr_rate=Decimal(mdr_rate),
        fixed_fee=Decimal(fixed_fee),
        tax_rate=Decimal(tax_rate),
    )


# ── Tests: basic fee calculation ──────────────────────────────────────────────

class TestBasicFeeCalculation:
    """
    Hand-calculation reference table:

    Test A — credit card 1.50% MDR, ₹10,000
      mdr_fee  = 10000 * 1.50 / 100           = 150.000000
      fixed    = 0.000000
      tax_fee  = 150 * 18.00 / 100            = 27.000000
      total    = 150 + 0 + 27                  = 177.000000
      net      = 10000 - 177                   = 9823.000000

    Test B — debit card 0.75% MDR, ₹8,000
      mdr_fee  = 8000 * 0.75 / 100            = 60.000000
      fixed    = 0.000000
      tax_fee  = 60 * 18.00 / 100             = 10.800000
      total    = 60 + 0 + 10.80               = 70.800000
      net      = 8000 - 70.80                  = 7929.200000

    Test C — UPI 0% MDR, ₹5,000
      mdr_fee  = 0.000000
      fixed    = 0.000000
      tax_fee  = 0.000000
      total    = 0.000000
      net      = 5000.000000
    """

    def test_credit_card_1_50_pct_10000(self):
        """Test A: standard credit card transaction."""
        rule   = make_rule(mdr_rate="1.50", tax_rate="18.00")
        result = compute_expected_fee(Decimal("10000"), rule)

        assert result.mdr_fee   == Decimal("150.000000")
        assert result.fixed_fee == Decimal("0.000000")
        assert result.tax_fee   == Decimal("27.000000")
        assert result.total_fee == Decimal("177.000000")
        assert result.net_amount == Decimal("9823.000000")

    def test_debit_card_0_75_pct_8000(self):
        """Test B: debit card — matches PAY004 amount for cross-reference with GT004."""
        rule   = make_rule(mdr_rate="0.75", tax_rate="18.00", card_category="debit")
        result = compute_expected_fee(Decimal("8000"), rule)

        assert result.mdr_fee    == Decimal("60.000000")
        assert result.tax_fee    == Decimal("10.800000")
        assert result.total_fee  == Decimal("70.800000")
        assert result.net_amount == Decimal("7929.200000")

    def test_upi_zero_mdr(self):
        """Test C: UPI should produce zero fees."""
        rule   = make_rule(mdr_rate="0.00", tax_rate="0.00",
                           payment_method="upi", card_category="na")
        result = compute_expected_fee(Decimal("5000"), rule)

        assert result.mdr_fee    == Decimal("0.000000")
        assert result.tax_fee    == Decimal("0.000000")
        assert result.total_fee  == Decimal("0.000000")
        assert result.net_amount == Decimal("5000.000000")


# ── Tests: fixed fee component ────────────────────────────────────────────────

class TestFixedFee:
    """
    Hand-calculation:
    Amount=1000, mdr_rate=1.60%, fixed_fee=5.00, tax_rate=18%
      mdr_fee  = 1000 * 1.60 / 100   = 16.000000
      fixed    = 5.000000
      tax_fee  = 16 * 18 / 100       = 2.880000
      total    = 16 + 5 + 2.88       = 23.880000
      net      = 1000 - 23.88         = 976.120000
    """

    def test_fixed_fee_added_to_total(self):
        rule   = make_rule(mdr_rate="1.60", fixed_fee="5.00", tax_rate="18.00")
        result = compute_expected_fee(Decimal("1000"), rule)

        assert result.fixed_fee  == Decimal("5.000000")
        assert result.total_fee  == Decimal("23.880000")
        assert result.net_amount == Decimal("976.120000")

    def test_fixed_fee_only_no_mdr(self):
        """
        Edge: mdr_rate=0 but fixed_fee=10. Tax on zero MDR = zero.
        total = 0 + 10 + 0 = 10.000000
        """
        rule   = make_rule(mdr_rate="0.00", fixed_fee="10.00", tax_rate="18.00")
        result = compute_expected_fee(Decimal("500"), rule)

        assert result.mdr_fee    == Decimal("0.000000")
        assert result.tax_fee    == Decimal("0.000000")   # 0% of 0 = 0
        assert result.fixed_fee  == Decimal("10.000000")
        assert result.total_fee  == Decimal("10.000000")


# ── Tests: rounding edge cases ────────────────────────────────────────────────

class TestRounding:
    """
    Rounding is the trickiest part. Hand-calculated to 8 decimal places:

    Case R1 — 1.80% MDR on ₹9,000 (PAY008 / GT008 reference)
      mdr_fee  = 9000 * 1.80 / 100  = 162.000000  (exact, no rounding needed)
      tax_fee  = 162 * 18 / 100     = 29.160000   (exact)
      total    = 191.160000

    Case R2 — 1.20% MDR on ₹15,000 (PAY002 / GT002 reference, tier rate)
      mdr_fee  = 15000 * 1.20 / 100 = 180.000000  (exact)
      tax_fee  = 180 * 18 / 100     = 32.400000   (exact)

    Case R3 — Rounding genuinely needed: 1.80% on ₹333.33
      mdr_fee  = 333.33 * 1.80 / 100 = 5.99994
                 → ROUND_HALF_UP to 6 d.p. → 5.999940
      tax_fee  = 5.999940 * 18 / 100  = 1.0799892
                 → ROUND_HALF_UP to 6 d.p. → 1.079989
      total    = 5.999940 + 0 + 1.079989 = 7.079929
      net      = 333.330000 - 7.079929 = 326.250071
    """

    def test_pay001_reference_1_80_pct_10000(self):
        """Matches PAY001 gateway overcharge rate — 1.80% on 10000."""
        rule   = make_rule(mdr_rate="1.80", tax_rate="18.00")
        result = compute_expected_fee(Decimal("10000"), rule)

        assert result.mdr_fee   == Decimal("180.000000")
        assert result.tax_fee   == Decimal("32.400000")
        assert result.total_fee == Decimal("212.400000")

    def test_pay008_reference_1_80_pct_9000(self):
        """Matches the stale-rate fee used in PAY008/GT008."""
        rule   = make_rule(mdr_rate="1.80", tax_rate="18.00")
        result = compute_expected_fee(Decimal("9000"), rule)

        assert result.mdr_fee   == Decimal("162.000000")
        assert result.tax_fee   == Decimal("29.160000")
        assert result.total_fee == Decimal("191.160000")

    def test_rounding_non_terminating_decimal(self):
        """Case R3: ensure ROUND_HALF_UP applied correctly on non-exact values."""
        rule   = make_rule(mdr_rate="1.80", tax_rate="18.00")
        result = compute_expected_fee(Decimal("333.33"), rule)

        # mdr_fee = 333.33 * 0.018 = 5.99994 → stored as 5.999940
        assert result.mdr_fee == Decimal("5.999940")
        # tax_fee = 5.999940 * 0.18 = 1.07998920 → ROUND_HALF_UP → 1.079989
        assert result.tax_fee == Decimal("1.079989")
        assert result.total_fee == Decimal("7.079929")

    def test_result_precision_is_six_decimal_places(self):
        """All money fields in FeeBreakdown must be quantized to 6 d.p."""
        rule   = make_rule(mdr_rate="1.50", tax_rate="18.00")
        result = compute_expected_fee(Decimal("100"), rule)
        # Check exponent: Decimal("1.500000") has exponent -6
        for field_name in ("mdr_fee", "fixed_fee", "tax_fee", "total_fee", "net_amount"):
            val = getattr(result, field_name)
            assert val == val.quantize(PRECISION), (
                f"{field_name}={val} is not at 6 d.p. precision"
            )


# ── Tests: tax base correctness ───────────────────────────────────────────────

class TestTaxBase:
    """
    Confirm that tax is always computed on MDR fee, NOT on transaction amount.
    This is the correct behaviour; computing on gross is Case Type 3 (wrong_tax_base).

    Hand-calculation:
    Amount=20000, mdr_rate=1.60%, tax_rate=18%
      correct:  tax = (20000 * 0.016) * 0.18 = 320 * 0.18 = 57.600000
      WRONG:    tax = 20000 * 0.18            = 3600.000000  (case type 3)
    """

    def test_tax_is_on_mdr_fee_not_gross_amount(self):
        """Correct behaviour: tax base = MDR fee."""
        rule   = make_rule(mdr_rate="1.60", tax_rate="18.00")
        result = compute_expected_fee(Decimal("20000"), rule)

        # mdr_fee = 20000 * 1.60 / 100 = 320.000000
        assert result.mdr_fee == Decimal("320.000000")
        # tax = 320 * 18 / 100 = 57.600000  (NOT 3600)
        assert result.tax_fee == Decimal("57.600000")
        assert result.tax_fee != Decimal("3600.000000")  # explicit NOT the wrong value


# ── Tests: input validation ───────────────────────────────────────────────────

class TestInputValidation:

    def test_float_amount_raises_type_error(self):
        """FINANCIAL MATH CHECKPOINT: passing a float must raise TypeError."""
        rule = make_rule()
        with pytest.raises(TypeError) as exc_info:
            compute_expected_fee(1000.0, rule)   # float, not Decimal
        assert "float" in str(exc_info.value).lower()

    def test_negative_amount_raises_value_error(self):
        """Negative amounts (refunds) must be rejected — handle in lifecycle.py."""
        rule = make_rule()
        with pytest.raises(ValueError) as exc_info:
            compute_expected_fee(Decimal("-500"), rule)
        assert "negative" in str(exc_info.value).lower() or "non-negative" in str(exc_info.value).lower()

    def test_zero_amount_returns_zero_fees(self):
        """Zero-amount transactions are valid (some test/auth transactions)."""
        rule   = make_rule()
        result = compute_expected_fee(Decimal("0"), rule)
        assert result.total_fee  == Decimal("0.000000")
        assert result.net_amount == Decimal("0.000000")

    def test_returns_fee_breakdown_type(self):
        """Result must be a FeeBreakdown instance, not a dict or tuple."""
        rule   = make_rule()
        result = compute_expected_fee(Decimal("1000"), rule)
        assert isinstance(result, FeeBreakdown)

    def test_rule_id_propagated_to_result(self):
        """The rule_id used in the calculation must be recorded in the result."""
        rule   = make_rule(rule_id="RUL_TEST_42")
        result = compute_expected_fee(Decimal("1000"), rule)
        assert result.rule_id == "RUL_TEST_42"


# ── Tests: leakage delta validation (GT cross-check) ─────────────────────────

class TestLeakageDeltaCrossCheck:
    """
    Compute the leakage for each ground-truth case from first principles,
    using the calculator twice (actual rate vs correct rate).

    GT001: wrong_mdr — PAY001 ₹10,000
      correct: 1.50% MDR + 18% GST → total_fee = 177.000000
      actual:  1.80% MDR + 18% GST → total_fee = 212.400000
      leakage = 212.400000 - 177.000000 = 35.400000  ✓ matches ground_truth.csv

    GT004: duplicate_fee — PAY004 ₹8,000 debit
      correct total_fee per row: 70.800000
      duplicate row adds another 70.800000
      leakage = 70.800000  ✓ matches ground_truth.csv
    """

    def test_gt001_leakage_equals_ground_truth(self):
        rule_correct = make_rule(mdr_rate="1.50", tax_rate="18.00")
        rule_actual  = make_rule(mdr_rate="1.80", tax_rate="18.00")

        correct = compute_expected_fee(Decimal("10000"), rule_correct)
        actual  = compute_expected_fee(Decimal("10000"), rule_actual)
        leakage = actual.total_fee - correct.total_fee

        assert leakage == Decimal("35.400000"), f"GT001 leakage mismatch: {leakage}"

    def test_gt004_duplicate_leakage(self):
        """The 'leakage' for a duplicate is exactly one copy of the fee row."""
        rule   = make_rule(mdr_rate="0.75", tax_rate="18.00", card_category="debit")
        result = compute_expected_fee(Decimal("8000"), rule)

        assert result.total_fee == Decimal("70.800000"), (
            f"GT004: expected 70.800000, got {result.total_fee}"
        )
