"""
tests/test_cash_position.py
===========================
Unit tests for the Day 8 cash-position calculator.

All tests are purely deterministic — no LLM calls, no live API, no network.
We construct known Decimal inputs and assert exact outputs.

Test structure:
  1. TestCashPositionVerdict         — the three verdicts (SAFE/AT_RISK/CRITICAL)
  2. TestCashPositionMath            — arithmetic correctness (invariants)
  3. TestCashPositionGapCategorisation — expected_inflows vs at_risk gap routing
  4. TestCashPositionTextReport      — as_text_report() output coverage
  5. TestCashPositionEdgeCases       — zero obligations, no gaps, etc.
"""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cash_impact.position_calculator import (
    compute_cash_position,
    CashPosition,
    AT_RISK_THRESHOLD_PCT,
)

# ── Test helpers ──────────────────────────────────────────────────────────────

def _audit_summary(
    fee_leakage: str = "0",
    bank_at_risk: str = "0",
    total_volume: str = "1000000",
) -> dict:
    return {
        "total_fee_leakage_inr":       fee_leakage,
        "total_bank_cash_at_risk_inr": bank_at_risk,
        "total_combined_impact_inr":   str(Decimal(fee_leakage) + Decimal(bank_at_risk)),
        "total_volume_inr":            total_volume,
    }


def _gap(payment_id: str, gap_type: str, cash_impact: str, net_settled: str = "10000") -> dict:
    return {
        "payment_id":          payment_id,
        "settlement_id":       f"SEL{payment_id[3:]}",
        "gap_type":            gap_type,
        "cash_impact_inr":     cash_impact,
        "net_settled_amount":  net_settled,
        "net_posted_amount":   "0",
        "should_escalate":     gap_type != "HOLD_PLACED_THEN_CLEARED",
    }


def _pos(
    bank_gaps=None,
    obligations=Decimal("100000"),
    fee_leakage="0",
    bank_at_risk="0",
    cleared_cash=None,
) -> CashPosition:
    return compute_cash_position(
        bank_gaps=bank_gaps or [],
        audit_summary_dict=_audit_summary(fee_leakage, bank_at_risk),
        obligations_7d_inr=obligations,
        as_of_date=date(2024, 3, 31),
        cleared_cash_inr=cleared_cash,
    )


# ── 1. Verdict tests ──────────────────────────────────────────────────────────

class TestCashPositionVerdict:
    """
    Tests the three verdict paths.

    Verdict formula:
      CRITICAL  — buffer < 0
      AT_RISK   — 0 <= buffer < obligations * 20%
      SAFE      — buffer >= obligations * 20%

    These tests supply cleared_cash explicitly to control the buffer exactly.
    """

    def test_safe_when_buffer_exceeds_threshold(self):
        """
        obligations = Rs 100,000
        20% threshold = Rs 20,000
        cleared_cash = Rs 200,000, at_risk = 0
        buffer = 200,000 - 100,000 = 100,000 → SAFE
        """
        pos = _pos(
            cleared_cash=Decimal("200000"),
            obligations=Decimal("100000"),
        )
        assert pos.verdict == "SAFE"
        assert pos.buffer_inr > 0

    def test_at_risk_when_buffer_below_threshold(self):
        """
        obligations = Rs 100,000
        20% threshold = Rs 20,000
        cleared_cash = Rs 110,000, at_risk = 0
        buffer = 10,000 → below threshold → AT_RISK
        """
        pos = _pos(
            cleared_cash=Decimal("110000"),
            obligations=Decimal("100000"),
        )
        assert pos.verdict == "AT_RISK"
        assert pos.buffer_inr >= 0

    def test_critical_when_buffer_negative(self):
        """
        obligations = Rs 100,000
        cleared_cash = Rs 50,000, fee_leakage = Rs 5,000 (at_risk)
        net_position = 50,000 + 0 - 5,000 = 45,000
        buffer = 45,000 - 100,000 = -55,000 → CRITICAL
        """
        pos = _pos(
            cleared_cash=Decimal("50000"),
            obligations=Decimal("100000"),
            fee_leakage="5000",
        )
        assert pos.verdict == "CRITICAL"
        assert pos.buffer_inr < 0

    def test_safe_at_threshold_boundary(self):
        """
        Buffer exactly equal to 20% of obligations → SAFE (boundary inclusive).
        obligations = 100,000, 20% = 20,000
        cleared_cash = 120,000 → net = 120,000 → buffer = 20,000 = threshold → SAFE
        """
        pos = _pos(
            cleared_cash=Decimal("120000"),
            obligations=Decimal("100000"),
        )
        assert pos.verdict == "SAFE"

    def test_at_risk_just_below_threshold(self):
        """
        Buffer just below 20% threshold → AT_RISK.
        obligations = 100,000, threshold = 20,000
        cleared_cash = 119,999.99 → buffer = 19,999.99 < 20,000 → AT_RISK
        """
        pos = _pos(
            cleared_cash=Decimal("119999.99"),
            obligations=Decimal("100000"),
        )
        assert pos.verdict == "AT_RISK"

    def test_narrative_contains_verdict(self):
        """Narrative text should describe the verdict clearly."""
        pos_safe = _pos(cleared_cash=Decimal("200000"), obligations=Decimal("100000"))
        pos_crit = _pos(cleared_cash=Decimal("50000"), obligations=Decimal("100000"),
                        fee_leakage="5000")
        assert "shortfall" in pos_crit.narrative.lower() or "critical" in pos_crit.narrative.lower()
        assert "buffer" in pos_safe.narrative.lower()


# ── 2. Math / invariants ──────────────────────────────────────────────────────

class TestCashPositionMath:
    """
    Tests that the arithmetic in CashPosition is internally consistent.
    validate() checks:
      1. net_position == cleared_cash + expected_inflows - at_risk
      2. buffer == net_position - obligations
      3. verdict matches buffer vs threshold
    """

    def test_validate_passes_for_safe_position(self):
        pos = _pos(cleared_cash=Decimal("200000"), obligations=Decimal("100000"))
        assert pos.validate() == []

    def test_validate_passes_for_at_risk_position(self):
        pos = _pos(cleared_cash=Decimal("110000"), obligations=Decimal("100000"))
        assert pos.validate() == []

    def test_validate_passes_for_critical_position(self):
        pos = _pos(
            cleared_cash=Decimal("50000"),
            obligations=Decimal("100000"),
            fee_leakage="5000",
        )
        assert pos.validate() == []

    def test_net_position_formula(self):
        """net_position = cleared_cash + expected_inflows - at_risk."""
        gaps = [_gap("PAY099", "SETTLEMENT_NOT_POSTED", "5000", net_settled="5000")]
        pos = compute_cash_position(
            bank_gaps=gaps,
            audit_summary_dict=_audit_summary(fee_leakage="1000"),
            obligations_7d_inr=Decimal("50000"),
            as_of_date=date(2024, 3, 31),
            cleared_cash_inr=Decimal("30000"),
        )
        expected_net = Decimal("30000") + pos.expected_inflows_inr - pos.at_risk_inr
        assert pos.net_position_inr == expected_net.quantize(
            Decimal("0.000001"), rounding="ROUND_HALF_UP"
        )
        assert pos.validate() == []

    def test_buffer_formula(self):
        """buffer = net_position - obligations."""
        pos = _pos(cleared_cash=Decimal("200000"), obligations=Decimal("80000"))
        assert pos.buffer_inr == pos.net_position_inr - pos.obligations_7d_inr
        assert pos.validate() == []

    def test_buffer_pct_formula(self):
        """buffer_pct = buffer / obligations (when obligations > 0)."""
        pos = _pos(cleared_cash=Decimal("150000"), obligations=Decimal("100000"))
        expected_pct = (pos.buffer_inr / pos.obligations_7d_inr).quantize(
            Decimal("0.000001"), rounding="ROUND_HALF_UP"
        )
        assert pos.buffer_pct == expected_pct
        assert pos.validate() == []

    def test_total_at_risk_is_fee_leakage_plus_bank_risk(self):
        """at_risk = fee_leakage + POSTED_AMOUNT_MISMATCH/REVERSED cash impacts."""
        gaps = [_gap("PAY050", "POSTING_REVERSED", "2000")]
        pos = compute_cash_position(
            bank_gaps=gaps,
            audit_summary_dict=_audit_summary(fee_leakage="500"),
            obligations_7d_inr=Decimal("10000"),
            as_of_date=date(2024, 3, 31),
            cleared_cash_inr=Decimal("20000"),
        )
        # at_risk = fee_leakage(500) + gap_at_risk(2000) = 2500
        assert pos.at_risk_inr == Decimal("2500").quantize(
            Decimal("0.000001"), rounding="ROUND_HALF_UP"
        )
        assert pos.validate() == []


# ── 3. Gap categorisation ─────────────────────────────────────────────────────

class TestCashPositionGapCategorisation:
    """
    Tests that each gap type is routed to the correct bucket.

    SETTLEMENT_NOT_POSTED    → expected_inflows (money owed, not received yet)
    SETTLEMENT_POSTED_LATE   → expected_inflows (same — we eventually got it)
    POSTED_AMOUNT_MISMATCH   → at_risk (bank paid less than settled)
    POSTING_REVERSED         → at_risk (posting cancelled)
    HOLD_PLACED_THEN_CLEARED → ignored (monitoring only; cleared within SLA)
    """

    def test_not_posted_goes_to_expected_inflows(self):
        gaps = [_gap("PAY001", "SETTLEMENT_NOT_POSTED", "10000")]
        pos = compute_cash_position(
            bank_gaps=gaps,
            audit_summary_dict=_audit_summary(),
            obligations_7d_inr=Decimal("50000"),
            as_of_date=date(2024, 3, 31),
            cleared_cash_inr=Decimal("60000"),
        )
        assert pos.expected_inflows_inr == Decimal("10000").quantize(
            Decimal("0.000001"), rounding="ROUND_HALF_UP"
        )
        assert pos.validate() == []

    def test_posted_late_goes_to_expected_inflows(self):
        gaps = [_gap("PAY002", "SETTLEMENT_POSTED_LATE", "8000")]
        pos = compute_cash_position(
            bank_gaps=gaps,
            audit_summary_dict=_audit_summary(),
            obligations_7d_inr=Decimal("50000"),
            as_of_date=date(2024, 3, 31),
            cleared_cash_inr=Decimal("60000"),
        )
        assert pos.expected_inflows_inr == Decimal("8000").quantize(
            Decimal("0.000001"), rounding="ROUND_HALF_UP"
        )

    def test_amount_mismatch_goes_to_at_risk(self):
        gaps = [_gap("PAY003", "POSTED_AMOUNT_MISMATCH", "500")]
        pos = compute_cash_position(
            bank_gaps=gaps,
            audit_summary_dict=_audit_summary(fee_leakage="0"),
            obligations_7d_inr=Decimal("10000"),
            as_of_date=date(2024, 3, 31),
            cleared_cash_inr=Decimal("20000"),
        )
        # at_risk = gap_at_risk(500) + fee_leakage(0) = 500
        assert pos.at_risk_inr == Decimal("500").quantize(
            Decimal("0.000001"), rounding="ROUND_HALF_UP"
        )
        assert pos.validate() == []

    def test_reversed_posting_goes_to_at_risk(self):
        gaps = [_gap("PAY004", "POSTING_REVERSED", "15000")]
        pos = compute_cash_position(
            bank_gaps=gaps,
            audit_summary_dict=_audit_summary(),
            obligations_7d_inr=Decimal("50000"),
            as_of_date=date(2024, 3, 31),
            cleared_cash_inr=Decimal("80000"),
        )
        assert pos.at_risk_inr == Decimal("15000").quantize(
            Decimal("0.000001"), rounding="ROUND_HALF_UP"
        )

    def test_hold_then_cleared_not_in_either_bucket(self):
        """HOLD_PLACED_THEN_CLEARED is monitoring only — should not affect at_risk or expected_inflows."""
        gaps = [_gap("PAY005", "HOLD_PLACED_THEN_CLEARED", "3000")]
        pos = compute_cash_position(
            bank_gaps=gaps,
            audit_summary_dict=_audit_summary(),
            obligations_7d_inr=Decimal("50000"),
            as_of_date=date(2024, 3, 31),
            cleared_cash_inr=Decimal("100000"),
        )
        assert pos.expected_inflows_inr == Decimal("0")
        assert pos.at_risk_inr == Decimal("0")
        assert pos.validate() == []

    def test_mixed_gaps_categorised_correctly(self):
        """Multiple gap types — each goes to the right bucket."""
        gaps = [
            _gap("PAY001", "SETTLEMENT_NOT_POSTED", "10000"),   # expected
            _gap("PAY002", "SETTLEMENT_POSTED_LATE", "5000"),   # expected
            _gap("PAY003", "POSTED_AMOUNT_MISMATCH", "1000"),   # at_risk
            _gap("PAY004", "POSTING_REVERSED", "2000"),         # at_risk
            _gap("PAY005", "HOLD_PLACED_THEN_CLEARED", "3000"), # neither
        ]
        pos = compute_cash_position(
            bank_gaps=gaps,
            audit_summary_dict=_audit_summary(fee_leakage="500"),
            obligations_7d_inr=Decimal("50000"),
            as_of_date=date(2024, 3, 31),
            cleared_cash_inr=Decimal("80000"),
        )
        assert pos.expected_inflows_inr == Decimal("15000").quantize(
            Decimal("0.000001"), rounding="ROUND_HALF_UP"
        )
        # at_risk = mismatch(1000) + reversed(2000) + fee_leakage(500) = 3500
        assert pos.at_risk_inr == Decimal("3500").quantize(
            Decimal("0.000001"), rounding="ROUND_HALF_UP"
        )
        assert pos.validate() == []

    def test_exceptions_list_contains_at_risk_gaps_only(self):
        """Only POSTED_AMOUNT_MISMATCH and POSTING_REVERSED appear in exceptions list."""
        gaps = [
            _gap("PAY001", "SETTLEMENT_NOT_POSTED", "10000"),
            _gap("PAY002", "POSTING_REVERSED", "8000"),
        ]
        pos = compute_cash_position(
            bank_gaps=gaps,
            audit_summary_dict=_audit_summary(),
            obligations_7d_inr=Decimal("50000"),
            as_of_date=date(2024, 3, 31),
            cleared_cash_inr=Decimal("80000"),
        )
        # Only PAY002 (POSTING_REVERSED) should appear in exceptions
        exc_pids = [e["payment_id"] for e in pos.exceptions]
        assert "PAY002" in exc_pids
        assert "PAY001" not in exc_pids


# ── 4. Text report output ─────────────────────────────────────────────────────

class TestCashPositionTextReport:
    def test_report_has_verdict_header(self):
        pos = _pos(cleared_cash=Decimal("200000"), obligations=Decimal("100000"))
        text = pos.as_text_report()
        assert "VERDICT" in text
        assert "SAFE" in text

    def test_critical_verdict_highlighted(self):
        pos = _pos(
            cleared_cash=Decimal("50000"),
            obligations=Decimal("100000"),
            fee_leakage="5000",
        )
        text = pos.as_text_report()
        assert "CRITICAL" in text

    def test_report_shows_obligations(self):
        pos = _pos(cleared_cash=Decimal("200000"), obligations=Decimal("75000"))
        text = pos.as_text_report()
        assert "75,000.00" in text

    def test_report_shows_buffer(self):
        pos = _pos(cleared_cash=Decimal("200000"), obligations=Decimal("100000"))
        text = pos.as_text_report()
        assert "Buffer" in text

    def test_report_shows_as_of_date(self):
        pos = compute_cash_position(
            bank_gaps=[],
            audit_summary_dict=_audit_summary(),
            obligations_7d_inr=Decimal("50000"),
            as_of_date=date(2024, 3, 31),
            cleared_cash_inr=Decimal("100000"),
        )
        text = pos.as_text_report()
        assert "2024-03-31" in text

    def test_exceptions_shown_when_at_risk_gaps_exist(self):
        gaps = [_gap("PAY003", "POSTING_REVERSED", "5000")]
        pos = compute_cash_position(
            bank_gaps=gaps,
            audit_summary_dict=_audit_summary(),
            obligations_7d_inr=Decimal("50000"),
            as_of_date=date(2024, 3, 31),
            cleared_cash_inr=Decimal("80000"),
        )
        text = pos.as_text_report()
        assert "POSTING_REVERSED" in text
        # exceptions table shows settlement_id as primary key
        assert "SEL003" in text   # settlement_id derived from PAY003 in _gap()


# ── 5. Edge cases ─────────────────────────────────────────────────────────────

class TestCashPositionEdgeCases:
    def test_zero_obligations_gives_safe(self):
        """No obligations → no risk → always SAFE."""
        pos = _pos(cleared_cash=Decimal("100"), obligations=Decimal("0"))
        assert pos.verdict == "SAFE"
        assert pos.validate() == []

    def test_no_bank_gaps_all_clean(self):
        """No gaps → expected_inflows = 0, at_risk from fee leakage only."""
        pos = compute_cash_position(
            bank_gaps=[],
            audit_summary_dict=_audit_summary(fee_leakage="0"),
            obligations_7d_inr=Decimal("50000"),
            as_of_date=date(2024, 3, 31),
            cleared_cash_inr=Decimal("100000"),
        )
        assert pos.expected_inflows_inr == Decimal("0")
        assert pos.at_risk_inr == Decimal("0")
        assert pos.verdict == "SAFE"
        assert pos.validate() == []

    def test_realworld_scenario_feeshield_audit(self):
        """
        Simulates the actual FeeShield audit batch numbers:
          - Fee leakage:   Rs 6,458.24
          - Bank at-risk:  Rs 28,965.46 (POSTING_REVERSED + MISMATCH)
          - Expected inflows: Rs 28,965.46 (SETTLEMENT_NOT_POSTED)
          - Cleared cash:  Rs 500,000 (representative merchant balance)
          - Obligations:   Rs 400,000 (payroll + vendor week)

        Expected verdict: SAFE (buffer = 500,000 - at_risk - obligations = large positive)
        """
        gaps = [
            _gap("PAY_GT009", "SETTLEMENT_NOT_POSTED", "28965.46"),   # expected
        ]
        pos = compute_cash_position(
            bank_gaps=gaps,
            audit_summary_dict=_audit_summary(
                fee_leakage="6458.24",
                bank_at_risk="28965.46",
                total_volume="5714234.56",
            ),
            obligations_7d_inr=Decimal("400000"),
            as_of_date=date(2024, 3, 31),
            cleared_cash_inr=Decimal("500000"),
        )
        # at_risk = fee_leakage(6458.24) + bank_gap_at_risk(0 — not_posted is expected, not at_risk)
        # expected_inflows = 28965.46
        # net = 500000 + 28965.46 - 6458.24 = 522507.22
        # buffer = 522507.22 - 400000 = 122507.22 → well above 20% threshold (80000) → SAFE
        assert pos.verdict == "SAFE"
        assert pos.buffer_inr > Decimal("100000")
        assert pos.validate() == []

    def test_no_decimal_floats_anywhere(self):
        """All Decimal fields must be Decimal, not float — financial integrity check."""
        pos = _pos(cleared_cash=Decimal("100000"), obligations=Decimal("80000"))
        assert isinstance(pos.cleared_cash_inr, Decimal)
        assert isinstance(pos.expected_inflows_inr, Decimal)
        assert isinstance(pos.at_risk_inr, Decimal)
        assert isinstance(pos.net_position_inr, Decimal)
        assert isinstance(pos.obligations_7d_inr, Decimal)
        assert isinstance(pos.buffer_inr, Decimal)
        assert isinstance(pos.buffer_pct, Decimal)
