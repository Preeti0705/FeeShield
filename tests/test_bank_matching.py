"""
tests/test_bank_matching.py — Unit tests for the settlement-to-bank matcher
============================================================================
Test strategy:
  - All tests are self-contained: they build minimal in-memory data instead of
    reading from disk. This keeps tests deterministic and fast.
  - A few integration tests DO read the real CSVs to verify the planted cases.
  - LLM is NEVER called here. This module is purely deterministic.
  - All expected values are hand-calculated (same rule as test_fees.py).

Test classes:
  TestNormalPosting          — clean settlement produces no gap
  TestSettlementNotPosted    — GT009 pattern: no bank row
  TestSettlementPostedLate   — GT010 pattern: posting_date > settle + SLA
  TestPostedAmountMismatch   — GT011 pattern: posted Rs500 short
  TestPostingReversed        — GT012 pattern: reversed + reversal_entry
  TestHoldPlacedThenCleared  — GT013 pattern: hold then posted within SLA
  TestHoldExceedsSLA         — hold that exceeds SLA → should_escalate=True
  TestAmountTolerance        — differences <= 0.01 are rounding, not gaps
  TestSummariseGaps          — summary dict counts and cash total
  TestIntegration            — reads real CSVs, checks all 5 planted bank cases
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reconciliation.bank_matching import (
    BankGap,
    match_settlements_to_bank,
    summarise_gaps,
    POSTING_SLA_DAYS,
    HOLD_SLA_DAYS,
    AMOUNT_TOLERANCE,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _d(v: str) -> Decimal:
    return Decimal(v)

def _sel(settlement_id="SEL001", payment_id="PAY001",
         settlement_date="2024-01-05",
         net_settled_amount="9700.000000") -> dict:
    """Minimal settlement row dict (as returned by load_settlements)."""
    return {
        "settlement_id":      settlement_id,
        "payment_id":         payment_id,
        "settlement_date":    date.fromisoformat(settlement_date),
        "net_settled_amount": _d(net_settled_amount),
        "actual_mdr_fee":     _d("150.000000"),
        "actual_tax":         _d("27.000000"),
        "actual_fixed_fee":   _d("0.000000"),
        "notes":              "",
    }

def _bank(settlement_id="SEL001",
          posting_date="2024-01-07",
          posted_amount="9700.000000",
          status="posted",
          notes="") -> dict:
    """Minimal bank_feed row dict (as returned by load_bank_feed)."""
    return {
        "bank_txn_id":    "BKTXN0001",
        "settlement_id":  settlement_id,
        "posting_date":   date.fromisoformat(posting_date),
        "posted_amount":  _d(posted_amount),
        "status":         status,
        "notes":          notes,
    }


# ── TestNormalPosting ──────────────────────────────────────────────────────────

class TestNormalPosting:
    """Clean settlements produce NO BankGap."""

    def test_on_time_correct_amount(self):
        """Posted within SLA, exact amount → no gap."""
        sels = [_sel(settlement_date="2024-01-05", net_settled_amount="9700.000000")]
        bank = [_bank(posting_date="2024-01-07", posted_amount="9700.000000")]
        gaps = match_settlements_to_bank(sels, bank)
        assert gaps == []

    def test_posted_on_sla_boundary(self):
        """Posted exactly on day 3 (= SLA) → no gap (boundary is inclusive)."""
        sels = [_sel(settlement_date="2024-01-05")]
        bank = [_bank(posting_date="2024-01-08")]  # 3 days later
        gaps = match_settlements_to_bank(sels, bank)
        assert gaps == []

    def test_amount_within_tolerance(self):
        """Posted 1 paisa short (0.01) → within tolerance → no gap."""
        sels = [_sel(net_settled_amount="9700.000000")]
        bank = [_bank(posted_amount="9699.990000")]  # gap = 0.01
        gaps = match_settlements_to_bank(sels, bank, amount_tolerance=_d("0.01"))
        assert gaps == []

    def test_dup_settlements_skipped(self):
        """_DUP settlement rows loaded by load_settlements are already stripped.
        Here we simulate what happens if a DUP somehow reaches the matcher:
        it should produce a gap (no bank row), but the real loader skips them."""
        # The loader skips _DUP rows, so this test confirms the matcher itself
        # treats an unknown settlement_id as not-posted.
        sels = [_sel(settlement_id="SEL004_DUP")]
        bank = []   # no bank row for a _DUP
        gaps = match_settlements_to_bank(sels, bank)
        assert len(gaps) == 1
        assert gaps[0].gap_type == "SETTLEMENT_NOT_POSTED"


# ── TestSettlementNotPosted ────────────────────────────────────────────────────

class TestSettlementNotPosted:
    """GT009 pattern: settlement exists, no bank row."""

    def test_no_bank_row(self):
        sels = [_sel(settlement_id="SEL009", net_settled_amount="17660.160000")]
        bank = []
        gaps = match_settlements_to_bank(sels, bank)
        assert len(gaps) == 1
        g = gaps[0]
        assert g.gap_type          == "SETTLEMENT_NOT_POSTED"
        assert g.settlement_id     == "SEL009"
        assert g.posting_date      is None
        assert g.net_posted_amount == _d("0")
        assert g.cash_impact_inr   == _d("17660.160000")
        assert g.delay_days        == 0
        assert g.should_escalate   is True

    def test_other_settlement_unaffected(self):
        """SEL001 has a bank row; SEL009 does not. Only SEL009 flagged."""
        sels = [
            _sel(settlement_id="SEL001", net_settled_amount="9700.000000"),
            _sel(settlement_id="SEL009", net_settled_amount="17660.160000"),
        ]
        bank = [_bank(settlement_id="SEL001")]
        gaps = match_settlements_to_bank(sels, bank)
        assert len(gaps) == 1
        assert gaps[0].settlement_id == "SEL009"


# ── TestSettlementPostedLate ───────────────────────────────────────────────────

class TestSettlementPostedLate:
    """GT010 pattern: bank row exists but posting_date > settlement_date + SLA."""

    def test_posted_7_days_late(self):
        """Posted 7 days late (SLA=3) → SETTLEMENT_POSTED_LATE."""
        settle_dt = date(2024, 2, 14)  # SEL010 = PAY010 txn_date + 2
        post_dt   = settle_dt + timedelta(days=7)
        sels = [_sel(settlement_id="SEL010",
                     settlement_date=str(settle_dt),
                     net_settled_amount="6290.000000")]
        bank = [_bank(settlement_id="SEL010",
                      posting_date=str(post_dt),
                      posted_amount="6290.000000")]
        gaps = match_settlements_to_bank(sels, bank)
        assert len(gaps) == 1
        g = gaps[0]
        assert g.gap_type        == "SETTLEMENT_POSTED_LATE"
        assert g.delay_days      == 7
        assert g.cash_impact_inr == _d("0")  # money DID arrive
        assert g.should_escalate is True

    def test_posted_4_days_late(self):
        """Posted 4 days late (just over SLA=3) → flagged."""
        settle_dt = date(2024, 1, 10)
        post_dt   = settle_dt + timedelta(days=4)
        sels = [_sel(settlement_date=str(settle_dt))]
        bank = [_bank(posting_date=str(post_dt))]
        gaps = match_settlements_to_bank(sels, bank)
        assert len(gaps) == 1
        assert gaps[0].gap_type == "SETTLEMENT_POSTED_LATE"

    def test_posted_exactly_sla_not_flagged(self):
        """Posted exactly SLA days (3) → NOT flagged (inclusive boundary)."""
        settle_dt = date(2024, 1, 10)
        post_dt   = settle_dt + timedelta(days=POSTING_SLA_DAYS)  # exactly 3
        sels = [_sel(settlement_date=str(settle_dt))]
        bank = [_bank(posting_date=str(post_dt))]
        gaps = match_settlements_to_bank(sels, bank)
        assert gaps == []

    def test_custom_sla(self):
        """Custom posting_sla_days=5: delay=4 should NOT be flagged."""
        settle_dt = date(2024, 1, 10)
        post_dt   = settle_dt + timedelta(days=4)
        sels = [_sel(settlement_date=str(settle_dt))]
        bank = [_bank(posting_date=str(post_dt))]
        gaps = match_settlements_to_bank(sels, bank, posting_sla_days=5)
        assert gaps == []


# ── TestPostedAmountMismatch ───────────────────────────────────────────────────

class TestPostedAmountMismatch:
    """GT011 pattern: posted_amount differs from net_settled by > tolerance."""

    def test_short_by_500(self):
        """Posted Rs500 short → POSTED_AMOUNT_MISMATCH with impact=500."""
        net = _d("24050.000000")
        posted = net - _d("500")
        sels = [_sel(settlement_id="SEL011",
                     settlement_date="2024-02-10",   # explicit settle date
                     net_settled_amount=str(net))]
        bank = [_bank(settlement_id="SEL011",
                      posting_date="2024-02-12",      # 2 days later = within SLA
                      posted_amount=str(posted))]
        gaps = match_settlements_to_bank(sels, bank)
        assert len(gaps) == 1
        g = gaps[0]
        assert g.gap_type        == "POSTED_AMOUNT_MISMATCH"
        assert g.cash_impact_inr == _d("500.000000")
        assert g.should_escalate is True


    def test_short_by_tolerance_boundary(self):
        """Posted exactly tolerance (0.01) short → no gap."""
        net    = _d("9700.000000")
        posted = net - AMOUNT_TOLERANCE
        sels = [_sel(net_settled_amount=str(net))]
        bank = [_bank(posting_date="2024-01-07", posted_amount=str(posted))]
        gaps = match_settlements_to_bank(sels, bank)
        assert gaps == []

    def test_over_posted(self):
        """Posted Rs100 MORE than settled (bank error in merchant's favor) → MISMATCH."""
        net    = _d("9700.000000")
        posted = net + _d("100")
        sels = [_sel(net_settled_amount=str(net))]
        bank = [_bank(posting_date="2024-01-07", posted_amount=str(posted))]
        gaps = match_settlements_to_bank(sels, bank)
        assert len(gaps) == 1
        assert gaps[0].gap_type        == "POSTED_AMOUNT_MISMATCH"
        assert gaps[0].cash_impact_inr == _d("100.000000")


# ── TestPostingReversed ────────────────────────────────────────────────────────

class TestPostingReversed:
    """GT012 pattern: 'reversed' row + 'reversal_entry' row for same settlement."""

    def test_reversed_pattern(self):
        """Two rows: original (reversed) + reversal_entry → POSTING_REVERSED."""
        net = _d("10805.300000")
        sels = [_sel(settlement_id="SEL012", net_settled_amount=str(net))]
        bank = [
            _bank(settlement_id="SEL012",
                  posting_date="2024-03-07",
                  posted_amount=str(net),
                  status="reversed",
                  notes="PLANTED:posting_reversed|original_post"),
            _bank(settlement_id="SEL012",
                  posting_date="2024-03-08",
                  posted_amount="0.000000",
                  status="reversal_entry",
                  notes="PLANTED:posting_reversed|reversal_debit"),
        ]
        gaps = match_settlements_to_bank(sels, bank)
        assert len(gaps) == 1
        g = gaps[0]
        assert g.gap_type           == "POSTING_REVERSED"
        assert g.net_posted_amount  == _d("0")
        assert g.cash_impact_inr    == net          # full amount at risk
        assert g.should_escalate    is True
        assert "original_post"      in g.notes

    def test_reversal_detected_before_amount_check(self):
        """Reversal must be classified before POSTED_AMOUNT_MISMATCH to avoid wrong label."""
        net = _d("5000.000000")
        sels = [_sel(net_settled_amount=str(net))]
        # If we checked amount first, the net would be 0 → Rs5000 shortfall → MISMATCH
        # Correct: should be POSTING_REVERSED
        bank = [
            _bank(posting_date="2024-01-07", posted_amount=str(net), status="reversed"),
            _bank(posting_date="2024-01-08", posted_amount="0.000000", status="reversal_entry"),
        ]
        gaps = match_settlements_to_bank(sels, bank)
        assert gaps[0].gap_type == "POSTING_REVERSED"


# ── TestHoldPlacedThenCleared ─────────────────────────────────────────────────

class TestHoldPlacedThenCleared:
    """GT013 pattern: 'held' row then 'posted' row within HOLD_SLA_DAYS."""

    def test_hold_cleared_within_sla(self):
        """Hold clears in 2 days (SLA=3) → should_escalate=False (monitoring only)."""
        net = _d("9260.000000")
        hold_dt  = date(2024, 3, 12)
        clear_dt = hold_dt + timedelta(days=2)
        sels = [_sel(settlement_id="SEL013",
                     settlement_date="2024-03-11",
                     net_settled_amount=str(net))]
        bank = [
            _bank(settlement_id="SEL013",
                  posting_date=str(hold_dt),
                  posted_amount=str(net),
                  status="held",
                  notes="PLANTED:hold_placed_then_cleared|hold_event"),
            _bank(settlement_id="SEL013",
                  posting_date=str(clear_dt),
                  posted_amount=str(net),
                  status="posted",
                  notes="PLANTED:hold_placed_then_cleared|cleared_event"),
        ]
        gaps = match_settlements_to_bank(sels, bank)
        assert len(gaps) == 1
        g = gaps[0]
        assert g.gap_type        == "HOLD_PLACED_THEN_CLEARED"
        assert g.should_escalate is False
        assert g.cash_impact_inr == _d("0")  # cleared amount matches

    def test_hold_exceeds_sla(self):
        """Hold clears in 5 days (SLA=3) → should_escalate=True."""
        net = _d("9260.000000")
        hold_dt  = date(2024, 3, 12)
        clear_dt = hold_dt + timedelta(days=5)
        sels = [_sel(settlement_date="2024-03-11", net_settled_amount=str(net))]
        bank = [
            _bank(posting_date=str(hold_dt), posted_amount=str(net), status="held"),
            _bank(posting_date=str(clear_dt), posted_amount=str(net), status="posted"),
        ]
        gaps = match_settlements_to_bank(sels, bank, hold_sla_days=3)
        assert len(gaps) == 1
        assert gaps[0].should_escalate is True
        assert gaps[0].gap_type        == "HOLD_PLACED_THEN_CLEARED"


# ── TestSummariseGaps ──────────────────────────────────────────────────────────

class TestSummariseGaps:
    """Test the summary dict helper."""

    def test_empty_gaps(self):
        summary = summarise_gaps([])
        assert summary["total_gaps"]         == 0
        assert summary["escalate_count"]     == 0
        assert summary["monitoring_count"]   == 0
        assert summary["total_cash_at_risk"] == _d("0")
        assert summary["by_gap_type"]        == {}

    def test_mixed_gaps(self):
        """2 escalate + 1 monitoring; total cash = 500 + 17660.16 + 0."""
        sels = [
            _sel("SEL009", "PAY009", "2024-01-12", "17660.160000"),
            _sel("SEL010", "PAY010", "2024-02-14", "6290.000000"),
            _sel("SEL013", "PAY013", "2024-03-11", "9260.000000"),
        ]
        bank = [
            # SEL009: no bank row
            # SEL010: posted 7 days late
            _bank("SEL010", "2024-02-21", "6290.000000"),
            # SEL013: hold then cleared in 2 days
            _bank("SEL013", "2024-03-12", "9260.000000", status="held"),
            _bank("SEL013", "2024-03-14", "9260.000000", status="posted"),
        ]
        gaps = match_settlements_to_bank(sels, bank)
        assert len(gaps) == 3
        summary = summarise_gaps(gaps)
        assert summary["escalate_count"]   == 2
        assert summary["monitoring_count"] == 1
        assert summary["total_cash_at_risk"] == _d("17660.160000")
        assert summary["by_gap_type"]["SETTLEMENT_NOT_POSTED"]  == 1
        assert summary["by_gap_type"]["SETTLEMENT_POSTED_LATE"] == 1
        assert summary["by_gap_type"]["HOLD_PLACED_THEN_CLEARED"] == 1


# ── TestIntegration ────────────────────────────────────────────────────────────

class TestIntegration:
    """
    Read real CSVs from data/ and verify the 5 planted bank cases are detected.
    This test reads from disk — it may be skipped if data/ CSVs don't exist.
    """

    @pytest.fixture(autouse=True)
    def load_data(self):
        data_dir = Path(__file__).parent.parent / "data"
        if not (data_dir / "settlements.csv").exists():
            pytest.skip("data/ CSVs not found — run python data/generate_data.py first")
        from src.reconciliation.bank_matching import load_settlements, load_bank_feed
        self.settlements = load_settlements(data_dir / "settlements.csv")
        self.bank_feed   = load_bank_feed(data_dir / "bank_feed.csv")
        self.gaps        = match_settlements_to_bank(self.settlements, self.bank_feed)
        self.gaps_by_sel = {g.settlement_id: g for g in self.gaps}

    def test_gt009_not_posted(self):
        """SEL009 must produce SETTLEMENT_NOT_POSTED."""
        assert "SEL009" in self.gaps_by_sel
        g = self.gaps_by_sel["SEL009"]
        assert g.gap_type      == "SETTLEMENT_NOT_POSTED"
        assert g.posting_date  is None
        # GT009 bank impact = net settled = 17660.160000
        assert g.cash_impact_inr == Decimal("17660.160000")
        assert g.should_escalate is True

    def test_gt010_posted_late(self):
        """SEL010 must produce SETTLEMENT_POSTED_LATE with delay >= 7 days."""
        assert "SEL010" in self.gaps_by_sel
        g = self.gaps_by_sel["SEL010"]
        assert g.gap_type    == "SETTLEMENT_POSTED_LATE"
        assert g.delay_days  >= 7
        assert g.should_escalate is True

    def test_gt011_amount_mismatch(self):
        """SEL011 must produce POSTED_AMOUNT_MISMATCH with impact = Rs500."""
        assert "SEL011" in self.gaps_by_sel
        g = self.gaps_by_sel["SEL011"]
        assert g.gap_type        == "POSTED_AMOUNT_MISMATCH"
        assert g.cash_impact_inr == Decimal("500.000000")
        assert g.should_escalate is True

    def test_gt012_reversed(self):
        """SEL012 must produce POSTING_REVERSED."""
        assert "SEL012" in self.gaps_by_sel
        g = self.gaps_by_sel["SEL012"]
        assert g.gap_type          == "POSTING_REVERSED"
        assert g.net_posted_amount == Decimal("0")
        assert g.should_escalate   is True

    def test_gt013_hold_cleared_monitoring_only(self):
        """SEL013 must produce HOLD_PLACED_THEN_CLEARED with should_escalate=False."""
        assert "SEL013" in self.gaps_by_sel
        g = self.gaps_by_sel["SEL013"]
        assert g.gap_type        == "HOLD_PLACED_THEN_CLEARED"
        assert g.should_escalate is False

    def test_clean_settlements_not_in_gaps(self):
        """SEL001 (a normal clean settlement) must NOT appear in gaps."""
        assert "SEL001" not in self.gaps_by_sel

    def test_total_gap_count(self):
        """Exactly 4 escalations + 1 monitoring = 5 total gaps from planted cases.
        There may also be gaps from the systematic pools — that's expected and OK.
        We check at minimum 4 escalations and 1 monitoring."""
        escalations = [g for g in self.gaps if g.should_escalate]
        monitoring  = [g for g in self.gaps if not g.should_escalate]
        assert len(escalations) >= 4, f"Expected >=4 escalations, got {len(escalations)}"
        assert len(monitoring)  >= 1, f"Expected >=1 monitoring gap, got {len(monitoring)}"
