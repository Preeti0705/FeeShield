"""
bank_matching.py — Settlement-to-Bank deterministic matcher
============================================================
Day 3B of the AI Finance Controller build.

What this module does
---------------------
Walks every settlement row and asks 5 structured questions about the bank feed.
Produces a list of BankGap dataclasses — one per settlement with a finding.

The 5 gap types (in detection order):
  1. SETTLEMENT_NOT_POSTED   — settlement exists, no bank row found
  2. SETTLEMENT_POSTED_LATE  — bank row found but posting_date > settlement_date + SLA
  3. POSTED_AMOUNT_MISMATCH  — posted_amount differs from net_settled_amount by > tolerance
  4. POSTING_REVERSED        — net bank credit is zero (original + reversal_entry cancel out)
  5. HOLD_PLACED_THEN_CLEARED — hold placed but cleared within HOLD_SLA_DAYS (monitoring only)

Design rules (same as the fee engine)
--------------------------------------
  - ALL money comparisons use Decimal. Never float.
  - SLA thresholds are module-level constants — declared once, imported everywhere.
  - This module is DETERMINISTIC: no LLM calls, no randomness, no heuristics.
    Every threshold is an explicit constant. Unit tests can assert exact output.
  - The output BankGap dataclass is a TYPED CONTRACT with the agent layer.
    If an agent tries to access a wrong attribute, it gets AttributeError immediately,
    not a silently wrong value.
  - DUP settlement rows (settlement_id ending in '_DUP') are skipped —
    they are internal accounting artifacts with no corresponding bank transfer.

Financial math rule
-------------------
  net_match_tolerance: Decimal = Decimal("0.01")
  If |posted_amount - net_settled_amount| <= tolerance → amounts match (rounding artifact).
  If gap > tolerance → POSTED_AMOUNT_MISMATCH.
  For POSTING_REVERSED: net = sum(all posted_amount rows for this settlement_id).
  If net <= 0 → fully reversed.
"""

import csv
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Literal

# ── SLA Constants ─────────────────────────────────────────────────────────────
# These are the financial authority. Change here and every check updates.

POSTING_SLA_DAYS: int = 3          # bank must post within 3 days of settlement_date
HOLD_SLA_DAYS: int    = 3          # hold must clear within 3 days to be "monitoring only"
AMOUNT_TOLERANCE: Decimal = Decimal("0.01")  # differences ≤ 1 paisa = rounding, not a gap

# ── Money helper ───────────────────────────────────────────────────────────────

SIX_DP = Decimal("0.000001")

def _d(v: str) -> Decimal:
    """Read a money string from CSV into Decimal. Raises on float inputs."""
    if isinstance(v, float):
        raise TypeError(f"STOP: float {v!r} passed to _d(). Use string from CSV.")
    return Decimal(str(v))


def _money(v: Decimal) -> Decimal:
    """Quantize to 6 d.p. with ROUND_HALF_UP for comparison."""
    return v.quantize(SIX_DP, rounding=ROUND_HALF_UP)


# ── Gap type literal ───────────────────────────────────────────────────────────

GapType = Literal[
    "SETTLEMENT_NOT_POSTED",
    "SETTLEMENT_POSTED_LATE",
    "POSTED_AMOUNT_MISMATCH",
    "POSTING_REVERSED",
    "HOLD_PLACED_THEN_CLEARED",
]


# ── Output dataclass ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BankGap:
    """
    One finding from the settlement-to-bank matcher.

    Fields
    ------
    settlement_id     : FK to settlements.csv
    payment_id        : FK to payments.csv
    gap_type          : one of the 5 GapType literals
    settlement_date   : date the settlement was created
    posting_date      : date the first bank row posted (None if not posted)
    net_settled_amount: what the PSP says the merchant should receive
    net_posted_amount : what actually hit the bank (Decimal; 0 if not posted/reversed)
    cash_impact_inr   : |net_settled - net_posted| — the at-risk cash amount
    delay_days        : calendar days between settlement_date and first posting_date
                        (0 if not posted; negative not possible by construction)
    notes             : raw notes field from the bank_feed row(s), joined with ' | '
    should_escalate   : False for HOLD_PLACED_THEN_CLEARED within SLA; True otherwise
    """
    settlement_id:      str
    payment_id:         str
    gap_type:           GapType
    settlement_date:    date
    posting_date:       date | None
    net_settled_amount: Decimal
    net_posted_amount:  Decimal
    cash_impact_inr:    Decimal
    delay_days:         int
    notes:              str
    should_escalate:    bool

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for JSON / Streamlit table)."""
        return {
            "settlement_id":       self.settlement_id,
            "payment_id":          self.payment_id,
            "gap_type":            self.gap_type,
            "settlement_date":     str(self.settlement_date),
            "posting_date":        str(self.posting_date) if self.posting_date else "",
            "net_settled_amount":  str(self.net_settled_amount),
            "net_posted_amount":   str(self.net_posted_amount),
            "cash_impact_inr":     str(self.cash_impact_inr),
            "delay_days":          self.delay_days,
            "notes":               self.notes,
            "should_escalate":     self.should_escalate,
        }


# ── CSV loaders ────────────────────────────────────────────────────────────────

def load_settlements(path: str | Path) -> list[dict]:
    """
    Load settlements.csv. Returns rows as dicts with:
      - monetary fields as Decimal (net_settled_amount, actual_mdr_fee, actual_tax, actual_fixed_fee)
      - settlement_date as date object
      - all other fields as str
    Skips _DUP rows (internal accounting only, no bank transfer).
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["settlement_id"].endswith("_DUP"):
                continue
            rows.append({
                **r,
                "settlement_date":    date.fromisoformat(r["settlement_date"]),
                "net_settled_amount": _d(r["net_settled_amount"]),
                "actual_mdr_fee":     _d(r["actual_mdr_fee"]),
                "actual_tax":         _d(r["actual_tax"]),
                "actual_fixed_fee":   _d(r["actual_fixed_fee"]),
            })
    return rows


def load_bank_feed(path: str | Path) -> list[dict]:
    """
    Load bank_feed.csv. Returns rows with:
      - posted_amount as Decimal
      - posting_date as date object
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                **r,
                "posting_date":  date.fromisoformat(r["posting_date"]),
                "posted_amount": _d(r["posted_amount"]),
            })
    return rows


# ── Core matching logic ────────────────────────────────────────────────────────

def match_settlements_to_bank(
    settlements: list[dict],
    bank_feed:   list[dict],
    posting_sla_days: int   = POSTING_SLA_DAYS,
    hold_sla_days:    int   = HOLD_SLA_DAYS,
    amount_tolerance: Decimal = AMOUNT_TOLERANCE,
) -> list[BankGap]:
    """
    Main entry point. Walk every settlement and classify its bank status.

    Parameters
    ----------
    settlements       : output of load_settlements()
    bank_feed         : output of load_bank_feed()
    posting_sla_days  : max days between settlement_date and posting_date (default 3)
    hold_sla_days     : max days a hold may stay before it becomes an exception (default 3)
    amount_tolerance  : max Decimal shortfall treated as rounding, not a gap (default 0.01)

    Returns
    -------
    List of BankGap, one per settlement that has a finding.
    Clean settlements (posted on time, correct amount, no reversal/hold) produce no BankGap.

    Detection order matters
    -----------------------
    We check SETTLEMENT_NOT_POSTED first — if no bank rows exist, all other checks are moot.
    For settlements with rows, we check POSTING_REVERSED before POSTED_AMOUNT_MISMATCH
    because a reversed posting has net_amount=0 (would also trip the mismatch check if
    checked first, producing a misleading label).
    """

    # Build a lookup: settlement_id → [bank_row, ...]
    # Note: a settlement may have multiple bank rows (reversal, hold + clear)
    bank_by_sel: dict[str, list[dict]] = {}
    for row in bank_feed:
        sid = row["settlement_id"]
        bank_by_sel.setdefault(sid, []).append(row)

    gaps: list[BankGap] = []

    for sel in settlements:
        sid          = sel["settlement_id"]
        pid          = sel["payment_id"]
        settle_date  = sel["settlement_date"]
        net_settled  = _money(sel["net_settled_amount"])
        bank_rows    = bank_by_sel.get(sid, [])

        # ── Check 1: SETTLEMENT_NOT_POSTED ────────────────────────────────────
        if not bank_rows:
            gaps.append(BankGap(
                settlement_id      = sid,
                payment_id         = pid,
                gap_type           = "SETTLEMENT_NOT_POSTED",
                settlement_date    = settle_date,
                posting_date       = None,
                net_settled_amount = net_settled,
                net_posted_amount  = _money(Decimal("0")),
                cash_impact_inr    = net_settled,
                delay_days         = 0,
                notes              = "No bank posting found for this settlement",
                should_escalate    = True,
            ))
            continue

        # Sort bank rows by posting_date ascending (important for hold→clear detection)
        bank_rows_sorted = sorted(bank_rows, key=lambda r: r["posting_date"])

        # Separate by status
        posted_rows   = [r for r in bank_rows_sorted if r["status"] == "posted"]
        reversed_rows = [r for r in bank_rows_sorted if r["status"] == "reversed"]
        rev_entry_rows= [r for r in bank_rows_sorted if r["status"] == "reversal_entry"]
        held_rows     = [r for r in bank_rows_sorted if r["status"] == "held"]

        # ── Check 2: POSTING_REVERSED ─────────────────────────────────────────
        # Condition: at least one 'reversed' row AND at least one 'reversal_entry' row.
        # Net cash = sum of all posted_amounts across all rows (reversal_entry has amount=0,
        # 'reversed' row keeps the original amount for audit trail — net still ≈ 0).
        if reversed_rows and rev_entry_rows:
            first_posting = bank_rows_sorted[0]
            # Cash impact = original posted amount. After reversal the merchant
            # holds Rs0. The 'reversal_entry' row has posted_amount=0 by design.
            impact = _money(sum(
                (r["posted_amount"] for r in reversed_rows),
                Decimal("0")
            ))
            delay  = (first_posting["posting_date"] - settle_date).days
            all_notes = " | ".join(r["notes"] for r in bank_rows_sorted if r["notes"])
            gaps.append(BankGap(
                settlement_id      = sid,
                payment_id         = pid,
                gap_type           = "POSTING_REVERSED",
                settlement_date    = settle_date,
                posting_date       = first_posting["posting_date"],
                net_settled_amount = net_settled,
                net_posted_amount  = _money(Decimal("0")),
                cash_impact_inr    = impact,
                delay_days         = delay,
                notes              = all_notes,
                should_escalate    = True,
            ))
            continue

        # ── Check 3: HOLD_PLACED_THEN_CLEARED ────────────────────────────────
        # Condition: has a 'held' row AND a subsequent 'posted' row.
        # If hold_duration <= hold_sla_days → monitoring (should_escalate=False)
        # If hold_duration > hold_sla_days → escalate
        if held_rows and posted_rows:
            hold_row  = held_rows[0]
            clear_row = posted_rows[-1]   # last posted row is the cleared state
            hold_duration = (clear_row["posting_date"] - hold_row["posting_date"]).days
            escalate  = hold_duration > hold_sla_days
            delay     = (hold_row["posting_date"] - settle_date).days
            # After clearing, amounts should match
            net_posted = _money(clear_row["posted_amount"])
            shortfall  = _money(abs(net_settled - net_posted))
            all_notes  = " | ".join(r["notes"] for r in bank_rows_sorted if r["notes"])
            gaps.append(BankGap(
                settlement_id      = sid,
                payment_id         = pid,
                gap_type           = "HOLD_PLACED_THEN_CLEARED",
                settlement_date    = settle_date,
                posting_date       = clear_row["posting_date"],
                net_settled_amount = net_settled,
                net_posted_amount  = net_posted,
                cash_impact_inr    = shortfall if shortfall > amount_tolerance else Decimal("0"),
                delay_days         = delay,
                notes              = all_notes,
                should_escalate    = escalate,
            ))
            continue

        # For remaining checks we use the first 'posted' (or any) row
        primary_row = posted_rows[0] if posted_rows else bank_rows_sorted[0]
        posting_date = primary_row["posting_date"]
        delay_days   = (posting_date - settle_date).days
        net_posted   = _money(primary_row["posted_amount"])
        notes_val    = primary_row.get("notes", "")

        # ── Check 4: SETTLEMENT_POSTED_LATE ──────────────────────────────────
        if delay_days > posting_sla_days:
            gaps.append(BankGap(
                settlement_id      = sid,
                payment_id         = pid,
                gap_type           = "SETTLEMENT_POSTED_LATE",
                settlement_date    = settle_date,
                posting_date       = posting_date,
                net_settled_amount = net_settled,
                net_posted_amount  = net_posted,
                cash_impact_inr    = _money(Decimal("0")),  # money arrived, just late
                delay_days         = delay_days,
                notes              = notes_val,
                should_escalate    = True,
            ))
            continue

        # ── Check 5: POSTED_AMOUNT_MISMATCH ──────────────────────────────────
        shortfall = _money(net_settled - net_posted)
        if abs(shortfall) > amount_tolerance:
            gaps.append(BankGap(
                settlement_id      = sid,
                payment_id         = pid,
                gap_type           = "POSTED_AMOUNT_MISMATCH",
                settlement_date    = settle_date,
                posting_date       = posting_date,
                net_settled_amount = net_settled,
                net_posted_amount  = net_posted,
                cash_impact_inr    = _money(abs(shortfall)),
                delay_days         = delay_days,
                notes              = notes_val,
                should_escalate    = True,
            ))
            continue

        # Settlement is clean — no BankGap produced

    return gaps


# ── Summary helpers ────────────────────────────────────────────────────────────

def summarise_gaps(gaps: list[BankGap]) -> dict:
    """
    Returns a summary dict suitable for the Orchestrator agent's batch report.

    Keys
    ----
    total_gaps           : int
    escalate_count       : int  (should_escalate=True)
    monitoring_count     : int  (should_escalate=False)
    total_cash_at_risk   : Decimal  (sum of cash_impact_inr)
    by_gap_type          : dict[GapType, int]  (count per type)
    """
    total_cash = sum(
        (g.cash_impact_inr for g in gaps),
        Decimal("0")   # start value ensures result is Decimal even when gaps is empty
    )
    by_type: dict[str, int] = {}
    for g in gaps:
        by_type[g.gap_type] = by_type.get(g.gap_type, 0) + 1

    return {
        "total_gaps":         len(gaps),
        "escalate_count":     sum(1 for g in gaps if g.should_escalate),
        "monitoring_count":   sum(1 for g in gaps if not g.should_escalate),
        "total_cash_at_risk": _money(total_cash),
        "by_gap_type":        by_type,
    }
