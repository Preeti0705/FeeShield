"""
src/cash_impact/position_calculator.py
=======================================
Day 8 — Cash-Position Calculator

What this module does
---------------------
Answers the question a CFO or finance ops manager actually asks:

  "Given everything our bank account holds RIGHT NOW, everything still EXPECTED
   to arrive from the payment gateway, and everything we know is AT-RISK
   (overcharged fees we haven't recovered + settlement gaps) — can we cover
   our obligations (payroll, vendor payments) due in the next 7 days?"

The answer is a single verdict:
  SAFE     — buffer > 0 after covering obligations; no action needed
  AT_RISK  — buffer is positive but thin (< 20% of obligations); watch closely
  CRITICAL — buffer negative; cash shortfall exists, immediate action needed

Why deterministic, not LLM?
-----------------------------
Cash position is financial fact. The calculator reads exact Decimal values from
the audit pipeline and applies arithmetic. No interpretation or language needed.
An LLM adding/subtracting Decimals would be both slower and less trustworthy
than three lines of Python.

Data flow
---------
                ┌─────────────────────────────────┐
                │  bank_feed.csv (posted rows)     │  → cleared_cash
                └─────────────────────────────────┘
                ┌─────────────────────────────────┐
                │  BankGap list (not-posted /      │  → expected_inflows
                │  late-posted settlements)        │
                └─────────────────────────────────┘
                ┌─────────────────────────────────┐
                │  AuditBatchSummary               │  → at_risk_amount
                │  (fee leakage + bank gaps)       │
                └─────────────────────────────────┘
                ┌─────────────────────────────────┐
                │  obligations.csv (or inline)     │  → obligations_7d
                └─────────────────────────────────┘

  net_position = cleared_cash + expected_inflows - at_risk_amount
  buffer       = net_position - obligations_7d

  SAFE     if buffer >= obligations_7d * 0.20
  AT_RISK  if 0 <= buffer < obligations_7d * 0.20
  CRITICAL if buffer < 0

Design rules (same as the rest of FeeShield)
---------------------------------------------
  - ALL money is Decimal. Never float.
  - ROUND_HALF_UP everywhere.
  - Thresholds are module-level constants — declared once, easy to audit.
  - `CashPosition` is a frozen dataclass — immutable once computed.
  - `validate()` checks internal consistency (same pattern as BatchRoutingReport).
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal

# ── Constants ──────────────────────────────────────────────────────────────────

SIX_DP  = Decimal("0.000001")
TWO_DP  = Decimal("0.01")

# Buffer threshold: if buffer < 20% of obligations → AT_RISK (not yet CRITICAL)
AT_RISK_THRESHOLD_PCT = Decimal("0.20")

# Gap types that represent cash we *expect* to receive but haven't yet
# (contractually owed, just delayed or missing — still in our favour long-term)
EXPECTED_INFLOW_GAP_TYPES = {
    "SETTLEMENT_NOT_POSTED",    # gateway says settled, bank hasn't posted
    "SETTLEMENT_POSTED_LATE",   # bank posted late (we now have it, but delayed)
}

# Gap types that represent cash already gone/locked — pure AT-RISK
AT_RISK_GAP_TYPES = {
    "POSTED_AMOUNT_MISMATCH",   # bank posted less than settled amount
    "POSTING_REVERSED",         # posting was reversed — cash gone back
}

Verdict = Literal["SAFE", "AT_RISK", "CRITICAL"]


# ── Output dataclass ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CashPosition:
    """
    Complete cash position snapshot at the time of the audit run.

    Fields
    ------
    as_of_date          : the date this position was computed
    cleared_cash_inr    : sum of all cleanly posted bank credits (no gaps)
    expected_inflows_inr: sum of outstanding settlements contractually owed
                          (SETTLEMENT_NOT_POSTED + SETTLEMENT_POSTED_LATE)
    at_risk_inr         : fee overcharges + posted-amount-mismatch + reversals
                          (cash we paid extra, or lost, that needs recovery)
    net_position_inr    : cleared_cash + expected_inflows - at_risk
    obligations_7d_inr  : total payroll + vendor obligations due in 7 days
    buffer_inr          : net_position - obligations_7d  (negative = shortfall)
    buffer_pct          : buffer as % of obligations_7d (to evaluate headroom)
    verdict             : SAFE | AT_RISK | CRITICAL
    exceptions          : list of specific gap IDs driving the at-risk amount
    narrative           : 1-2 sentence plain-English summary for the report
    """
    as_of_date:           date
    cleared_cash_inr:     Decimal
    expected_inflows_inr: Decimal
    at_risk_inr:          Decimal
    net_position_inr:     Decimal
    obligations_7d_inr:   Decimal
    buffer_inr:           Decimal
    buffer_pct:           Decimal          # 0.00 → 1.00 scale
    verdict:              Verdict
    exceptions:           list             # list[dict] — gap/claim details driving risk
    narrative:            str

    def _f(self, v: Decimal) -> str:
        return f"Rs {v.quantize(TWO_DP, rounding=ROUND_HALF_UP):>14,.2f}"

    def validate(self) -> list[str]:
        """
        Returns a list of internal consistency errors.
        Empty list = the position is self-consistent.

        Checks:
          1. net_position == cleared_cash + expected_inflows - at_risk
          2. buffer == net_position - obligations_7d
          3. verdict matches buffer thresholds
        """
        errors = []
        expected_net = (
            self.cleared_cash_inr + self.expected_inflows_inr - self.at_risk_inr
        ).quantize(SIX_DP, rounding=ROUND_HALF_UP)
        if self.net_position_inr != expected_net:
            errors.append(
                f"net_position({self.net_position_inr}) != "
                f"cleared+expected-at_risk({expected_net})"
            )

        expected_buffer = (
            self.net_position_inr - self.obligations_7d_inr
        ).quantize(SIX_DP, rounding=ROUND_HALF_UP)
        if self.buffer_inr != expected_buffer:
            errors.append(
                f"buffer({self.buffer_inr}) != "
                f"net_position-obligations({expected_buffer})"
            )

        # Verdict check
        if self.obligations_7d_inr > 0:
            threshold = (self.obligations_7d_inr * AT_RISK_THRESHOLD_PCT).quantize(
                SIX_DP, rounding=ROUND_HALF_UP
            )
            if self.buffer_inr < 0 and self.verdict != "CRITICAL":
                errors.append(f"buffer<0 but verdict is {self.verdict}, expected CRITICAL")
            elif 0 <= self.buffer_inr < threshold and self.verdict not in ("AT_RISK", "CRITICAL"):
                errors.append(f"thin buffer but verdict is {self.verdict}, expected AT_RISK")
            elif self.buffer_inr >= threshold and self.verdict != "SAFE":
                errors.append(f"buffer adequate but verdict is {self.verdict}, expected SAFE")

        return errors

    def as_text_report(self) -> str:
        """Human-readable cash position report for terminal output."""
        verdict_banner = {
            "SAFE":     "  [OK]  VERDICT: SAFE",
            "AT_RISK":  "  [!!]  VERDICT: AT-RISK",
            "CRITICAL": "  [XX]  VERDICT: CRITICAL -- CASH SHORTFALL",
        }[self.verdict]

        lines = [
            "",
            "=" * 70,
            "  CASH POSITION REPORT",
            f"  As of: {self.as_of_date}",
            "=" * 70,
            f"  Cleared cash (confirmed bank postings)  : {self._f(self.cleared_cash_inr)}",
            f"  Expected inflows (outstanding settlement): {self._f(self.expected_inflows_inr)}",
            f"  At-risk deductions (overcharge + gaps)  : {self._f(self.at_risk_inr)}",
            "-" * 70,
            f"  Net cash position                       : {self._f(self.net_position_inr)}",
            f"  7-day obligations (payroll + vendors)   : {self._f(self.obligations_7d_inr)}",
            f"  Buffer                                  : {self._f(self.buffer_inr)}",
            f"  Buffer as % of obligations              : {self.buffer_pct * 100:.1f}%",
            "=" * 70,
            verdict_banner,
            "=" * 70,
        ]

        if self.exceptions:
            lines.append(f"  Exceptions driving risk ({len(self.exceptions)} items):")
            for ex in self.exceptions[:10]:   # cap at 10 for readability
                sid = ex.get("settlement_id", ex.get("claim_id", "?"))
                gap = ex.get("gap_type", ex.get("root_cause", "?"))
                amt = ex.get("cash_impact_inr", ex.get("claim_amount_inr", "0"))
                try:
                    amt_fmt = f"Rs {Decimal(str(amt)).quantize(TWO_DP, rounding=ROUND_HALF_UP):,.2f}"
                except Exception:
                    amt_fmt = str(amt)
                lines.append(f"    {sid:20s}  {gap:30s}  {amt_fmt}")
            if len(self.exceptions) > 10:
                lines.append(f"    ... and {len(self.exceptions) - 10} more")
            lines.append("=" * 70)

        if self.narrative:
            lines.append(f"  Summary: {self.narrative}")
            lines.append("=" * 70)

        return "\n".join(lines)


# ── Core calculator ────────────────────────────────────────────────────────────

def compute_cash_position(
    bank_gaps: list[dict],
    audit_summary_dict: dict,
    obligations_7d_inr: Decimal,
    as_of_date: date | None = None,
    cleared_cash_inr: Decimal | None = None,
) -> CashPosition:
    """
    Compute the merchant's cash position from audit data.

    Parameters
    ----------
    bank_gaps : list[dict]
        Serialised BankGap dicts from bank_matching.py (all gaps, not just escalated).
        Used to split into expected_inflows vs at_risk amounts.

    audit_summary_dict : dict
        The batch_stats dict from run_agents.py, which contains:
          - total_fee_leakage_inr (str Decimal) — overcharges not yet recovered
          - total_bank_cash_at_risk_inr (str Decimal) — bank gap cash at risk
          - total_combined_impact_inr (str Decimal)

    obligations_7d_inr : Decimal
        Total payroll + vendor payments due in the next 7 days.
        In a production system this comes from an ERP / payroll system.
        For this project, we inject it as a parameter so tests can control it.

    as_of_date : date, optional
        Defaults to today.

    cleared_cash_inr : Decimal, optional
        If provided (e.g. from an actual bank balance API), used directly.
        If None, estimated as: sum(net_settled_amount for gaps) - total bank at-risk.
        In production this would come from a real-time bank balance feed.
        For this project we estimate from audit data.

    Returns
    -------
    CashPosition (frozen dataclass) with verdict and full breakdown.

    Why estimated cleared_cash?
      We don't have a live bank balance feed. But we know:
        - How much the gateway *settled* across all payments (net_settled_amount sum)
        - How much of that arrived cleanly (total - at-risk amounts)
      So: cleared_cash = total_settled - cash_at_risk_bank_gaps
      This is a lower-bound estimate — conservative and appropriate for a risk signal.
    """
    if as_of_date is None:
        as_of_date = date.today()

    # ── Parse audit summary ────────────────────────────────────────────────────
    def _d(key: str) -> Decimal:
        v = audit_summary_dict.get(key, "0")
        try:
            return Decimal(str(v)).quantize(SIX_DP, rounding=ROUND_HALF_UP)
        except Exception:
            return Decimal("0")

    total_fee_leakage  = _d("total_fee_leakage_inr")
    total_bank_at_risk = _d("total_bank_cash_at_risk_inr")

    # ── Categorise bank gaps ────────────────────────────────────────────────────
    # expected_inflows: money contractually owed, not yet received
    # gap_at_risk: money that posted incorrectly or was reversed

    expected_inflow_gaps = []
    gap_at_risk_items = []

    for g in bank_gaps:
        gap_type = g.get("gap_type", "")
        cash_impact = Decimal("0")
        try:
            cash_impact = Decimal(str(g.get("cash_impact_inr", "0"))).quantize(
                SIX_DP, rounding=ROUND_HALF_UP
            )
        except Exception:
            pass

        if gap_type in EXPECTED_INFLOW_GAP_TYPES:
            expected_inflow_gaps.append({**g, "_cash": cash_impact})
        elif gap_type in AT_RISK_GAP_TYPES:
            gap_at_risk_items.append({**g, "_cash": cash_impact})

    expected_inflows = sum(
        (g["_cash"] for g in expected_inflow_gaps), Decimal("0")
    ).quantize(SIX_DP, rounding=ROUND_HALF_UP)

    gap_at_risk_total = sum(
        (g["_cash"] for g in gap_at_risk_items), Decimal("0")
    ).quantize(SIX_DP, rounding=ROUND_HALF_UP)

    # ── Total at-risk = fee overcharges + bank reversals/mismatches ────────────
    # Fee leakage = cash we paid extra to the gateway (deducted from our settlement)
    # gap_at_risk = cash that posted wrong or got reversed in the bank
    # NOT included: SETTLEMENT_NOT_POSTED / LATE — those are expected inflows, not losses
    total_at_risk = (total_fee_leakage + gap_at_risk_total).quantize(
        SIX_DP, rounding=ROUND_HALF_UP
    )

    # ── Cleared cash estimate ──────────────────────────────────────────────────
    if cleared_cash_inr is None:
        # Estimate: sum of all net_settled_amounts in the gaps we know about
        # minus the at-risk amounts. This is a lower bound.
        total_settled_in_gaps = sum(
            (
                Decimal(str(g.get("net_settled_amount", "0"))).quantize(SIX_DP, rounding=ROUND_HALF_UP)
                for g in bank_gaps
            ),
            Decimal("0"),
        )
        # Clean postings = gaps where gap_type not in our known categories
        # (the deterministic engine only records gaps; clean postings have no BankGap record)
        # So we use the audit summary's total volume minus at-risk as our cleared cash proxy.
        total_vol = _d("total_volume_inr")
        if total_vol > 0:
            # Fraction that is clean = total_vol - fee_leakage - bank_at_risk
            # But this overstates cleared cash since total_vol includes un-settled payments
            # Use a conservative estimate: total settled in gaps as proxy
            cleared_cash_inr = max(
                total_settled_in_gaps - total_at_risk,
                Decimal("0"),
            )
        else:
            cleared_cash_inr = Decimal("0")

    cleared_cash_inr = cleared_cash_inr.quantize(SIX_DP, rounding=ROUND_HALF_UP)

    # ── Net position ───────────────────────────────────────────────────────────
    net_position = (cleared_cash_inr + expected_inflows - total_at_risk).quantize(
        SIX_DP, rounding=ROUND_HALF_UP
    )

    # ── Buffer ─────────────────────────────────────────────────────────────────
    obligations = obligations_7d_inr.quantize(SIX_DP, rounding=ROUND_HALF_UP)
    buffer = (net_position - obligations).quantize(SIX_DP, rounding=ROUND_HALF_UP)

    # ── Buffer % ───────────────────────────────────────────────────────────────
    if obligations > 0:
        buffer_pct = (buffer / obligations).quantize(SIX_DP, rounding=ROUND_HALF_UP)
    else:
        buffer_pct = Decimal("1")  # no obligations → effectively safe

    # ── Verdict ────────────────────────────────────────────────────────────────
    if buffer < 0:
        verdict: Verdict = "CRITICAL"
    elif obligations > 0 and buffer < (obligations * AT_RISK_THRESHOLD_PCT):
        verdict = "AT_RISK"
    else:
        verdict = "SAFE"

    # ── Exceptions list ────────────────────────────────────────────────────────
    # Surface the specific gaps and overcharge claims driving the at-risk amount
    exceptions = []
    for g in gap_at_risk_items:
        exceptions.append({
            "settlement_id":    g.get("settlement_id", ""),
            "payment_id":       g.get("payment_id", ""),
            "gap_type":         g.get("gap_type", ""),
            "cash_impact_inr":  str(g["_cash"]),
        })

    # ── Narrative ──────────────────────────────────────────────────────────────
    if verdict == "CRITICAL":
        narrative = (
            f"Cash shortfall of Rs {abs(buffer).quantize(TWO_DP, rounding=ROUND_HALF_UP):,.2f} "
            f"against 7-day obligations. Immediate recovery of "
            f"Rs {total_at_risk.quantize(TWO_DP, rounding=ROUND_HALF_UP):,.2f} in "
            f"overcharges and posting gaps is required to meet upcoming payments."
        )
    elif verdict == "AT_RISK":
        narrative = (
            f"Buffer of Rs {buffer.quantize(TWO_DP, rounding=ROUND_HALF_UP):,.2f} "
            f"({buffer_pct * 100:.1f}% of obligations) is below the 20% safety threshold. "
            f"Recovery of Rs {total_at_risk.quantize(TWO_DP, rounding=ROUND_HALF_UP):,.2f} "
            f"in confirmed overcharges would restore adequate headroom."
        )
    else:
        narrative = (
            f"Buffer of Rs {buffer.quantize(TWO_DP, rounding=ROUND_HALF_UP):,.2f} "
            f"({buffer_pct * 100:.1f}% of obligations) exceeds the 20% safety threshold. "
            f"Recovery of Rs {total_at_risk.quantize(TWO_DP, rounding=ROUND_HALF_UP):,.2f} "
            f"in confirmed overcharges will improve the position further."
        )

    position = CashPosition(
        as_of_date=as_of_date,
        cleared_cash_inr=cleared_cash_inr,
        expected_inflows_inr=expected_inflows,
        at_risk_inr=total_at_risk,
        net_position_inr=net_position,
        obligations_7d_inr=obligations,
        buffer_inr=buffer,
        buffer_pct=buffer_pct,
        verdict=verdict,
        exceptions=exceptions,
        narrative=narrative,
    )

    # Validate internal consistency — raises AssertionError if broken
    errors = position.validate()
    if errors:
        raise AssertionError(
            "CashPosition internal consistency error(s):\n" + "\n".join(errors)
        )

    return position
