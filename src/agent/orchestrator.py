"""
src/agent/orchestrator.py
=========================
Deterministic batch routing and outcome aggregation for the agent layer.

What this module does:
  After the LangGraph graph completes, we have:
    - All variance_records (deterministic engine — all 236 settlements)
    - All bank_gaps (deterministic engine — all 236 bank postings)
    - reviews[] (Reviewer verdicts — only the investigated subset)
    - routing_decision (Orchestrator's final call)

  This module computes the FULL batch count report that the master plan requires:
    "N transactions: X auto-matched, Y investigated, Z claim-ready,
     W escalation, V monitoring, U human_review"
  with numbers that sum correctly.

Why separate from graph.py / nodes.py?
  The Orchestrator LLM node writes the dispute letter and decides escalate/claim/monitor
  for the *investigated subset*. But the full batch report needs to account for ALL
  transactions — including the 220+ that passed the deterministic engine cleanly.
  That aggregation is deterministic math, not LLM work, so it belongs here.

Routing buckets (mutually exclusive, collectively exhaustive):
  auto_matched   — passed all deterministic checks, no variance, bank posting clean
  claim_ready    — Reviewer approved, routing_decision='claim' or 'monitor'
  escalated      — Reviewer approved or escalated, routing_decision='escalate'
  human_review   — Reviewer sent to human_queue (label disagreement)
  monitoring     — Bank gaps that are non-escalating (HOLD_PLACED_THEN_CLEARED etc.)

Invariant that must hold:
  auto_matched + investigated == total_processed
  claim_ready + escalated + human_review + monitoring == investigated
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

TWO_DP = Decimal("0.01")
SIX_DP = Decimal("0.000001")


@dataclass(frozen=True)
class BatchRoutingReport:
    """
    Complete batch routing outcome report.

    All counts are transaction-level (one payment_id = one count unit).
    All amounts are Rs with 2dp rounding.

    Invariants:
      auto_matched + investigated == total_processed
      claim_ready + escalated + human_review + monitoring == investigated
    """
    # ── Totals ─────────────────────────────────────────────────────────────────
    total_processed: int       # all payments that went through the deterministic engine
    total_volume_inr: Decimal  # gross GMV processed

    # ── Routing buckets ────────────────────────────────────────────────────────
    auto_matched: int          # clean — passed deterministic engine with no issues
    investigated: int          # sent to LangGraph agent pipeline

    # Sub-buckets of investigated (must sum to investigated)
    claim_ready: int           # Reviewer approved + routing_decision in ('claim', 'monitor')
    escalated: int             # routing_decision == 'escalate'
    monitoring: int            # non-escalating bank gaps (HOLD_PLACED_THEN_CLEARED)
    human_review: int          # sent to human_queue (label disagreement)

    # ── Financial impact ───────────────────────────────────────────────────────
    total_fee_leakage_inr: Decimal    # sum of all confirmed fee overcharges
    total_bank_cash_at_risk_inr: Decimal  # sum of escalated bank gap cash impacts
    total_claim_amount_inr: Decimal   # claim_ready + escalated leakage

    # ── Root cause breakdown ───────────────────────────────────────────────────
    by_root_cause: dict  # {root_cause_label: {"count": int, "amount_inr": Decimal}}

    def _f(self, val: Decimal) -> str:
        return f"Rs {val.quantize(TWO_DP, rounding=ROUND_HALF_UP):,.2f}"

    def validate_invariants(self) -> list[str]:
        """
        Returns a list of invariant violations. Empty list = all good.
        Used in tests to prove the numbers add up.
        """
        errors = []
        if self.auto_matched + self.investigated != self.total_processed:
            errors.append(
                f"auto_matched({self.auto_matched}) + investigated({self.investigated}) "
                f"!= total_processed({self.total_processed})"
            )
        sub_total = self.claim_ready + self.escalated + self.monitoring + self.human_review
        if sub_total != self.investigated:
            errors.append(
                f"claim_ready({self.claim_ready}) + escalated({self.escalated}) + "
                f"monitoring({self.monitoring}) + human_review({self.human_review}) "
                f"= {sub_total} != investigated({self.investigated})"
            )
        return errors

    def as_text_report(self) -> str:
        """Human-readable batch count report for terminal output."""
        lines = [
            "",
            "=" * 70,
            "  BATCH ROUTING REPORT",
            "=" * 70,
            f"  Total transactions processed : {self.total_processed:>6}",
            f"  Total volume                 : {self._f(self.total_volume_inr):>16}",
            "-" * 70,
            f"  [AUTO-MATCHED]   Clean — no issues detected  : {self.auto_matched:>6}",
            f"  [INVESTIGATED]   Sent to AI agent pipeline   : {self.investigated:>6}",
            "-" * 70,
            "  Breakdown of investigated cases:",
            f"    Claim-ready  (approved, file claim)          : {self.claim_ready:>6}",
            f"    Escalated    (high-value, fast-track)         : {self.escalated:>6}",
            f"    Monitoring   (bank gaps, self-resolving)      : {self.monitoring:>6}",
            f"    Human review (label disagreement)             : {self.human_review:>6}",
            "-" * 70,
            f"  Total fee leakage detected   : {self._f(self.total_fee_leakage_inr):>16}",
            f"  Bank cash at risk            : {self._f(self.total_bank_cash_at_risk_inr):>16}",
            f"  TOTAL CLAIM AMOUNT           : {self._f(self.total_claim_amount_inr):>16}",
            "=" * 70,
        ]
        if self.by_root_cause:
            lines.append("  Root cause breakdown:")
            for cause, data in sorted(
                self.by_root_cause.items(),
                key=lambda x: x[1]["amount_inr"],
                reverse=True,
            ):
                amt = data["amount_inr"].quantize(TWO_DP, rounding=ROUND_HALF_UP)
                lines.append(f"    {cause:<35} {data['count']:>3} cases  Rs {amt:>10,.2f}")
            lines.append("=" * 70)
        return "\n".join(lines)


# ── Routing constants ─────────────────────────────────────────────────────────

# Bank gap types that are self-monitoring — not escalated, no claim filed
_MONITORING_GAP_TYPES = {"HOLD_PLACED_THEN_CLEARED"}

# routing_decision values from the Orchestrator that map to claim_ready bucket
_CLAIM_READY_DECISIONS = {"claim", "monitor"}

# routing_decision values from the Orchestrator that map to escalated bucket
_ESCALATED_DECISIONS = {"escalate"}


def route_batch_outcomes(
    variance_records: list[dict],
    bank_gaps: list[dict],
    reviews: list[dict],
    routing_decision: str,
    total_volume_inr: Decimal,
    total_fee_leakage_inr: Decimal,
    total_bank_cash_at_risk_inr: Decimal,
) -> BatchRoutingReport:
    """
    Compute the full batch routing report.

    Parameters
    ----------
    variance_records   : all serialised FeeVarianceRecord dicts (from run_audit)
    bank_gaps          : all serialised BankGap dicts (from run_audit)
    reviews            : Reviewer output — only the investigated subset
    routing_decision   : final routing_decision from the Orchestrator node
    total_volume_inr   : gross volume from aggregator
    total_fee_leakage_inr       : total fee leakage from aggregator
    total_bank_cash_at_risk_inr : total bank cash at risk from aggregator

    Returns
    -------
    BatchRoutingReport with validated invariants.

    Why all three data sources?
      variance_records + bank_gaps give us the FULL batch (all N transactions).
      reviews give us the INVESTIGATED subset (the agent-processed cases).
      We compute auto_matched = total - investigated, which must be >= 0.
    """
    # ── Count total processed ─────────────────────────────────────────────────
    # total_processed = all unique payment_ids seen by the deterministic engine
    all_fee_pids = {r["payment_id"] for r in variance_records}
    all_bank_pids = {g["payment_id"] for g in bank_gaps}
    total_pids = all_fee_pids | all_bank_pids
    total_processed = len(total_pids)

    # ── Count investigated ────────────────────────────────────────────────────
    investigated_pids = {r["payment_id"] for r in reviews}
    investigated = len(investigated_pids)
    auto_matched = total_processed - investigated

    # ── Sub-bucket counts for investigated ───────────────────────────────────
    # Non-escalating bank PIDs — these are monitoring cases
    monitoring_pids = {
        g["payment_id"]
        for g in bank_gaps
        if g.get("gap_type") in _MONITORING_GAP_TYPES
        and g["payment_id"] in investigated_pids
    }

    # Human review pids — those with approved=False in reviews
    human_review_pids_set = {r["payment_id"] for r in reviews if not r.get("approved", True)}

    # Remaining approved pids (not human_review, not monitoring)
    approved_pids = investigated_pids - human_review_pids_set - monitoring_pids

    if routing_decision in _ESCALATED_DECISIONS:
        escalated = len(approved_pids)
        claim_ready = 0
    else:
        escalated = 0
        claim_ready = len(approved_pids)

    monitoring = len(monitoring_pids)
    human_review_count = len(human_review_pids_set)

    # ── Root cause breakdown ───────────────────────────────────────────────────
    # Pull from reviews for the investigated subset
    by_root_cause: dict = {}
    for rev in reviews:
        cause = rev.get("llm_root_cause") or rev.get("det_root_cause") or "unknown"
        try:
            amount = Decimal(str(rev.get("financial_impact_inr", "0")))
        except Exception:
            amount = Decimal("0")
        if cause not in by_root_cause:
            by_root_cause[cause] = {"count": 0, "amount_inr": Decimal("0")}
        by_root_cause[cause]["count"] += 1
        by_root_cause[cause]["amount_inr"] += amount

    # ── Total claim amount ─────────────────────────────────────────────────────
    total_claim = (total_fee_leakage_inr + total_bank_cash_at_risk_inr).quantize(
        SIX_DP, rounding=ROUND_HALF_UP
    )

    report = BatchRoutingReport(
        total_processed=total_processed,
        total_volume_inr=total_volume_inr,
        auto_matched=auto_matched,
        investigated=investigated,
        claim_ready=claim_ready,
        escalated=escalated,
        monitoring=monitoring,
        human_review=human_review_count,
        total_fee_leakage_inr=total_fee_leakage_inr,
        total_bank_cash_at_risk_inr=total_bank_cash_at_risk_inr,
        total_claim_amount_inr=total_claim,
        by_root_cause=by_root_cause,
    )

    # Validate invariants — surface bugs loudly
    violations = report.validate_invariants()
    if violations:
        raise AssertionError(
            "BatchRoutingReport invariant violation(s):\n" + "\n".join(violations)
        )

    return report
