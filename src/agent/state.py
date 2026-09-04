"""
src/agent/state.py
==================
Shared state schema for the LangGraph multi-agent audit graph.

The state is a TypedDict that flows through all three agents:
  Investigator → Reviewer → Orchestrator

Design rationale:
  - Immutable once set per agent pass: each agent APPENDS to lists rather than
    overwriting, so the full reasoning chain is preserved in state.
  - All financial values arrive as pre-computed dicts (serialised from FeeVarianceRecord
    and BankGap dataclasses) — the agents read them but never recompute the math.
    The math was done by the deterministic pipeline. The agents' job is explanation,
    classification verification, and routing.
  - `routing_decision` is set by the Orchestrator and drives the final output:
      'escalate'  → send to senior ops + write dispute letter
      'claim'     → generate standard claim packet
      'monitor'   → log but don't escalate
      'no_action' → clean, do nothing
"""

from typing import TypedDict, Annotated
import operator


class AuditState(TypedDict):
    # --- Input ---
    payment_ids:      list[str]   # which payments this run covers
    variance_records: list[dict]  # serialised FeeVarianceRecord dicts (from fee_variance.py)
    bank_gaps:        list[dict]  # serialised BankGap dicts (from bank_matching.py)
    claim_items:      list[dict]  # serialised ClaimItem dicts (from evidence/builder.py)

    # --- Investigator output ---
    # One entry per payment_id that has a finding
    investigations: Annotated[list[dict], operator.add]

    # --- Reviewer output ---
    # One entry per investigation: agrees or flags for human review
    reviews: Annotated[list[dict], operator.add]

    # Reviewer's batch-level routing signal — drives the conditional edge.
    # 'approved'     → all findings verified, proceed to Orchestrator
    # 'escalate'     → at least one high-confidence, high-value finding, fast-track escalation
    # 'human_review' → investigator/classifier disagreement, route to human queue
    reviewer_verdict: str

    # payment_ids flagged by Reviewer as needing human review (investigator/classifier mismatch)
    human_review_pids: Annotated[list[str], operator.add]

    # --- Orchestrator output ---
    final_claims:     list[dict]  # Orchestrator-approved claims, ready for filing
    dispute_letter:   str         # Gemini-generated formal dispute letter (markdown)
    routing_decision: str         # 'escalate' | 'claim' | 'monitor' | 'no_action'
    batch_summary:    str         # human-readable paragraph summary of the entire batch
