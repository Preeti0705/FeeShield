"""
tests/test_agents.py
====================
Unit tests for the multi-agent pipeline (Investigator → Reviewer → Orchestrator).

CRITICAL RULE (from master plan):
  Never call the live Gemini API from pytest.
  All LLM calls are mocked to return fixed, known outputs.
  This ensures tests are fast (<1s), deterministic, and never burn quota.

What we test:
  1. reviewer_node agreement logic — the core deterministic backstop
  2. _reviewer_router conditional edge function — which branch the graph takes
  3. _human_queue_node — terminal node for human review cases
  4. run_agent_audit plumbing — graph wires up and returns the right keys

What we do NOT test here:
  - Live Gemini API responses (tested by running run_agents.py manually)
  - The actual quality of LLM investigation output (subjective, not unit-testable)
  - graph.build_audit_graph() with real nodes (would call the live API)

Mocking approach:
  We mock 'investigator_node' at the graph level so the graph machinery
  (state merging, edge routing) is exercised with known fixed investigations,
  then the real reviewer_node and real _reviewer_router run on those fixtures.
  This tests that the graph's conditional routing is correct for every verdict case.
"""

import json
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.nodes import (
    reviewer_node,
    _normalise_cause,
    _det_cause_from_notes,
)
from src.agent.graph import _reviewer_router, _human_queue_node
from src.agent.state import AuditState


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — these are the "known answers" we'd get from Gemini in a real run
# ─────────────────────────────────────────────────────────────────────────────

def _make_investigation(
    payment_id: str,
    inv_type: str,
    root_cause: str,
    settlement_id: str = "",
) -> dict:
    """Build a mock Investigator output dict with a specific root cause in the JSON."""
    llm_json = json.dumps({
        "payment_id": payment_id,
        "settlement_id": settlement_id,
        "root_cause": root_cause,
        "financial_impact_inr": "35.40",
        "contract_rule_cited": "RUL001",
        "explanation": f"Gateway applied {root_cause} to this transaction.",
        "recommended_action": "claim",
    })
    return {
        "payment_id":         payment_id,
        "settlement_id":      settlement_id,
        "investigation_type": inv_type,
        "llm_output":         llm_json,
    }


def _make_variance_record(
    payment_id: str,
    notes: str,
    variance: str = "35.40",
    has_variance: bool = True,
) -> dict:
    return {
        "payment_id":       payment_id,
        "settlement_id":    f"SEL{payment_id[3:]}",
        "has_variance":     has_variance,
        "fee_variance_inr": variance,
        "notes":            notes,
        "rule_id":          "RUL001",
    }


def _make_bank_gap(
    payment_id: str,
    settlement_id: str,
    gap_type: str,
    cash_impact: str = "1500.00",
    should_escalate: bool = True,
) -> dict:
    return {
        "payment_id":     payment_id,
        "settlement_id":  settlement_id,
        "gap_type":       gap_type,
        "cash_impact_inr": cash_impact,
        "should_escalate": should_escalate,
    }


def _make_base_state(
    investigations: list[dict],
    variance_records: list[dict],
    bank_gaps: list[dict] | None = None,
    payment_ids: list[str] | None = None,
) -> AuditState:
    """Minimal valid AuditState for testing reviewer_node in isolation."""
    return {
        "payment_ids":       payment_ids or [],
        "variance_records":  variance_records,
        "bank_gaps":         bank_gaps or [],
        "claim_items":       [],
        "investigations":    investigations,
        "reviews":           [],
        "reviewer_verdict":  "",
        "human_review_pids": [],
        "final_claims":      [],
        "dispute_letter":    "",
        "routing_decision":  "",
        "batch_summary":     "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tests for _normalise_cause and _det_cause_from_notes (pure functions)
# ─────────────────────────────────────────────────────────────────────────────

class TestNormaliseCause:
    def test_known_fee_cause(self):
        assert _normalise_cause("wrong_mdr") == "wrong_mdr"

    def test_bank_cause_uppercase(self):
        assert _normalise_cause("SETTLEMENT_NOT_POSTED") == "SETTLEMENT_NOT_POSTED"

    def test_bank_cause_lowercase_normalised(self):
        assert _normalise_cause("settlement_not_posted") == "SETTLEMENT_NOT_POSTED"

    def test_none_returns_none(self):
        assert _normalise_cause(None) is None

    def test_empty_string_returns_none(self):
        assert _normalise_cause("") is None

    def test_unknown_string_returned_as_is(self):
        assert _normalise_cause("totally_made_up") == "totally_made_up"


class TestDetCauseFromNotes:
    def test_wrong_mdr(self):
        assert _det_cause_from_notes("wrong_mdr rate applied") == "wrong_mdr"

    def test_volume_tier(self):
        assert _det_cause_from_notes("missed_volume_tier for MER001") == "missed_volume_tier"

    def test_tax_base(self):
        assert _det_cause_from_notes("wrong_tax_base applied") == "wrong_tax_base"

    def test_duplicate_fee(self):
        assert _det_cause_from_notes("duplicate fee deduction") == "duplicate_fee"

    def test_version_violation(self):
        assert _det_cause_from_notes("contract_version_violation: v1 used post cutover") == "contract_version_violation"

    def test_version_via_v1_keyword(self):
        assert _det_cause_from_notes("gateway used v1 rate") == "contract_version_violation"

    def test_no_match_returns_none(self):
        assert _det_cause_from_notes("clean transaction") is None

    def test_empty_notes(self):
        assert _det_cause_from_notes("") is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tests for reviewer_node agreement logic
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewerNodeAgreement:
    """
    Tests the core deterministic agreement logic.
    No LLM is involved — we control what the "LLM output" JSON says.
    """

    def test_agree_when_labels_match_wrong_mdr(self):
        """LLM says wrong_mdr, classifier notes say wrong_mdr → agree."""
        state = _make_base_state(
            investigations=[_make_investigation("PAY001", "fee_variance", "wrong_mdr")],
            variance_records=[_make_variance_record("PAY001", "wrong_mdr rate applied")],
        )
        result = reviewer_node(state)

        assert len(result["reviews"]) == 1
        review = result["reviews"][0]
        assert review["reviewer_agreement"] == "agree"
        assert review["approved"] is True
        assert review["llm_root_cause"] == "wrong_mdr"
        assert review["det_root_cause"] == "wrong_mdr"

    def test_agree_missed_volume_tier(self):
        """LLM says missed_volume_tier, classifier notes say missed_volume_tier → agree."""
        state = _make_base_state(
            investigations=[_make_investigation("PAY002", "fee_variance", "missed_volume_tier")],
            variance_records=[_make_variance_record("PAY002", "missed_volume_tier: gmv > 500000")],
        )
        result = reviewer_node(state)
        assert result["reviews"][0]["reviewer_agreement"] == "agree"
        assert result["reviews"][0]["approved"] is True

    def test_agree_wrong_tax_base(self):
        state = _make_base_state(
            investigations=[_make_investigation("PAY003", "fee_variance", "wrong_tax_base")],
            variance_records=[_make_variance_record("PAY003", "wrong_tax_base on gross txn amount")],
        )
        result = reviewer_node(state)
        assert result["reviews"][0]["reviewer_agreement"] == "agree"

    def test_agree_duplicate_fee(self):
        state = _make_base_state(
            investigations=[_make_investigation("PAY004", "fee_variance", "duplicate_fee")],
            variance_records=[_make_variance_record("PAY004", "duplicate fee deduction for second settlement row")],
        )
        result = reviewer_node(state)
        assert result["reviews"][0]["reviewer_agreement"] == "agree"

    def test_agree_contract_version_violation(self):
        state = _make_base_state(
            investigations=[_make_investigation("PAY008", "fee_variance", "contract_version_violation")],
            variance_records=[_make_variance_record("PAY008", "contract_version_violation: gateway used v1 after cutover")],
        )
        result = reviewer_node(state)
        assert result["reviews"][0]["reviewer_agreement"] == "agree"

    def test_disagree_when_labels_differ(self):
        """LLM says wrong_mdr but classifier notes say duplicate_fee → disagree."""
        state = _make_base_state(
            investigations=[_make_investigation("PAY099", "fee_variance", "wrong_mdr")],
            variance_records=[_make_variance_record("PAY099", "duplicate fee deduction")],
        )
        result = reviewer_node(state)
        review = result["reviews"][0]
        assert review["reviewer_agreement"] == "disagree"
        assert review["approved"] is False
        assert "PAY099" in result["human_review_pids"]

    def test_no_label_when_llm_output_is_empty(self):
        """LLM returns empty string (quota exhausted fallback had no JSON) → no_label."""
        state = _make_base_state(
            investigations=[{
                "payment_id": "PAY010",
                "settlement_id": "SEL010",
                "investigation_type": "fee_variance",
                "llm_output": "",   # nothing parseable
            }],
            variance_records=[_make_variance_record("PAY010", "wrong_mdr")],
        )
        result = reviewer_node(state)
        review = result["reviews"][0]
        assert review["reviewer_agreement"] == "no_label"
        assert review["approved"] is False
        assert "PAY010" in result["human_review_pids"]

    def test_bank_gap_agree(self):
        """Investigator says SETTLEMENT_NOT_POSTED, bank_gap dict has same gap_type → agree."""
        state = _make_base_state(
            investigations=[_make_investigation(
                "PAY100", "bank_gap", "SETTLEMENT_NOT_POSTED", settlement_id="SEL100"
            )],
            variance_records=[],
            bank_gaps=[_make_bank_gap("PAY100", "SEL100", "SETTLEMENT_NOT_POSTED")],
        )
        result = reviewer_node(state)
        review = result["reviews"][0]
        assert review["reviewer_agreement"] == "agree"
        assert review["approved"] is True

    def test_bank_gap_disagree(self):
        """Investigator says POSTING_REVERSED but bank record says SETTLEMENT_POSTED_LATE → disagree."""
        state = _make_base_state(
            investigations=[_make_investigation(
                "PAY101", "bank_gap", "POSTING_REVERSED", settlement_id="SEL101"
            )],
            variance_records=[],
            bank_gaps=[_make_bank_gap("PAY101", "SEL101", "SETTLEMENT_POSTED_LATE")],
        )
        result = reviewer_node(state)
        review = result["reviews"][0]
        assert review["reviewer_agreement"] == "disagree"
        assert review["approved"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tests for reviewer_verdict (batch routing signal)
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewerVerdict:
    """
    Tests the batch-level verdict that drives the conditional edge.
    Formula:
      ALL agree                              → 'approved'
      ANY disagree/no_label, low value       → 'human_review'
      ANY disagree/no_label, value > Rs 1000 → 'escalate'
    """

    def test_all_agree_gives_approved(self):
        state = _make_base_state(
            investigations=[
                _make_investigation("PAY001", "fee_variance", "wrong_mdr"),
                _make_investigation("PAY002", "fee_variance", "missed_volume_tier"),
            ],
            variance_records=[
                _make_variance_record("PAY001", "wrong_mdr"),
                _make_variance_record("PAY002", "missed_volume_tier"),
            ],
        )
        result = reviewer_node(state)
        assert result["reviewer_verdict"] == "approved"
        assert result["human_review_pids"] == []

    def test_one_disagree_low_value_gives_human_review(self):
        """Disagreement on a Rs 35.40 case → human_review (below Rs 1,000 threshold)."""
        state = _make_base_state(
            investigations=[
                _make_investigation("PAY001", "fee_variance", "wrong_mdr"),    # agree
                _make_investigation("PAY002", "fee_variance", "wrong_mdr"),    # disagree (notes say duplicate)
            ],
            variance_records=[
                _make_variance_record("PAY001", "wrong_mdr", variance="35.40"),
                _make_variance_record("PAY002", "duplicate fee deduction", variance="35.40"),
            ],
        )
        result = reviewer_node(state)
        assert result["reviewer_verdict"] == "human_review"
        assert "PAY002" in result["human_review_pids"]

    def test_disagree_high_value_gives_escalate(self):
        """Disagreement on a Rs 3,542.40 case → escalate (above Rs 1,000 threshold)."""
        state = _make_base_state(
            investigations=[
                _make_investigation("PAY003", "fee_variance", "wrong_mdr"),  # disagree: notes=wrong_tax_base
            ],
            variance_records=[
                _make_variance_record("PAY003", "wrong_tax_base applied on gross", variance="3542.40"),
            ],
        )
        result = reviewer_node(state)
        assert result["reviewer_verdict"] == "escalate"
        assert "PAY003" in result["human_review_pids"]

    def test_empty_investigations_gives_approved(self):
        """No investigations → nothing to disagree about → approved."""
        state = _make_base_state(investigations=[], variance_records=[])
        result = reviewer_node(state)
        assert result["reviewer_verdict"] == "approved"
        assert result["human_review_pids"] == []


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tests for _reviewer_router (conditional edge function)
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewerRouter:
    """
    Tests the function that drives the conditional edge in graph.py.
    _reviewer_router(state) → 'orchestrator' | 'human_queue'
    """

    def _state_with_verdict(self, verdict: str) -> AuditState:
        return {
            "payment_ids": [],
            "variance_records": [],
            "bank_gaps": [],
            "claim_items": [],
            "investigations": [],
            "reviews": [],
            "reviewer_verdict":  verdict,
            "human_review_pids": [],
            "final_claims": [],
            "dispute_letter": "",
            "routing_decision": "",
            "batch_summary": "",
        }

    def test_approved_routes_to_orchestrator(self):
        assert _reviewer_router(self._state_with_verdict("approved")) == "orchestrator"

    def test_escalate_routes_to_orchestrator(self):
        # High-value disagreement still needs a dispute letter → orchestrator
        assert _reviewer_router(self._state_with_verdict("escalate")) == "orchestrator"

    def test_human_review_routes_to_human_queue(self):
        assert _reviewer_router(self._state_with_verdict("human_review")) == "human_queue"

    def test_empty_verdict_defaults_to_human_queue(self):
        # If reviewer_verdict is somehow empty, the safe default is human_queue
        # (never auto-file a claim without an explicit 'approved' or 'escalate' verdict)
        assert _reviewer_router(self._state_with_verdict("")) == "human_queue"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tests for _human_queue_node
# ─────────────────────────────────────────────────────────────────────────────

class TestHumanQueueNode:
    def test_no_dispute_letter_filed(self):
        from src.agent.graph import _human_queue_node
        state = {
            "payment_ids": ["PAY099"],
            "variance_records": [],
            "bank_gaps": [],
            "claim_items": [],
            "investigations": [],
            "reviews": [],
            "reviewer_verdict": "human_review",
            "human_review_pids": ["PAY099", "PAY101"],
            "final_claims": [],
            "dispute_letter": "",
            "routing_decision": "",
            "batch_summary": "",
        }
        result = _human_queue_node(state)
        assert result["routing_decision"] == "human_review"
        assert result["dispute_letter"] == ""
        assert result["final_claims"] == []
        assert "PAY099" in result["batch_summary"]
        assert "PAY101" in result["batch_summary"]

    def test_routing_decision_is_human_review(self):
        from src.agent.graph import _human_queue_node
        state = {
            "payment_ids": [],
            "variance_records": [], "bank_gaps": [], "claim_items": [],
            "investigations": [], "reviews": [],
            "reviewer_verdict": "human_review",
            "human_review_pids": [],
            "final_claims": [], "dispute_letter": "", "routing_decision": "", "batch_summary": "",
        }
        result = _human_queue_node(state)
        assert result["routing_decision"] == "human_review"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Integration: graph with mocked investigator_node
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphWithMockedInvestigator:
    """
    Tests the full graph plumbing with the Investigator mocked to return
    a known investigation.  The real reviewer_node and _reviewer_router run
    so we test actual conditional routing behaviour end-to-end.

    Why mock investigator_node, not the LLM?
      Mocking at the node level is simpler and faster.  The Investigator's
      internal LLM logic is already tested separately by the manual smoke test.
      Here we want to verify that the STATE flows correctly through the graph
      and the CONDITIONAL EDGE fires correctly.
    """

    def _mock_inv_node(self, fixed_investigations: list[dict]):
        """Return a function that pretends to be investigator_node."""
        def _node(state, contract_rules_dicts=None):
            return {"investigations": fixed_investigations}
        return _node

    def _mock_orch_node(self):
        def _node(state, approved_claims=None, batch_summary_stats=None):
            return {
                "final_claims":     approved_claims or [],
                "dispute_letter":   "# MOCK DISPUTE LETTER",
                "routing_decision": "claim",
                "batch_summary":    "Mock orchestrator ran.",
            }
        return _node

    def _build_test_state(self, variance_records: list[dict]) -> AuditState:
        return {
            "payment_ids":       [r["payment_id"] for r in variance_records],
            "variance_records":  variance_records,
            "bank_gaps":         [],
            "claim_items":       [],
            "investigations":    [],
            "reviews":           [],
            "reviewer_verdict":  "",
            "human_review_pids": [],
            "final_claims":      [],
            "dispute_letter":    "",
            "routing_decision":  "",
            "batch_summary":     "",
        }

    def test_all_agree_reaches_orchestrator(self):
        """
        Graph path: investigator → reviewer (approved) → orchestrator → END
        Expected: routing_decision='claim', dispute_letter has content.
        """
        from functools import partial
        from langgraph.graph import StateGraph, START, END
        from src.agent.nodes import reviewer_node
        from src.agent.graph import _reviewer_router, _human_queue_node

        # Build a minimal graph with mocked nodes
        builder = StateGraph(AuditState)
        builder.add_node("investigator", self._mock_inv_node([
            _make_investigation("PAY001", "fee_variance", "wrong_mdr"),
        ]))
        builder.add_node("reviewer",     reviewer_node)
        builder.add_node("orchestrator", self._mock_orch_node())
        builder.add_node("human_queue",  _human_queue_node)
        builder.add_edge(START, "investigator")
        builder.add_edge("investigator", "reviewer")
        builder.add_conditional_edges("reviewer", _reviewer_router, {
            "orchestrator": "orchestrator",
            "human_queue":  "human_queue",
        })
        builder.add_edge("orchestrator", END)
        builder.add_edge("human_queue",  END)
        graph = builder.compile()

        initial = self._build_test_state([
            _make_variance_record("PAY001", "wrong_mdr rate applied"),
        ])
        result = graph.invoke(initial)

        assert result["routing_decision"] == "claim"
        assert result["dispute_letter"] == "# MOCK DISPUTE LETTER"
        assert result["human_review_pids"] == []

    def test_disagree_routes_to_human_queue(self):
        """
        Graph path: investigator → reviewer (human_review) → human_queue → END
        Expected: routing_decision='human_review', no dispute letter.
        """
        from langgraph.graph import StateGraph, START, END
        from src.agent.nodes import reviewer_node
        from src.agent.graph import _reviewer_router, _human_queue_node

        builder = StateGraph(AuditState)
        builder.add_node("investigator", self._mock_inv_node([
            # LLM says wrong_mdr but classifier notes say duplicate_fee → disagree
            _make_investigation("PAY099", "fee_variance", "wrong_mdr"),
        ]))
        builder.add_node("reviewer",     reviewer_node)
        builder.add_node("orchestrator", self._mock_orch_node())
        builder.add_node("human_queue",  _human_queue_node)
        builder.add_edge(START, "investigator")
        builder.add_edge("investigator", "reviewer")
        builder.add_conditional_edges("reviewer", _reviewer_router, {
            "orchestrator": "orchestrator",
            "human_queue":  "human_queue",
        })
        builder.add_edge("orchestrator", END)
        builder.add_edge("human_queue",  END)
        graph = builder.compile()

        initial = self._build_test_state([
            _make_variance_record("PAY099", "duplicate fee deduction", variance="35.40"),
        ])
        result = graph.invoke(initial)

        assert result["routing_decision"] == "human_review"
        assert result["dispute_letter"] == ""
        assert "PAY099" in result["human_review_pids"]


# ─────────────────────────────────────────────────────────────────────────────
# Day 7: Tests for route_batch_outcomes and BatchRoutingReport
# ─────────────────────────────────────────────────────────────────────────────

from decimal import Decimal
from src.agent.orchestrator import route_batch_outcomes, BatchRoutingReport


def _make_report(
    variance_records=None,
    bank_gaps=None,
    reviews=None,
    routing_decision="claim",
    total_volume=Decimal("100000"),
    total_fee_leakage=Decimal("500"),
    total_bank_risk=Decimal("0"),
) -> BatchRoutingReport:
    """Helper to build a BatchRoutingReport with sensible defaults."""
    return route_batch_outcomes(
        variance_records=variance_records or [],
        bank_gaps=bank_gaps or [],
        reviews=reviews or [],
        routing_decision=routing_decision,
        total_volume_inr=total_volume,
        total_fee_leakage_inr=total_fee_leakage,
        total_bank_cash_at_risk_inr=total_bank_risk,
    )


def _approved_review(payment_id: str, cause: str = "wrong_mdr", impact: str = "35.40") -> dict:
    return {
        "payment_id": payment_id,
        "settlement_id": f"SEL{payment_id[3:]}",
        "investigation_type": "fee_variance",
        "llm_root_cause": cause,
        "det_root_cause": cause,
        "reviewer_agreement": "agree",
        "confidence_score": 95,
        "approved": True,
        "financial_impact_inr": impact,
        "llm_summary": "",
    }


def _rejected_review(payment_id: str, cause: str = "wrong_mdr", impact: str = "35.40") -> dict:
    return {
        "payment_id": payment_id,
        "settlement_id": f"SEL{payment_id[3:]}",
        "investigation_type": "fee_variance",
        "llm_root_cause": cause,
        "det_root_cause": "duplicate_fee",  # mismatch
        "reviewer_agreement": "disagree",
        "confidence_score": 60,
        "approved": False,
        "financial_impact_inr": impact,
        "llm_summary": "",
    }


def _var_record(payment_id: str, has_variance: bool = True) -> dict:
    return {
        "payment_id": payment_id,
        "settlement_id": f"SEL{payment_id[3:]}",
        "has_variance": has_variance,
        "fee_variance_inr": "35.40",
        "notes": "wrong_mdr",
        "rule_id": "RUL001",
    }


def _bank_gap_dict(payment_id: str, gap_type: str, escalate: bool = True) -> dict:
    return {
        "payment_id": payment_id,
        "settlement_id": f"SEL{payment_id[3:]}",
        "gap_type": gap_type,
        "cash_impact_inr": "5000.00",
        "should_escalate": escalate,
    }


class TestBatchRoutingReportInvariants:
    """
    The master plan requires: numbers must sum correctly.
    We enforce this via validate_invariants() which returns a list of violations.
    An empty list means the report is internally consistent.

    Invariant 1: auto_matched + investigated == total_processed
    Invariant 2: claim_ready + escalated + monitoring + human_review == investigated
    """

    def test_empty_batch_has_no_violations(self):
        report = _make_report()
        assert report.validate_invariants() == []
        assert report.total_processed == 0
        assert report.investigated == 0
        assert report.auto_matched == 0

    def test_all_clean_transactions(self):
        """10 payments, all passed deterministic engine cleanly, none investigated."""
        variance_records = [_var_record(f"PAY{i:03d}", has_variance=False) for i in range(1, 11)]
        report = _make_report(variance_records=variance_records)
        assert report.total_processed == 10
        assert report.investigated == 0
        assert report.auto_matched == 10
        assert report.validate_invariants() == []

    def test_all_approved_claim_routing(self):
        """5 payments all investigated and approved with routing_decision='claim'."""
        var_records = [_var_record(f"PAY{i:03d}") for i in range(1, 6)]
        reviews = [_approved_review(f"PAY{i:03d}") for i in range(1, 6)]
        report = _make_report(variance_records=var_records, reviews=reviews, routing_decision="claim")
        assert report.total_processed == 5
        assert report.investigated == 5
        assert report.auto_matched == 0
        assert report.claim_ready == 5
        assert report.escalated == 0
        assert report.human_review == 0
        assert report.validate_invariants() == []

    def test_all_approved_escalate_routing(self):
        """5 investigations all approved but escalate routing."""
        var_records = [_var_record(f"PAY{i:03d}") for i in range(1, 6)]
        reviews = [_approved_review(f"PAY{i:03d}") for i in range(1, 6)]
        report = _make_report(
            variance_records=var_records, reviews=reviews, routing_decision="escalate"
        )
        assert report.escalated == 5
        assert report.claim_ready == 0
        assert report.validate_invariants() == []

    def test_mixed_approved_and_human_review(self):
        """3 approved + 2 rejected → 3 claim_ready, 2 human_review."""
        var_records = [_var_record(f"PAY{i:03d}") for i in range(1, 6)]
        reviews = (
            [_approved_review(f"PAY{i:03d}") for i in range(1, 4)]
            + [_rejected_review(f"PAY{i:03d}") for i in range(4, 6)]
        )
        report = _make_report(variance_records=var_records, reviews=reviews, routing_decision="claim")
        assert report.claim_ready == 3
        assert report.human_review == 2
        assert report.investigated == 5
        assert report.validate_invariants() == []

    def test_monitoring_bank_gap_counted_correctly(self):
        """HOLD_PLACED_THEN_CLEARED bank gap investigated → goes into monitoring bucket."""
        var_records = [_var_record("PAY001")]
        bank_gaps = [_bank_gap_dict("PAY200", "HOLD_PLACED_THEN_CLEARED", escalate=False)]
        # Both PAY001 (fee) and PAY200 (bank) were investigated
        reviews = [
            _approved_review("PAY001"),
            {
                "payment_id": "PAY200",
                "settlement_id": "SEL200",
                "investigation_type": "bank_gap",
                "llm_root_cause": "HOLD_PLACED_THEN_CLEARED",
                "det_root_cause": "HOLD_PLACED_THEN_CLEARED",
                "reviewer_agreement": "agree",
                "confidence_score": 90,
                "approved": True,
                "financial_impact_inr": "0",
                "llm_summary": "",
            },
        ]
        report = _make_report(
            variance_records=var_records,
            bank_gaps=bank_gaps,
            reviews=reviews,
            routing_decision="claim",
        )
        assert report.investigated == 2
        assert report.monitoring == 1   # PAY200 — HOLD_PLACED_THEN_CLEARED
        assert report.claim_ready == 1  # PAY001 — fee approved
        assert report.validate_invariants() == []

    def test_mixed_full_batch(self):
        """
        10 total payments: 5 clean, 3 approved (claim), 1 rejected (human_review), 1 monitoring.
        Invariants must hold.
        """
        clean = [_var_record(f"PAY{i:03d}", has_variance=False) for i in range(1, 6)]
        flagged_fee = [_var_record(f"PAY{i:03d}", has_variance=True) for i in range(6, 9)]
        rejected_fee = [_var_record("PAY009", has_variance=True)]
        bank_gaps = [_bank_gap_dict("PAY010", "HOLD_PLACED_THEN_CLEARED", escalate=False)]

        reviews = (
            [_approved_review(f"PAY{i:03d}") for i in range(6, 9)]
            + [_rejected_review("PAY009")]
            + [{
                "payment_id": "PAY010",
                "settlement_id": "SEL010",
                "investigation_type": "bank_gap",
                "llm_root_cause": "HOLD_PLACED_THEN_CLEARED",
                "det_root_cause": "HOLD_PLACED_THEN_CLEARED",
                "reviewer_agreement": "agree",
                "confidence_score": 90,
                "approved": True,
                "financial_impact_inr": "0",
                "llm_summary": "",
            }]
        )

        all_var = clean + flagged_fee + rejected_fee
        report = _make_report(
            variance_records=all_var,
            bank_gaps=bank_gaps,
            reviews=reviews,
            routing_decision="claim",
        )
        assert report.total_processed == 10
        assert report.auto_matched == 5
        assert report.investigated == 5
        assert report.claim_ready == 3
        assert report.human_review == 1
        assert report.monitoring == 1
        assert report.escalated == 0
        assert report.validate_invariants() == []


class TestBatchRoutingReportFinancials:
    """Tests for the financial totals in BatchRoutingReport."""

    def test_total_claim_amount_is_fee_plus_bank_risk(self):
        report = _make_report(
            total_fee_leakage=Decimal("6458.24"),
            total_bank_risk=Decimal("28965.46"),
        )
        expected = Decimal("35423.70")
        assert report.total_claim_amount_inr == expected

    def test_zero_bank_risk_total_equals_fee_leakage(self):
        report = _make_report(
            total_fee_leakage=Decimal("500.00"),
            total_bank_risk=Decimal("0"),
        )
        assert report.total_claim_amount_inr == Decimal("500.000000")

    def test_root_cause_breakdown_sums_by_cause(self):
        """Multiple investigations with same cause should aggregate."""
        var_records = [_var_record(f"PAY{i:03d}") for i in range(1, 4)]
        reviews = [
            _approved_review("PAY001", cause="wrong_mdr", impact="35.40"),
            _approved_review("PAY002", cause="wrong_mdr", impact="53.10"),
            _approved_review("PAY003", cause="wrong_tax_base", impact="3542.40"),
        ]
        report = _make_report(variance_records=var_records, reviews=reviews)
        assert "wrong_mdr" in report.by_root_cause
        assert report.by_root_cause["wrong_mdr"]["count"] == 2
        assert report.by_root_cause["wrong_mdr"]["amount_inr"] == Decimal("88.50")
        assert report.by_root_cause["wrong_tax_base"]["count"] == 1


class TestBatchRoutingReportTextOutput:
    """Tests for the as_text_report() method — ensures invariant values appear in output."""

    def test_report_contains_total_processed(self):
        var_records = [_var_record(f"PAY{i:03d}") for i in range(1, 6)]
        report = _make_report(variance_records=var_records)
        text = report.as_text_report()
        assert "5" in text  # total_processed

    def test_report_contains_claim_amount(self):
        report = _make_report(
            total_fee_leakage=Decimal("6458.24"),
            total_bank_risk=Decimal("28965.46"),
        )
        text = report.as_text_report()
        assert "35,423.70" in text

    def test_report_has_batch_routing_header(self):
        report = _make_report()
        text = report.as_text_report()
        assert "BATCH ROUTING REPORT" in text
