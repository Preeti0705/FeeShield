"""
src/agent/graph.py
==================
Builds and compiles the LangGraph multi-agent audit graph.

Graph topology:
  START → investigator → reviewer → orchestrator → END

Why this linear topology (not parallel)?
  Each agent depends on the previous agent's output:
  - Reviewer needs Investigator's findings to verify them.
  - Orchestrator needs Reviewer's verdicts to know which claims are approved.
  A parallel topology makes no sense here — this is a sequential verification chain.

Why LangGraph instead of a hand-written loop?
  LangGraph gives us:
  1. State persistence — every step's output is preserved, inspectable, and replayable.
  2. Checkpointing — if the LLM call fails mid-run, LangGraph can resume from the last checkpoint.
  3. Streaming — the graph can stream partial outputs so a UI can show progress live.
  4. Visualisation — graph.get_graph().draw_mermaid() gives an automatic flowchart.

How to use:
  from src.agent.graph import build_audit_graph, run_agent_audit
  result = run_agent_audit(variance_records, bank_gaps, claim_items, contract_rules_dicts, batch_stats)
  print(result["dispute_letter"])
  print(result["routing_decision"])
"""

from functools import partial
from langgraph.graph import StateGraph, START, END

from src.agent.state import AuditState
from src.agent.nodes import investigator_node, reviewer_node, orchestrator_node


def build_audit_graph(
    contract_rules_dicts: list[dict],
    approved_claims:      list[dict],
    batch_summary_stats:  dict,
):
    """
    Build and compile the three-agent audit graph.

    Parameters
    ----------
    contract_rules_dicts  : list of ContractRule as dicts (for Investigator tool)
    approved_claims       : list of ClaimItem dicts (for Orchestrator tool)
    batch_summary_stats   : dict from aggregator.py (for Orchestrator tool)

    Returns
    -------
    Compiled LangGraph graph ready to invoke.
    """
    builder = StateGraph(AuditState)

    # Wrap nodes with their closed-over data using functools.partial
    inv_node   = partial(investigator_node, contract_rules_dicts=contract_rules_dicts)
    orch_node  = partial(orchestrator_node,
                         approved_claims=approved_claims,
                         batch_summary_stats=batch_summary_stats)

    # Add nodes
    builder.add_node("investigator", inv_node)
    builder.add_node("reviewer",     reviewer_node)
    builder.add_node("orchestrator", orch_node)

    # Add edges: linear chain
    builder.add_edge(START,          "investigator")
    builder.add_edge("investigator", "reviewer")
    builder.add_edge("reviewer",     "orchestrator")
    builder.add_edge("orchestrator", END)

    return builder.compile()


def run_agent_audit(
    variance_records:     list[dict],
    bank_gaps:            list[dict],
    claim_items:          list[dict],
    contract_rules_dicts: list[dict],
    batch_summary_stats:  dict,
    payment_ids:          list[str] | None = None,
) -> dict:
    """
    Convenience runner: builds the graph, initialises state, invokes, returns final state.

    Parameters
    ----------
    variance_records     : serialised FeeVarianceRecord dicts
    bank_gaps            : serialised BankGap dicts
    claim_items          : serialised ClaimItem dicts
    contract_rules_dicts : ContractRule as plain dicts for the Investigator tool
    batch_summary_stats  : summary dict from aggregator.aggregate_audit_results()
    payment_ids          : optional filter; if None, all flagged payments are processed

    Returns
    -------
    Final AuditState dict with dispute_letter, routing_decision, reviews, etc.
    """
    # Approved claims = fee claims + escalated bank claims from claim_items
    approved_claims = [c for c in claim_items if c.get("confidence_score", 0) >= 85]

    graph = build_audit_graph(
        contract_rules_dicts=contract_rules_dicts,
        approved_claims=approved_claims,
        batch_summary_stats=batch_summary_stats,
    )

    # Filter to just flagged payments if payment_ids list provided
    if payment_ids is None:
        payment_ids = list({r["payment_id"] for r in variance_records if r.get("has_variance")} |
                          {g["payment_id"] for g in bank_gaps if g.get("should_escalate")})

    initial_state: AuditState = {
        "payment_ids":      payment_ids,
        "variance_records": variance_records,
        "bank_gaps":        bank_gaps,
        "claim_items":      claim_items,
        "investigations":   [],
        "reviews":          [],
        "final_claims":     [],
        "dispute_letter":   "",
        "routing_decision": "",
        "batch_summary":    "",
    }

    final_state = graph.invoke(initial_state)
    return final_state
