"""
src/agent/tools.py
==================
Tool definitions for the Investigator and Orchestrator agents.

What "tools" are in LangGraph:
  A tool is a Python function decorated with @tool that the LLM can *choose* to call.
  LangGraph's ToolNode automatically invokes them when the LLM emits a tool-call message.
  The LLM sees the tool's docstring as its description — so docstrings here are the
  LLM's instruction manual.

Tools for the Investigator Agent (3 tools):
  - get_fee_variance(payment_id)      → full FeeVarianceRecord details for a payment
  - get_bank_gap(settlement_id)       → full BankGap details for a settlement
  - get_contract_rule(rule_id)        → the ContractRule that governed the transaction

Tools for the Orchestrator Agent (2 tools):
  - get_claims_for_gateway(gateway_id) → all approved claims for a specific gateway
  - get_batch_summary_stats()          → aggregated financial figures for the whole batch

Design rules:
  - Tools receive their data store by closure (injected at graph build time),
    not by global state. This makes them testable in isolation.
  - Tools always return dicts (JSON-serialisable). Never Decimal, never date objects.
  - If a payment_id or settlement_id is not found, tools return {"error": "not found"}
    rather than raising. The LLM can reason about "not found" gracefully.
"""

from langchain_core.tools import tool
from decimal import Decimal


# ── Tool factory functions ─────────────────────────────────────────────────────
# Each factory closes over the pre-loaded data and returns the actual @tool function.
# This avoids global state and keeps tools testable.

def make_investigator_tools(
    variance_records: list[dict],
    bank_gaps:        list[dict],
    claim_items:      list[dict],
    contract_rules:   list[dict],
):
    """
    Build the three Investigator tools closed over the audit run's data.
    Call once at graph initialisation time.
    """

    var_by_pay  = {r["payment_id"]: r for r in variance_records}
    gap_by_sel  = {g["settlement_id"]: g for g in bank_gaps}
    rule_by_id  = {r["rule_id"]: r for r in contract_rules}

    @tool
    def get_fee_variance(payment_id: str) -> dict:
        """
        Retrieve the full fee variance audit record for a specific payment.
        Returns expected fee, actual fee, variance amount, payment method,
        transaction amount, merchant, gateway, and root cause notes.
        Use this when you need to understand WHY a fee discrepancy occurred.
        """
        rec = var_by_pay.get(payment_id)
        if not rec:
            return {"error": f"No fee variance record found for payment_id={payment_id}"}
        return rec

    @tool
    def get_bank_gap(settlement_id: str) -> dict:
        """
        Retrieve the bank posting gap record for a specific settlement.
        Returns gap_type, settlement_date, posting_date, net_settled_amount,
        net_posted_amount, cash_impact_inr, delay_days, and whether to escalate.
        Use this when you need to understand a cash posting anomaly at the bank level.
        """
        gap = gap_by_sel.get(settlement_id)
        if not gap:
            return {"error": f"No bank gap found for settlement_id={settlement_id}"}
        return gap

    @tool
    def get_contract_rule(rule_id: str) -> dict:
        """
        Retrieve the contract rule that governed a transaction's fee calculation.
        Returns the MDR rate, fixed fee, tax rate, payment method, card category,
        and volume tier minimum GMV.
        Use this when you need to cite specific contract terms in your explanation
        or verify whether the gateway applied the correct rate.
        """
        rule = rule_by_id.get(rule_id)
        if not rule:
            return {"error": f"No contract rule found for rule_id={rule_id}"}
        return rule

    return [get_fee_variance, get_bank_gap, get_contract_rule]


def make_orchestrator_tools(
    approved_claims: list[dict],
    batch_summary:   dict,
):
    """
    Build the two Orchestrator tools closed over the reviewed and approved claims.
    Call after the Reviewer has produced its verdicts.
    """

    @tool
    def get_claims_for_gateway(gateway_id: str) -> list[dict]:
        """
        Return all approved audit claims for a specific payment gateway.
        Use this to gather all evidence for a formal dispute letter addressed
        to a single gateway partner (e.g. GW_ALPHA or GW_BETA).
        Each claim includes claim_id, root_cause, payment_id, settlement_id,
        claim amount in INR, and the evidence text.
        """
        return [c for c in approved_claims if c.get("gateway_id") == gateway_id]

    @tool
    def get_batch_summary_stats() -> dict:
        """
        Return aggregated financial statistics for the entire audit batch.
        Includes: total volume processed, total fee leakage, total bank cash at risk,
        combined financial impact, counts by root cause, and counts by bank gap type.
        Use this to populate the executive summary section of the dispute letter.
        """
        return batch_summary

    return [get_claims_for_gateway, get_batch_summary_stats]
