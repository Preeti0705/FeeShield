"""
dashboard.py — FeeShield AI Finance Controller
================================================
Day 9 Streamlit dashboard — clean, no sidebar.

4 tabs:
  1. Executive Summary  — KPI cards + leakage/gap charts
  2. Fee Analysis        — Filterable variance table + root cause chart
  3. Bank Reconciliation — Bank gap table + gap breakdown
  4. Cash Position       — SAFE / AT_RISK / CRITICAL treasury verdict

Usage:
  $env:PYTHONUTF8 = "1"
  .venv\\Scripts\\streamlit.exe run dashboard.py
"""

import csv
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
TWO_DP = Decimal("0.01")
SIX_DP = Decimal("0.000001")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FeeShield — AI Finance Controller",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* hide sidebar toggle & branding */
[data-testid="collapsedControl"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* header */
.fs-header {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0284c7 100%);
  padding: 1.6rem 2rem 1.3rem;
  border-radius: 14px;
  margin-bottom: 1.4rem;
}
.fs-header h1 { color:#f0f9ff; font-size:1.9rem; font-weight:700; margin:0; letter-spacing:-0.4px; }
.fs-header p  { color:#7dd3fc; font-size:0.85rem; margin:0.25rem 0 0; }

/* KPI card */
.kpi { background:linear-gradient(145deg,#1e293b,#0f172a); border:1px solid #334155; border-radius:12px; padding:1.1rem 1.3rem; text-align:center; }
.kpi-label { color:#94a3b8; font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.35rem; }
.kpi-value { font-size:1.5rem; font-weight:700; line-height:1; }
.kpi-sub   { color:#64748b; font-size:0.7rem; margin-top:.25rem; }
.kpi-danger  .kpi-value { color:#f87171; }
.kpi-warning .kpi-value { color:#fbbf24; }
.kpi-info    .kpi-value { color:#60a5fa; }
.kpi-neutral .kpi-value { color:#f1f5f9; }

/* section heading */
.sh { font-size:.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.1em;
      color:#475569; margin:1.1rem 0 .45rem; padding-bottom:.25rem; border-bottom:1px solid #1e293b; }

/* verdict banners */
.v-safe     { background:#064e3b; border:1.5px solid #10b981; border-radius:10px; padding:.9rem 1.3rem; }
.v-at-risk  { background:#451a03; border:1.5px solid #f59e0b; border-radius:10px; padding:.9rem 1.3rem; }
.v-critical { background:#450a0a; border:1.5px solid #ef4444; border-radius:10px; padding:.9rem 1.3rem; }
.v-title { font-size:1.05rem; font-weight:700; margin:0; }
.v-safe .v-title     { color:#34d399; }
.v-at-risk .v-title  { color:#fbbf24; }
.v-critical .v-title { color:#f87171; }
.v-body { color:#e2e8f0; font-size:.83rem; margin-top:.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _rs(v) -> str:
    try:
        d = Decimal(str(v).replace(",", "")).quantize(TWO_DP, rounding=ROUND_HALF_UP)
        return f"Rs {d:,.2f}"
    except Exception:
        return str(v)

def _d(v) -> Decimal:
    """Safe Decimal from anything — strips commas."""
    try:
        return Decimal(str(v).replace(",", "")).quantize(SIX_DP, rounding=ROUND_HALF_UP)
    except Exception:
        return Decimal("0")

# ── Audit pipeline (cached) ────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Running audit pipeline…")
def load_audit():
    from src.contracts.resolver import load_contracts, load_contract_rules
    from src.audit.fee_variance import audit_fee_variances
    from src.reconciliation.bank_matching import (
        load_settlements, load_bank_feed, match_settlements_to_bank
    )
    from src.rootcause.classifier import classify_fee_root_cause
    from src.confidence.scorer import compute_confidence_score
    from src.evidence.builder import build_fee_claim_item, build_bank_claim_item
    from src.leakage.aggregator import aggregate_audit_results

    contracts = load_contracts(DATA / "contracts.csv")
    rules     = load_contract_rules(DATA / "contract_rules.csv")

    payments_raw    = list(csv.DictReader(open(DATA / "payments.csv",    encoding="utf-8")))
    settlements_raw = list(csv.DictReader(open(DATA / "settlements.csv", encoding="utf-8")))
    settlements_t   = load_settlements(DATA / "settlements.csv")
    bank_feed_t     = load_bank_feed(DATA / "bank_feed.csv")

    variance_recs = audit_fee_variances(payments_raw, settlements_raw, contracts, rules)
    bank_gaps     = match_settlements_to_bank(settlements_t, bank_feed_t)

    classifications, claim_items = [], []
    idx = 1
    pay_map = {p["payment_id"]: p for p in payments_raw}

    for rec in variance_recs:
        pay_row = pay_map.get(rec.payment_id, {})
        gmv     = _d(pay_row.get("monthly_gmv", "0"))
        diag    = classify_fee_root_cause(rec, contracts, rules, monthly_gmv=gmv)
        classifications.append(diag)
        if rec.has_variance and rec.fee_variance_inr > Decimal("0"):
            conf = compute_confidence_score(diag.case_type, rec.fee_variance_inr)
            claim_items.append(build_fee_claim_item(f"CLM-FEE-{idx:04d}", rec, diag, conf))
            idx += 1

    for gap in bank_gaps:
        if gap.should_escalate:
            conf    = compute_confidence_score(gap.gap_type, gap.cash_impact_inr)
            pay_row = pay_map.get(gap.payment_id, {})
            claim_items.append(build_bank_claim_item(
                f"CLM-BNK-{idx:04d}", gap,
                pay_row.get("merchant_id", "MER001"),
                pay_row.get("gateway_id",  "GW_ALPHA"),
                conf,
            ))
            idx += 1

    summary = aggregate_audit_results(variance_recs, classifications, bank_gaps)

    return {
        "variance_recs":  variance_recs,
        "bank_gaps":      bank_gaps,
        "claim_items":    claim_items,
        "classifications": classifications,
        "summary":        summary,
    }


@st.cache_data(ttl=300)
def variance_df(variance_recs, classifications):
    class_map = {c.payment_id: c for c in classifications}
    rows = []
    for r in variance_recs:
        if r.has_variance and r.fee_variance_inr > Decimal("0"):
            diag = class_map.get(r.payment_id)
            rows.append({
                "Payment ID":    r.payment_id,
                "Merchant":      r.merchant_id,
                "Gateway":       r.gateway_id,
                "Expected (Rs)": float(r.expected_total_fee.quantize(TWO_DP, rounding=ROUND_HALF_UP)),
                "Charged (Rs)":  float(r.actual_total_fee.quantize(TWO_DP, rounding=ROUND_HALF_UP)),
                "Variance (Rs)": float(r.fee_variance_inr.quantize(TWO_DP, rounding=ROUND_HALF_UP)),
                "Root Cause":    diag.case_type if diag else "unknown",
                "Notes":         r.notes,
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def bank_gaps_df(bank_gaps):
    rows = []
    for g in bank_gaps:
        rows.append({
            "Payment ID":      g.payment_id,
            "Settlement ID":   g.settlement_id,
            "Gap Type":        g.gap_type,
            "Settled (Rs)":    float(g.net_settled_amount.quantize(TWO_DP, rounding=ROUND_HALF_UP)),
            "Posted (Rs)":     float(g.net_posted_amount.quantize(TWO_DP,  rounding=ROUND_HALF_UP)),
            "Cash Impact (Rs)":float(g.cash_impact_inr.quantize(TWO_DP,   rounding=ROUND_HALF_UP)),
            "Delay (days)":    g.delay_days,
            "Settlement Date": str(g.settlement_date),
            "Posting Date":    str(g.posting_date) if g.posting_date else "—",
            "Escalate":        "YES" if g.should_escalate else "—",
        })
    return pd.DataFrame(rows)


# ── Load ───────────────────────────────────────────────────────────────────────
data            = load_audit()
summary         = data["summary"]
variance_recs   = data["variance_recs"]
bank_gaps       = data["bank_gaps"]
claim_items     = data["claim_items"]
classifications = data["classifications"]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fs-header">
  <h1>🛡️ FeeShield — AI Finance Controller</h1>
  <p>Multi-source reconciliation · Contract engine · Bank matching · Cash-impact analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary",
    "💸 Fee Analysis",
    "🏦 Bank Reconciliation",
    "💰 Cash Position",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Summary
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI row
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""<div class="kpi kpi-info">
          <div class="kpi-label">Volume Processed</div>
          <div class="kpi-value">{_rs(summary.total_volume_inr)}</div>
          <div class="kpi-sub">{summary.total_transactions_audited} transactions</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""<div class="kpi kpi-danger">
          <div class="kpi-label">Fee Leakage</div>
          <div class="kpi-value">{_rs(summary.total_fee_leakage_inr)}</div>
          <div class="kpi-sub">{summary.fee_leakage_count} discrepancies</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""<div class="kpi kpi-warning">
          <div class="kpi-label">Bank Cash at Risk</div>
          <div class="kpi-value">{_rs(summary.total_bank_cash_at_risk_inr)}</div>
          <div class="kpi-sub">{summary.bank_gap_count} escalated gaps</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        total_claims = sum((_d(c.claim_amount_inr) for c in claim_items), Decimal("0"))
        st.markdown(f"""<div class="kpi kpi-danger">
          <div class="kpi-label">Total Claim Value</div>
          <div class="kpi-value">{_rs(total_claims)}</div>
          <div class="kpi-sub">{len(claim_items)} actionable claims</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1.3, 1])

    with col_l:
        st.markdown('<div class="sh">Leakage by Root Cause</div>', unsafe_allow_html=True)
        if summary.leakage_by_root_cause:
            rc_df = pd.DataFrame([
                {"Root Cause": k, "Leakage (Rs)": float(v.quantize(TWO_DP, rounding=ROUND_HALF_UP))}
                for k, v in sorted(summary.leakage_by_root_cause.items(), key=lambda x: x[1], reverse=True)
            ])
            st.bar_chart(rc_df.set_index("Root Cause"), use_container_width=True, height=250)

    with col_r:
        st.markdown('<div class="sh">Bank Gaps by Type</div>', unsafe_allow_html=True)
        if summary.bank_gaps_by_type:
            st.dataframe(
                pd.DataFrame([{"Gap Type": k, "Count": v} for k, v in summary.bank_gaps_by_type.items()]),
                use_container_width=True, hide_index=True,
            )

        st.markdown('<div class="sh">Leakage by Merchant</div>', unsafe_allow_html=True)
        if summary.leakage_by_merchant:
            st.dataframe(
                pd.DataFrame([
                    {"Merchant": k, "Leakage (Rs)": float(v.quantize(TWO_DP, rounding=ROUND_HALF_UP))}
                    for k, v in summary.leakage_by_merchant.items()
                ]),
                use_container_width=True, hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Fee Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    vdf = variance_df(variance_recs, classifications)

    if vdf.empty:
        st.info("No fee variances detected in this batch.")
    else:
        # Filters
        f1, f2 = st.columns(2)
        with f1:
            merch_sel = st.multiselect("Merchant", sorted(vdf["Merchant"].unique()), default=[])
        with f2:
            cause_sel = st.multiselect("Root Cause", sorted(vdf["Root Cause"].unique()), default=[])

        fdf = vdf.copy()
        if merch_sel:
            fdf = fdf[fdf["Merchant"].isin(merch_sel)]
        if cause_sel:
            fdf = fdf[fdf["Root Cause"].isin(cause_sel)]

        m1, m2, m3 = st.columns(3)
        m1.metric("Variances shown", len(fdf))
        m2.metric("Total variance", _rs(Decimal(str(round(fdf["Variance (Rs)"].sum(), 2)))))
        avg = fdf["Variance (Rs)"].mean() if len(fdf) > 0 else 0
        m3.metric("Avg per case", _rs(Decimal(str(round(avg, 2)))))

        st.markdown('<div class="sh">Variance by Root Cause</div>', unsafe_allow_html=True)
        rc_filt = fdf.groupby("Root Cause")["Variance (Rs)"].sum().reset_index()
        st.bar_chart(rc_filt.set_index("Root Cause"), use_container_width=True, height=210)

        st.markdown('<div class="sh">Fee Variance Detail</div>', unsafe_allow_html=True)
        st.dataframe(fdf, use_container_width=True, hide_index=True, height=380)

        st.download_button(
            "⬇ Download CSV", fdf.to_csv(index=False).encode("utf-8"),
            file_name="fee_variances.csv", mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Bank Reconciliation
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    gdf = bank_gaps_df(bank_gaps)

    if gdf.empty:
        st.info("No bank gaps detected.")
    else:
        total_impact = gdf["Cash Impact (Rs)"].sum()
        n_esc = (gdf["Escalate"] == "YES").sum()

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Total Gaps",     len(gdf))
        g2.metric("Escalated",      int(n_esc))
        g3.metric("Monitoring",     len(gdf) - int(n_esc))
        g4.metric("Cash Impact",    _rs(Decimal(str(round(total_impact, 2)))))

        st.markdown('<div class="sh">Cash Impact by Gap Type</div>', unsafe_allow_html=True)
        gt_chart = gdf.groupby("Gap Type")["Cash Impact (Rs)"].sum().reset_index()
        st.bar_chart(gt_chart.set_index("Gap Type"), use_container_width=True, height=200)

        gt_filter = st.multiselect(
            "Filter by Gap Type", sorted(gdf["Gap Type"].unique()), default=[]
        )
        show = gdf[gdf["Gap Type"].isin(gt_filter)] if gt_filter else gdf

        st.markdown('<div class="sh">Bank Gap Detail</div>', unsafe_allow_html=True)

        def _esc_style(v):
            if v == "YES":
                return "background-color:#450a0a; color:#f87171;"
            return "background-color:#064e3b; color:#34d399;"

        st.dataframe(
            show.style.map(_esc_style, subset=["Escalate"]),
            use_container_width=True, hide_index=True, height=340,
        )

        late = gdf[gdf["Gap Type"] == "SETTLEMENT_POSTED_LATE"].copy()
        if not late.empty:
            st.markdown('<div class="sh">Posting Delay (days) — Late Posts Only</div>', unsafe_allow_html=True)
            st.bar_chart(late.set_index("Payment ID")["Delay (days)"], use_container_width=True, height=160)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Cash Position
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    from src.cash_impact.position_calculator import compute_cash_position

    # Controls: injected balance + obligations
    cc1, cc2 = st.columns(2)
    with cc1:
        obligations_input = st.number_input(
            "7-day obligations (Rs)",
            min_value=0, max_value=50_000_000, value=400_000, step=10_000,
            help="Payroll + vendor payments due in 7 days. Comes from ERP in production.",
        )
    with cc2:
        balance_input = st.number_input(
            "Actual bank balance (Rs) — 0 = auto-estimate from audit",
            min_value=0, max_value=100_000_000, value=0, step=50_000,
            help="If 0, a conservative estimate is derived from settlement audit data.",
        )

    obligations_7d    = Decimal(str(obligations_input))
    cleared_override  = Decimal(str(balance_input)) if balance_input > 0 else None

    pos = compute_cash_position(
        bank_gaps=[g.to_dict() for g in bank_gaps],
        audit_summary_dict={
            "total_fee_leakage_inr":       str(summary.total_fee_leakage_inr),
            "total_bank_cash_at_risk_inr": str(summary.total_bank_cash_at_risk_inr),
            "total_volume_inr":            str(summary.total_volume_inr),
        },
        obligations_7d_inr=obligations_7d,
        cleared_cash_inr=cleared_override,
    )

    # Verdict banner
    vc = {"SAFE": "v-safe", "AT_RISK": "v-at-risk", "CRITICAL": "v-critical"}[pos.verdict]
    vi = {"SAFE": "✅", "AT_RISK": "⚠️", "CRITICAL": "🔴"}[pos.verdict]
    st.markdown(f"""
    <div class="{vc}" style="margin:0.5rem 0 1.2rem;">
      <p class="v-title">{vi}&nbsp; VERDICT: {pos.verdict.replace("_"," ")}</p>
      <p class="v-body">{pos.narrative}</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    p1, p2, p3 = st.columns(3)
    p1.metric("Cleared Cash",      _rs(pos.cleared_cash_inr),     help="Confirmed bank postings")
    p2.metric("Expected Inflows",  _rs(pos.expected_inflows_inr), help="Outstanding settlements owed by gateway")
    p3.metric("At-Risk Deductions",_rs(pos.at_risk_inr),          help="Fee overcharges + posting mismatches + reversals",
              delta=f"-{_rs(pos.at_risk_inr)}", delta_color="inverse")

    p4, p5, p6 = st.columns(3)
    p4.metric("Net Cash Position", _rs(pos.net_position_inr))
    p5.metric("7-day Obligations", _rs(pos.obligations_7d_inr))
    p6.metric(
        "Buffer",
        _rs(pos.buffer_inr),
        delta=f"{float(pos.buffer_pct * 100):.1f}% of obligations",
        delta_color="normal" if pos.buffer_inr >= 0 else "inverse",
    )

    # Waterfall chart
    st.markdown('<div class="sh">Cash Waterfall</div>', unsafe_allow_html=True)
    wf = pd.DataFrame({
        "Component": ["Cleared Cash", "Expected Inflows", "At-Risk (deducted)", "Net Position"],
        "Amount (Rs)": [
            float(pos.cleared_cash_inr.quantize(TWO_DP, rounding=ROUND_HALF_UP)),
            float(pos.expected_inflows_inr.quantize(TWO_DP, rounding=ROUND_HALF_UP)),
            -float(pos.at_risk_inr.quantize(TWO_DP, rounding=ROUND_HALF_UP)),
            float(pos.net_position_inr.quantize(TWO_DP, rounding=ROUND_HALF_UP)),
        ],
    })
    st.bar_chart(wf.set_index("Component"), use_container_width=True, height=220)

    # Exceptions table
    if pos.exceptions:
        st.markdown('<div class="sh">Exceptions Driving Risk</div>', unsafe_allow_html=True)
        exc_df = pd.DataFrame(pos.exceptions)
        exc_df["cash_impact_inr"] = exc_df["cash_impact_inr"].apply(
            lambda v: float(_d(v).quantize(TWO_DP, rounding=ROUND_HALF_UP))
        )
        exc_df.columns = ["Settlement ID", "Payment ID", "Gap Type", "Cash Impact (Rs)"]
        st.dataframe(exc_df, use_container_width=True, hide_index=True)

    # Invariant check
    errs = pos.validate()
    if errs:
        st.error("Position invariant violations:\n" + "\n".join(errs))
    else:
        st.caption(f"✓ Cash position invariants verified · As of {pos.as_of_date}")
