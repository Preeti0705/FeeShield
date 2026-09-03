"""
generate_data.py — FeeShield synthetic data generator
======================================================
Produces 8 CSV files + ground_truth.csv with exactly 8 planted anomaly types.

Design decisions flagged here (per project rules):
  - ALL money values are written as strings with 6 decimal places so that
    downstream code can read them back with Decimal() without precision loss.
    We NEVER use float for money — amounts are computed with Decimal throughout
    this file and only converted to string for CSV storage.
  - Rounding rule (applied everywhere, consistently): ROUND_HALF_UP to 6 d.p.
    for storage, ROUND_HALF_UP to 2 d.p. for display / claim amounts.
  - Random seed is fixed (42) so re-running always produces identical data.
  - ~200 payment records total (exactly 200 after planted cases).
"""

import csv
import random
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)

DATA_DIR = Path(__file__).parent
SIX_DP = Decimal("0.000001")   # storage precision
TWO_DP = Decimal("0.01")       # display / claim precision

# ── Money helpers (FINANCIAL MATH CHECKPOINT: Decimal only) ──────────────────

def d(value: str | int | float) -> Decimal:
    """Convert any value to Decimal. Raises TypeError on float to catch mistakes."""
    if isinstance(value, float):
        raise TypeError(
            f"STOP: float {value!r} passed to d(). Use a string literal instead."
        )
    return Decimal(str(value))


def money(value: Decimal) -> str:
    """Serialise a Decimal to a 6-decimal-place string for CSV storage."""
    return str(value.quantize(SIX_DP, rounding=ROUND_HALF_UP))


def pct(rate_str: str, base: Decimal) -> Decimal:
    """Apply a percentage rate (given as '1.80' meaning 1.80%) to a base amount."""
    return (d(rate_str) / d("100") * base).quantize(SIX_DP, rounding=ROUND_HALF_UP)


# ── Date helpers ─────────────────────────────────────────────────────────────

START_DATE = date(2024, 1, 1)
END_DATE   = date(2024, 3, 31)   # Q1 2024 — 91 days


def rand_date(start: date = START_DATE, end: date = END_DATE) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MERCHANTS
#    Two merchants: one small (MER001) and one mid-size (MER002).
#    Having two merchants lets us test groupby aggregation in Day 4.
# ═══════════════════════════════════════════════════════════════════════════════

MERCHANTS = [
    {"merchant_id": "MER001", "name": "Sunrise Retail", "gstin": "29AABCU9603R1ZX",
     "monthly_gmv_tier": "low"},
    {"merchant_id": "MER002", "name": "Metro Electronics", "gstin": "27AADCB2230M1ZT",
     "monthly_gmv_tier": "high"},
]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONTRACTS
#    Each merchant has one contract with ONE gateway.
#    Contract V1 is effective Jan-1; for MER001 a V2 takes effect Feb-1
#    (this is what triggers case type 8: stale contract version violation).
# ═══════════════════════════════════════════════════════════════════════════════

CONTRACTS = [
    # MER001: V1 expires Jan-31, V2 starts Feb-1
    {"contract_id": "CON001", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
     "version": 1, "effective_from": "2024-01-01", "effective_to": "2024-01-31",
     "status": "superseded"},
    {"contract_id": "CON002", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
     "version": 2, "effective_from": "2024-02-01", "effective_to": "2024-12-31",
     "status": "active"},
    # MER002: single contract, no version change
    {"contract_id": "CON003", "merchant_id": "MER002", "gateway_id": "GW_BETA",
     "version": 1, "effective_from": "2024-01-01", "effective_to": "2024-12-31",
     "status": "active"},
]

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONTRACT RULES
#    Each contract has rules keyed by (payment_method, card_category).
#    Volume tiers within a rule lower the MDR once monthly GMV crosses a threshold.
#
#    Design decision: volume_tier_min_gmv is in rupees; "0" means "no minimum".
#    The resolver (Day 2) will select the highest-GMV tier the merchant qualifies for.
#
#    Rate columns: mdr_rate (%), fixed_fee (₹), tax_rate (%).
#    CON002 has a lower MDR than CON001 — so using CON001 after Feb-1 is a violation.
# ═══════════════════════════════════════════════════════════════════════════════

CONTRACT_RULES = [
    # ── CON001: MER001 V1 (Jan only) ─────────────────────────────────────────
    {"rule_id": "RUL001", "contract_id": "CON001", "payment_method": "card",
     "card_category": "credit", "volume_tier_min_gmv": "0",
     "mdr_rate": "1.80", "fixed_fee": "0", "tax_rate": "18.00"},
    {"rule_id": "RUL002", "contract_id": "CON001", "payment_method": "card",
     "card_category": "debit", "volume_tier_min_gmv": "0",
     "mdr_rate": "0.90", "fixed_fee": "0", "tax_rate": "18.00"},
    {"rule_id": "RUL003", "contract_id": "CON001", "payment_method": "upi",
     "card_category": "na", "volume_tier_min_gmv": "0",
     "mdr_rate": "0.00", "fixed_fee": "0", "tax_rate": "0.00"},

    # ── CON002: MER001 V2 (Feb onward) — lower credit MDR, volume tiers ──────
    {"rule_id": "RUL004", "contract_id": "CON002", "payment_method": "card",
     "card_category": "credit", "volume_tier_min_gmv": "0",
     "mdr_rate": "1.50", "fixed_fee": "0", "tax_rate": "18.00"},
    {"rule_id": "RUL005", "contract_id": "CON002", "payment_method": "card",
     "card_category": "credit", "volume_tier_min_gmv": "500000",
     "mdr_rate": "1.20", "fixed_fee": "0", "tax_rate": "18.00"},
    {"rule_id": "RUL006", "contract_id": "CON002", "payment_method": "card",
     "card_category": "credit", "volume_tier_min_gmv": "2000000",
     "mdr_rate": "1.00", "fixed_fee": "0", "tax_rate": "18.00"},
    {"rule_id": "RUL007", "contract_id": "CON002", "payment_method": "card",
     "card_category": "debit", "volume_tier_min_gmv": "0",
     "mdr_rate": "0.75", "fixed_fee": "0", "tax_rate": "18.00"},
    {"rule_id": "RUL008", "contract_id": "CON002", "payment_method": "upi",
     "card_category": "na", "volume_tier_min_gmv": "0",
     "mdr_rate": "0.00", "fixed_fee": "0", "tax_rate": "0.00"},

    # ── CON003: MER002 — higher-volume merchant, volume tiers on debit too ───
    {"rule_id": "RUL009", "contract_id": "CON003", "payment_method": "card",
     "card_category": "credit", "volume_tier_min_gmv": "0",
     "mdr_rate": "1.60", "fixed_fee": "0", "tax_rate": "18.00"},
    {"rule_id": "RUL010", "contract_id": "CON003", "payment_method": "card",
     "card_category": "credit", "volume_tier_min_gmv": "500000",
     "mdr_rate": "1.30", "fixed_fee": "0", "tax_rate": "18.00"},
    {"rule_id": "RUL011", "contract_id": "CON003", "payment_method": "card",
     "card_category": "debit", "volume_tier_min_gmv": "0",
     "mdr_rate": "0.80", "fixed_fee": "0", "tax_rate": "18.00"},
    {"rule_id": "RUL012", "contract_id": "CON003", "payment_method": "card",
     "card_category": "debit", "volume_tier_min_gmv": "500000",
     "mdr_rate": "0.60", "fixed_fee": "0", "tax_rate": "18.00"},
    {"rule_id": "RUL013", "contract_id": "CON003", "payment_method": "upi",
     "card_category": "na", "volume_tier_min_gmv": "0",
     "mdr_rate": "0.00", "fixed_fee": "0", "tax_rate": "0.00"},
]

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLANTED ANOMALY DEFINITIONS
#    These are the 8 ground-truth case types, each mapped to specific payment IDs
#    so our benchmark can do exact-match precision/recall.
# ═══════════════════════════════════════════════════════════════════════════════

# We reserve payment IDs PAY001-PAY015 for planted cases.
# The rest (PAY016-PAY200) are normal clean payments.
#
# Case structure: (payment_id, case_type, description)
PLANTED_CASES = [
    # Case 1: Wrong MDR applied — gateway charges 1.80% instead of contracted 1.50%
    ("PAY001", "wrong_mdr",
     "MER001/CON002 credit card: gateway applied 1.80% (old V1 rate) instead of 1.50%"),

    # Case 2: Missed volume-tier discount — GMV exceeds ₹5L threshold but base rate used
    ("PAY002", "missed_volume_tier",
     "MER001/CON002 credit card: Feb GMV > 5L, tier rate 1.20% applies; gateway used 1.50%"),

    # Case 3: Wrong GST/tax base — tax computed on gross amount instead of MDR fee
    ("PAY003", "wrong_tax_base",
     "MER002/CON003 credit card: GST 18% charged on transaction amount, not on MDR fee"),

    # Case 4: Duplicate fee deduction — exact same fee deducted twice in same settlement
    ("PAY004", "duplicate_fee",
     "MER001/CON002 debit card: MDR fee deducted twice in settlement SEL004"),

    # Case 5: Legitimate refund — should NOT be flagged as leakage
    ("PAY005", "legitimate_refund",
     "MER001 refund processed correctly; net settlement adjusted properly"),

    # Case 6: Legitimate chargeback — should NOT be flagged as leakage
    ("PAY006", "legitimate_chargeback",
     "MER002 chargeback; dispute fee is contractually allowed, not leakage"),

    # Case 7: Timing difference — payment in late March, fee settled in April (outside window)
    ("PAY007", "timing_difference",
     "MER001 payment Mar-30; settlement dated Apr-2 (cross-month timing lag, not an error)"),

    # Case 8: Contract version violation — stale V1 rate applied to a Feb payment
    ("PAY008", "contract_version_violation",
     "MER001 payment Feb-15: gateway used CON001 (V1, expired Jan-31) rate 1.80% instead of CON002 (V2) rate 1.50%"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PAYMENT GENERATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

PAYMENT_METHODS = ["card", "card", "card", "upi"]   # bias toward card
CARD_CATEGORIES = ["credit", "debit"]


def normal_payment(pay_id: str, merchant_id: str, gateway_id: str,
                   txn_date: date, amount: Decimal,
                   payment_method: str, card_category: str) -> dict:
    """Create a clean payment row with correct fee fields."""
    return {
        "payment_id": pay_id,
        "merchant_id": merchant_id,
        "gateway_id": gateway_id,
        "txn_date": str(txn_date),
        "amount": money(amount),
        "payment_method": payment_method,
        "card_category": card_category,
        "status": "settled",
        "monthly_gmv": money(d("0")),  # filled in later per-month aggregation
        "notes": "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ASSEMBLE ALL PAYMENTS (planted + normal)
# ═══════════════════════════════════════════════════════════════════════════════

def build_payments() -> list[dict]:
    payments = []

    # ── PLANTED CASE 1: Wrong MDR ─────────────────────────────────────────────
    payments.append({
        "payment_id": "PAY001", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
        "txn_date": "2024-02-10", "amount": money(d("10000")),
        "payment_method": "card", "card_category": "credit",
        "status": "settled", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:wrong_mdr",
    })

    # ── PLANTED CASE 2: Missed volume-tier discount ───────────────────────────
    # GMV set high enough (>5L) in settlement so tier should kick in
    payments.append({
        "payment_id": "PAY002", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
        "txn_date": "2024-02-20", "amount": money(d("15000")),
        "payment_method": "card", "card_category": "credit",
        "status": "settled", "monthly_gmv": money(d("550000")),
        "notes": "PLANTED:missed_volume_tier",
    })

    # ── PLANTED CASE 3: Wrong tax base ───────────────────────────────────────
    payments.append({
        "payment_id": "PAY003", "merchant_id": "MER002", "gateway_id": "GW_BETA",
        "txn_date": "2024-01-15", "amount": money(d("20000")),
        "payment_method": "card", "card_category": "credit",
        "status": "settled", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:wrong_tax_base",
    })

    # ── PLANTED CASE 4: Duplicate fee ────────────────────────────────────────
    payments.append({
        "payment_id": "PAY004", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
        "txn_date": "2024-02-05", "amount": money(d("8000")),
        "payment_method": "card", "card_category": "debit",
        "status": "settled", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:duplicate_fee",
    })

    # ── PLANTED CASE 5: Legitimate refund (status = refunded) ────────────────
    payments.append({
        "payment_id": "PAY005", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
        "txn_date": "2024-01-22", "amount": money(d("5000")),
        "payment_method": "card", "card_category": "credit",
        "status": "refunded", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:legitimate_refund",
    })

    # ── PLANTED CASE 6: Legitimate chargeback (status = disputed) ────────────
    payments.append({
        "payment_id": "PAY006", "merchant_id": "MER002", "gateway_id": "GW_BETA",
        "txn_date": "2024-01-28", "amount": money(d("12000")),
        "payment_method": "card", "card_category": "credit",
        "status": "disputed", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:legitimate_chargeback",
    })

    # ── PLANTED CASE 7: Timing difference ────────────────────────────────────
    payments.append({
        "payment_id": "PAY007", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
        "txn_date": "2024-03-30", "amount": money(d("7500")),
        "payment_method": "card", "card_category": "debit",
        "status": "pending_settlement", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:timing_difference",
    })

    # ── PLANTED CASE 8: Contract version violation ────────────────────────────
    payments.append({
        "payment_id": "PAY008", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
        "txn_date": "2024-02-15", "amount": money(d("9000")),
        "payment_method": "card", "card_category": "credit",
        "status": "settled", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:contract_version_violation",
    })

    # ── NORMAL CLEAN PAYMENTS (PAY016 onward, 192 records) ───────────────────
    merchant_gateway = {
        "MER001": "GW_ALPHA",
        "MER002": "GW_BETA",
    }
    amount_ranges = {
        "MER001": (d("500"), d("25000")),
        "MER002": (d("1000"), d("80000")),
    }

    for i in range(16, 208):   # 192 extra records → total = 200
        merchant_id = random.choice(["MER001", "MER002"])
        gateway_id  = merchant_gateway[merchant_id]
        pm          = random.choice(PAYMENT_METHODS)
        cc          = random.choice(CARD_CATEGORIES) if pm == "card" else "na"
        lo, hi      = amount_ranges[merchant_id]
        # Decimal random amount: pick random integer cents between lo and hi
        amount_cents = random.randint(int(lo * 100), int(hi * 100))
        amount = d(str(amount_cents)) / d("100")
        txn_date = rand_date()

        payments.append({
            "payment_id": f"PAY{i:03d}",
            "merchant_id": merchant_id,
            "gateway_id": gateway_id,
            "txn_date": str(txn_date),
            "amount": money(amount),
            "payment_method": pm,
            "card_category": cc,
            "status": "settled",
            "monthly_gmv": money(d("0")),
            "notes": "",
        })

    return payments


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SETTLEMENTS
#    Each settled payment has one settlement record, EXCEPT PAY004 which has
#    a duplicate fee row in its settlement (case type 4).
#
#    Settlement columns: settlement_id, payment_id, settlement_date,
#    actual_mdr_fee, actual_tax, actual_fixed_fee, net_settled_amount, notes
#
#    FINANCIAL MATH CHECKPOINT: all fee math done with Decimal, ROUND_HALF_UP.
# ═══════════════════════════════════════════════════════════════════════════════

def settlement_for(payment: dict, rule_override: dict | None = None,
                   duplicate: bool = False, wrong_tax_base: bool = False) -> list[dict]:
    """
    Build settlement row(s) for a payment.
    rule_override: pass a dict with mdr_rate/tax_rate to plant a wrong rate.
    duplicate: if True, add a second identical fee deduction row (case 4).
    wrong_tax_base: if True, compute GST on gross amount instead of MDR fee (case 3).
    """
    amount = d(payment["amount"])
    pm     = payment["payment_method"]
    cc     = payment["card_category"]
    pid    = payment["payment_id"]

    # Determine the rate to use for this settlement
    if rule_override:
        mdr_rate = d(rule_override["mdr_rate"])
        tax_rate = d(rule_override["tax_rate"])
    else:
        # Look up the "normal" correct rate for this merchant/method/category
        mdr_rate, tax_rate = _lookup_normal_rate(payment)

    mdr_fee  = pct(str(mdr_rate), amount)
    if wrong_tax_base:
        # Bug: tax applied to gross amount, not mdr_fee
        tax_fee = pct(str(tax_rate), amount)
    else:
        tax_fee  = pct(str(tax_rate), mdr_fee)

    fixed_fee = d("0")
    net_settled = (amount - mdr_fee - tax_fee - fixed_fee).quantize(
        SIX_DP, rounding=ROUND_HALF_UP
    )

    settle_date = str(date.fromisoformat(payment["txn_date"]) + timedelta(days=2))

    rows = [{
        "settlement_id": f"SEL{pid[3:]}",
        "payment_id": pid,
        "settlement_date": settle_date,
        "actual_mdr_fee": money(mdr_fee),
        "actual_tax": money(tax_fee),
        "actual_fixed_fee": money(fixed_fee),
        "net_settled_amount": money(net_settled),
        "notes": payment.get("notes", ""),
    }]

    if duplicate:
        # Case 4: exact same fee row appears twice → leakage = mdr_fee + tax_fee
        rows.append({
            "settlement_id": f"SEL{pid[3:]}_DUP",
            "payment_id": pid,
            "settlement_date": settle_date,
            "actual_mdr_fee": money(mdr_fee),
            "actual_tax": money(tax_fee),
            "actual_fixed_fee": money(fixed_fee),
            "net_settled_amount": money(d("0")),   # money already taken in first row
            "notes": "DUPLICATE_DEDUCTION",
        })

    return rows


def _lookup_normal_rate(payment: dict) -> tuple[Decimal, Decimal]:
    """
    Return (mdr_rate, tax_rate) for a payment given its merchant, method, category.
    Uses the simple 'correct' rate — the resolver in Day 2 does the full logic.
    This is only for data generation, not the audit engine.
    """
    mid = payment["merchant_id"]
    pm  = payment["payment_method"]
    cc  = payment["card_category"]
    dt  = date.fromisoformat(payment["txn_date"])

    if mid == "MER001":
        if dt < date(2024, 2, 1):   # V1 contract
            rates = {"card_credit": ("1.80", "18.00"),
                     "card_debit":  ("0.90", "18.00"),
                     "upi_na":      ("0.00",  "0.00")}
        else:                        # V2 contract
            gmv = d(payment["monthly_gmv"])
            if pm == "card" and cc == "credit":
                if gmv >= d("2000000"):
                    mdr = "1.00"
                elif gmv >= d("500000"):
                    mdr = "1.20"
                else:
                    mdr = "1.50"
                return d(mdr), d("18.00")
            rates = {"card_debit": ("0.75", "18.00"),
                     "upi_na":     ("0.00",  "0.00")}
    else:   # MER002
        gmv = d(payment["monthly_gmv"])
        if pm == "card" and cc == "credit":
            mdr = "1.30" if gmv >= d("500000") else "1.60"
            return d(mdr), d("18.00")
        if pm == "card" and cc == "debit":
            mdr = "0.60" if gmv >= d("500000") else "0.80"
            return d(mdr), d("18.00")
        rates = {"upi_na": ("0.00", "0.00")}

    key = f"{pm}_{cc}"
    r, t = rates.get(key, ("0.00", "0.00"))
    return d(r), d(t)


def build_settlements(payments: list[dict]) -> list[dict]:
    settlements = []
    planted_notes = {p["notes"]: p for p in payments if "PLANTED" in p.get("notes", "")}

    for payment in payments:
        pid    = payment["payment_id"]
        status = payment["status"]
        notes  = payment.get("notes", "")

        # Skip non-settled payments
        if status in ("refunded", "disputed", "pending_settlement"):
            continue

        if pid == "PAY001":
            # Case 1: Wrong MDR — use old V1 rate (1.80%) instead of V2 (1.50%)
            rows = settlement_for(payment, rule_override={"mdr_rate": "1.80", "tax_rate": "18.00"})
        elif pid == "PAY002":
            # Case 2: Missed tier — use base rate 1.50% instead of tier rate 1.20%
            rows = settlement_for(payment, rule_override={"mdr_rate": "1.50", "tax_rate": "18.00"})
        elif pid == "PAY003":
            # Case 3: Wrong tax base
            rows = settlement_for(payment, wrong_tax_base=True)
        elif pid == "PAY004":
            # Case 4: Duplicate fee
            rows = settlement_for(payment, duplicate=True)
        elif pid == "PAY008":
            # Case 8: Contract version violation — use V1 rate post-V2 effective
            rows = settlement_for(payment, rule_override={"mdr_rate": "1.80", "tax_rate": "18.00"})
        else:
            rows = settlement_for(payment)

        settlements.extend(rows)

    return settlements


# ═══════════════════════════════════════════════════════════════════════════════
# 8. REFUNDS (Case 5 + a few normal refunds)
# ═══════════════════════════════════════════════════════════════════════════════

def build_refunds(payments: list[dict]) -> list[dict]:
    refunds = []
    # PAY005 is the planted refund
    for p in payments:
        if p["status"] == "refunded":
            refund_date = str(date.fromisoformat(p["txn_date"]) + timedelta(days=5))
            refunds.append({
                "refund_id": f"REF{p['payment_id'][3:]}",
                "payment_id": p["payment_id"],
                "merchant_id": p["merchant_id"],
                "refund_date": refund_date,
                "refund_amount": p["amount"],   # full refund for simplicity
                "reason": "customer_request",
                "notes": p.get("notes", ""),
            })
    return refunds


# ═══════════════════════════════════════════════════════════════════════════════
# 9. DISPUTES (Case 6 + chargeback fee)
# ═══════════════════════════════════════════════════════════════════════════════

def build_disputes(payments: list[dict]) -> list[dict]:
    disputes = []
    for p in payments:
        if p["status"] == "disputed":
            disputes.append({
                "dispute_id": f"DIS{p['payment_id'][3:]}",
                "payment_id": p["payment_id"],
                "merchant_id": p["merchant_id"],
                "dispute_date": str(date.fromisoformat(p["txn_date"]) + timedelta(days=10)),
                "disputed_amount": p["amount"],
                "chargeback_fee": money(d("250")),   # flat ₹250 per CON003
                "outcome": "merchant_loss",
                "notes": p.get("notes", ""),
            })
    return disputes


# ═══════════════════════════════════════════════════════════════════════════════
# 10. GROUND TRUTH CSV
#     One row per planted case. The benchmark in Day 4 uses this to compute
#     precision, recall, false-positive-rate.
#
#     Columns: case_id, payment_id, case_type, expected_leakage (or "none"),
#              should_flag (True/False — False for cases 5,6,7)
# ═══════════════════════════════════════════════════════════════════════════════

def build_ground_truth() -> list[dict]:
    return [
        {"case_id": "GT001", "payment_id": "PAY001", "case_type": "wrong_mdr",
         "description": "Gateway applied 1.80% instead of contracted 1.50% (CON002)",
         "expected_leakage_inr": money(
             pct("0.30", d("10000")) + pct("18.00", pct("0.30", d("10000")))
         ),
         "should_flag": "True"},

        {"case_id": "GT002", "payment_id": "PAY002", "case_type": "missed_volume_tier",
         "description": "Tier rate 1.20% applies (GMV>5L); gateway used 1.50%",
         "expected_leakage_inr": money(
             pct("0.30", d("15000")) + pct("18.00", pct("0.30", d("15000")))
         ),
         "should_flag": "True"},

        {"case_id": "GT003", "payment_id": "PAY003", "case_type": "wrong_tax_base",
         "description": "GST 18% on gross ₹20000 instead of on MDR fee",
         "expected_leakage_inr": money(
             # Correct: GST on mdr_fee = 1.60% of 20000 = 320; GST = 57.60
             # Actual:  GST on gross   = 18% of 20000   = 3600
             # Leakage = 3600 - 57.60 = 3542.40
             pct("18.00", d("20000")) - pct("18.00", pct("1.60", d("20000")))
         ),
         "should_flag": "True"},

        {"case_id": "GT004", "payment_id": "PAY004", "case_type": "duplicate_fee",
         "description": "MDR + tax deducted twice in same settlement for ₹8000 debit",
         "expected_leakage_inr": money(
             # Full second deduction: mdr_fee + tax on mdr_fee
             pct("0.75", d("8000")) + pct("18.00", pct("0.75", d("8000")))
         ),
         "should_flag": "True"},

        {"case_id": "GT005", "payment_id": "PAY005", "case_type": "legitimate_refund",
         "description": "Customer refund — correctly processed, not leakage",
         "expected_leakage_inr": money(d("0")),
         "should_flag": "False"},

        {"case_id": "GT006", "payment_id": "PAY006", "case_type": "legitimate_chargeback",
         "description": "Chargeback with contractual dispute fee — not leakage",
         "expected_leakage_inr": money(d("0")),
         "should_flag": "False"},

        {"case_id": "GT007", "payment_id": "PAY007", "case_type": "timing_difference",
         "description": "Mar-30 txn settled Apr-2 — cross-month lag, not an error",
         "expected_leakage_inr": money(d("0")),
         "should_flag": "False"},

        {"case_id": "GT008", "payment_id": "PAY008", "case_type": "contract_version_violation",
         "description": "Feb-15 txn: gateway used stale CON001 (V1) rate 1.80% instead of CON002 rate 1.50%",
         "expected_leakage_inr": money(
             pct("0.30", d("9000")) + pct("18.00", pct("0.30", d("9000")))
         ),
         "should_flag": "True"},
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 11. CSV WRITER
# ═══════════════════════════════════════════════════════════════════════════════

def write_csv(filename: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  [SKIP] {filename} — no rows to write")
        return
    path = DATA_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [OK]   {filename:35s}  {len(rows):>5} rows  →  {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n================================================")
    print("  FeeShield - Day 1 Data Generator")
    print("================================================\n")

    payments     = build_payments()
    settlements  = build_settlements(payments)
    refunds      = build_refunds(payments)
    disputes     = build_disputes(payments)
    ground_truth = build_ground_truth()

    print(f"  Payments built:    {len(payments)}")
    print(f"  Settlements built: {len(settlements)}")
    print(f"  Refunds built:     {len(refunds)}")
    print(f"  Disputes built:    {len(disputes)}")
    print(f"  Ground truth rows: {len(ground_truth)}")
    print()

    write_csv("merchants.csv",      MERCHANTS)
    write_csv("contracts.csv",      CONTRACTS)
    write_csv("contract_rules.csv", CONTRACT_RULES)
    write_csv("payments.csv",       payments)
    write_csv("settlements.csv",    settlements)
    write_csv("refunds.csv",        refunds)
    write_csv("disputes.csv",       disputes)
    write_csv("ground_truth.csv",   ground_truth)

    print("\n------------------------------------------------")
    print("  Sanity checks:")
    _sanity_checks(payments, settlements, ground_truth)
    print("\n  [DONE] Day 1 complete. All CSVs written to data/")
    print("------------------------------------------------\n")


def _sanity_checks(payments, settlements, ground_truth):
    # 1. Total payment count
    assert len(payments) == 200, f"Expected 200 payments, got {len(payments)}"
    print(f"  [✓] Payment count = {len(payments)}")

    # 2. All planted IDs present
    planted_ids = {c[0] for c in PLANTED_CASES}
    pay_ids     = {p["payment_id"] for p in payments}
    missing     = planted_ids - pay_ids
    assert not missing, f"Missing planted IDs: {missing}"
    print(f"  [✓] All 8 planted payment IDs present")

    # 3. Ground truth: exactly 5 should_flag=True, 3 should_flag=False
    flagged     = [r for r in ground_truth if r["should_flag"] == "True"]
    not_flagged = [r for r in ground_truth if r["should_flag"] == "False"]
    assert len(flagged)     == 5, f"Expected 5 flag=True rows, got {len(flagged)}"
    assert len(not_flagged) == 3, f"Expected 3 flag=False rows, got {len(not_flagged)}"
    print(f"  [✓] Ground truth: 5 should_flag=True, 3 should_flag=False")

    # 4. PAY004 has 2 settlement rows (duplicate)
    dup_rows = [s for s in settlements if s["payment_id"] == "PAY004"]
    assert len(dup_rows) == 2, f"Expected 2 settlement rows for PAY004, got {len(dup_rows)}"
    print(f"  [✓] PAY004 has {len(dup_rows)} settlement rows (duplicate confirmed)")

    # 5. No float in any money field (spot-check: parse back and round-trip)
    for p in payments[:10]:
        val = Decimal(p["amount"])   # would raise InvalidOperation if corrupt
    print(f"  [✓] Money fields parse cleanly as Decimal (spot-checked 10 rows)")

    # 6. FINANCIAL MATH CHECKPOINT: confirm GT001 expected_leakage hand-calc
    # PAY001: amount=10000, overcharge=0.30%, tax on overcharge=18%
    # overcharge_mdr = 10000 * 0.30 / 100 = 30.000000
    # overcharge_tax = 30 * 18 / 100       = 5.400000
    # total leakage                         = 35.400000
    gt001 = next(r for r in ground_truth if r["case_id"] == "GT001")
    expected = d("35.400000")
    actual   = d(gt001["expected_leakage_inr"])
    assert actual == expected, f"GT001 leakage mismatch: {actual} vs {expected}"
    print(f"  [✓] GT001 leakage hand-checked: ₹{actual} (correct)")


if __name__ == "__main__":
    main()
