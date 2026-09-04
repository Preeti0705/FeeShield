"""
generate_data.py — AI Finance Controller synthetic data generator
=================================================================
Produces 9 CSV files + ground_truth.csv with 13 planted anomaly types.

Extends the original FeeShield Day-1 generator (8 fee-side cases) with:
  - bank_feed.csv: realistic bank posting records (Day 3)
  - 5 new planted bank-side cases (GT009-GT013)
  - Systematic fee-error pattern injected across 35+30 normal payments
    so the aggregator has meaningful batch-level precision/recall numbers

Design decisions (carried forward from Day 1):
  - ALL money values stored as strings with 6 decimal places (Decimal round-trip safe).
  - ROUND_HALF_UP applied everywhere consistently.
  - Fixed random seed (42) — re-running always produces identical data.
  - No float anywhere in money math. The d() helper raises TypeError on float.
  - All dates: Python date objects stored as YYYY-MM-DD strings.
    No datetime, no timezone anywhere. Consistent convention prevents false
    SLA-violation flags from timezone mismatches in matching logic.

New: bank_txn_id uniqueness rule
  - bank_txn_id values are globally unique (BKTXN{n:04d} counter).
  - The ONLY exception is GT012 (reversal), where one settlement_id intentionally
    has two bank_feed rows: original post + reversal_entry. This is documented
    explicitly to distinguish it from accidental ID reuse.

Target counts (Day 3):
  payments:     240  (was 200; +5 bank-planted, +65 systematic injections)
  settlements:  ~238 (240 minus refunded/disputed/pending)
  bank_feed:    ~241 rows (settlements + 2 extra rows for reversal+hold events)
  ground_truth: 13 planted cases (was 8)
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
SIX_DP   = Decimal("0.000001")   # storage precision
TWO_DP   = Decimal("0.01")       # display / claim precision

# Bank-feed SLA constants — declared here so matching logic uses the same values
BANK_POSTING_SLA_DAYS = 3    # normal posting window: 1-3 days after settlement_date
BANK_HOLD_OK_DAYS     = 3    # hold resolved within 3 days → monitoring, not exception

# ── Money helpers (FINANCIAL MATH CHECKPOINT: Decimal only) ──────────────────

def d(value: str | int) -> Decimal:
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
END_DATE   = date(2024, 3, 28)   # pulled back from Mar-31 to leave room for
                                  # bank posting delays to stay within Q1 window


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
# PAY014-PAY015 are intentionally unused (reserved for future cases).
# Normal payments start at PAY016.
#
# Case structure: (payment_id, case_type, description)
PLANTED_CASES_FEE = [
    # Case 1: Wrong MDR applied
    ("PAY001", "wrong_mdr",
     "MER001/CON002 credit card: gateway applied 1.80% (old V1 rate) instead of 1.50%"),
    # Case 2: Missed volume-tier discount
    ("PAY002", "missed_volume_tier",
     "MER001/CON002 credit card: Feb GMV > 5L, tier rate 1.20% applies; gateway used 1.50%"),
    # Case 3: Wrong GST/tax base
    ("PAY003", "wrong_tax_base",
     "MER002/CON003 credit card: GST 18% charged on transaction amount, not on MDR fee"),
    # Case 4: Duplicate fee deduction
    ("PAY004", "duplicate_fee",
     "MER001/CON002 debit card: MDR fee deducted twice in settlement SEL004"),
    # Case 5: Legitimate refund
    ("PAY005", "legitimate_refund",
     "MER001 refund processed correctly; net settlement adjusted properly"),
    # Case 6: Legitimate chargeback
    ("PAY006", "legitimate_chargeback",
     "MER002 chargeback; dispute fee is contractually allowed, not leakage"),
    # Case 7: Timing difference
    ("PAY007", "timing_difference",
     "MER001 payment Mar-30; settlement dated Apr-2 (cross-month timing lag, not an error)"),
    # Case 8: Contract version violation
    ("PAY008", "contract_version_violation",
     "MER001 payment Feb-15: gateway used CON001 (V1, expired Jan-31) rate 1.80% instead of CON002 (V2) rate 1.50%"),
]

PLANTED_CASES_BANK = [
    # Case 9: Settlement created, never posts to bank
    ("PAY009", "SETTLEMENT_NOT_POSTED",
     "MER002/CON003 credit card: settlement SEL009 has no corresponding bank posting"),
    # Case 10: Settlement posts 7 days late (SLA = 3 days)
    ("PAY010", "SETTLEMENT_POSTED_LATE",
     "MER001/CON002 debit card: bank posting arrived 7 days after settlement date (SLA=3 days)"),
    # Case 11: Posted amount differs from settled amount by Rs500 (short credit)
    ("PAY011", "POSTED_AMOUNT_MISMATCH",
     "MER002/CON003 credit card: posted Rs500 less than the settled net amount"),
    # Case 12: Settlement posts, then reversed (net zero, merchant loses the float)
    ("PAY012", "POSTING_REVERSED",
     "MER001/CON002 credit card: bank posted then reversed -- net cash position: zero"),
    # Case 13: Hold placed, clears within 2 days -- monitoring only, NOT an exception
    ("PAY013", "HOLD_PLACED_THEN_CLEARED",
     "MER002/CON003 debit card: hold placed on posting; cleared within 2 days -- within normal window"),
]

# Keep a combined list for backwards-compat checks
PLANTED_CASES = PLANTED_CASES_FEE  # original 8 fee-side only

# ═══════════════════════════════════════════════════════════════════════════════
# 6. ASSEMBLE ALL PAYMENTS (planted + systematic-pattern + normal)
#
#    Systematic pattern injection (NEW in Day 3):
#    The prompt requires 30-50+ repeats of a pattern for credible batch-level
#    precision/recall. We inject two patterns into the normal payment pool:
#
#    PATTERN A -- wrong_mdr_systematic (35 payments, PAY016-PAY050):
#      MER001 card-credit, Feb-Mar 2024, settlement uses 1.80% not 1.50%
#      notes="SYSTEMATIC:wrong_mdr"
#
#    PATTERN B -- missed_tier_systematic (30 payments, PAY051-PAY080):
#      MER001 card-credit, monthly_gmv=600000, settlement uses 1.50% not 1.20%
#      notes="SYSTEMATIC:missed_volume_tier"
#
#    POOL C -- clean normal payments (160 payments, PAY081-PAY240)
#
#    Total: 8 fee-planted + 5 bank-planted + 2 reserved + 35 + 30 + 160 = 240
# ═══════════════════════════════════════════════════════════════════════════════

PAYMENT_METHODS = ["card", "card", "card", "upi"]   # bias toward card
CARD_CATEGORIES = ["credit", "debit"]


def build_payments() -> list[dict]:

    payments = []

    # ── PLANTED FEE CASES (PAY001-PAY008) ────────────────────────────────────

    payments.append({
        "payment_id": "PAY001", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
        "txn_date": "2024-02-10", "amount": money(d("10000")),
        "payment_method": "card", "card_category": "credit",
        "status": "settled", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:wrong_mdr",
    })
    payments.append({
        "payment_id": "PAY002", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
        "txn_date": "2024-02-20", "amount": money(d("15000")),
        "payment_method": "card", "card_category": "credit",
        "status": "settled", "monthly_gmv": money(d("550000")),
        "notes": "PLANTED:missed_volume_tier",
    })
    payments.append({
        "payment_id": "PAY003", "merchant_id": "MER002", "gateway_id": "GW_BETA",
        "txn_date": "2024-01-15", "amount": money(d("20000")),
        "payment_method": "card", "card_category": "credit",
        "status": "settled", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:wrong_tax_base",
    })
    payments.append({
        "payment_id": "PAY004", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
        "txn_date": "2024-02-05", "amount": money(d("8000")),
        "payment_method": "card", "card_category": "debit",
        "status": "settled", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:duplicate_fee",
    })
    payments.append({
        "payment_id": "PAY005", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
        "txn_date": "2024-01-22", "amount": money(d("5000")),
        "payment_method": "card", "card_category": "credit",
        "status": "refunded", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:legitimate_refund",
    })
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

    # PAY009-PAY013: bank-side planted cases
    payments.append({
        "payment_id": "PAY009", "merchant_id": "MER002", "gateway_id": "GW_BETA",
        "txn_date": "2024-01-10", "amount": money(d("18000")),
        "payment_method": "card", "card_category": "credit",
        "status": "settled", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:settlement_not_posted",
    })
    payments.append({
        "payment_id": "PAY010", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
        "txn_date": "2024-02-12", "amount": money(d("6500")),
        "payment_method": "card", "card_category": "debit",
        "status": "settled", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:settlement_posted_late",
    })
    payments.append({
        "payment_id": "PAY011", "merchant_id": "MER002", "gateway_id": "GW_BETA",
        "txn_date": "2024-02-08", "amount": money(d("25000")),
        "payment_method": "card", "card_category": "credit",
        "status": "settled", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:posted_amount_mismatch",
    })
    payments.append({
        "payment_id": "PAY012", "merchant_id": "MER001", "gateway_id": "GW_ALPHA",
        "txn_date": "2024-03-05", "amount": money(d("11000")),
        "payment_method": "card", "card_category": "credit",
        "status": "settled", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:posting_reversed",
    })
    payments.append({
        "payment_id": "PAY013", "merchant_id": "MER002", "gateway_id": "GW_BETA",
        "txn_date": "2024-03-10", "amount": money(d("9500")),
        "payment_method": "card", "card_category": "debit",
        "status": "settled", "monthly_gmv": money(d("0")),
        "notes": "PLANTED:hold_placed_then_cleared",
    })

    # PAY014-PAY015: reserved, intentionally unused

    # ── THREE POOLS: systematic + clean normal (PAY016-PAY240) ──────────────
    merchant_gateway = {"MER001": "GW_ALPHA", "MER002": "GW_BETA"}
    amount_ranges = {
        "MER001": (d("500"),  d("25000")),
        "MER002": (d("1000"), d("80000")),
    }

    def rand_amount(mid: str) -> Decimal:
        lo, hi = amount_ranges[mid]
        cents  = random.randint(int(lo * 100), int(hi * 100))
        return d(str(cents)) / d("100")

    SYS_START = date(2024, 2, 1)
    SYS_END   = date(2024, 3, 25)

    # Pool A: systematic wrong_mdr (35 payments, PAY016-PAY050)
    for i in range(16, 51):
        payments.append({
            "payment_id":     f"PAY{i:03d}",
            "merchant_id":    "MER001",
            "gateway_id":     "GW_ALPHA",
            "txn_date":       str(rand_date(SYS_START, SYS_END)),
            "amount":         money(rand_amount("MER001")),
            "payment_method": "card",
            "card_category":  "credit",
            "status":         "settled",
            "monthly_gmv":    money(d("0")),
            "notes":          "SYSTEMATIC:wrong_mdr",
        })

    # Pool B: systematic missed_volume_tier (30 payments, PAY051-PAY080)
    for i in range(51, 81):
        payments.append({
            "payment_id":     f"PAY{i:03d}",
            "merchant_id":    "MER001",
            "gateway_id":     "GW_ALPHA",
            "txn_date":       str(rand_date(SYS_START, SYS_END)),
            "amount":         money(rand_amount("MER001")),
            "payment_method": "card",
            "card_category":  "credit",
            "status":         "settled",
            "monthly_gmv":    money(d("600000")),
            "notes":          "SYSTEMATIC:missed_volume_tier",
        })

    # Pool C: clean normal (160 payments, PAY081-PAY240)
    for i in range(81, 241):
        merchant_id = random.choice(["MER001", "MER002"])
        pm          = random.choice(PAYMENT_METHODS)
        cc          = random.choice(CARD_CATEGORIES) if pm == "card" else "na"
        payments.append({
            "payment_id":     f"PAY{i:03d}",
            "merchant_id":    merchant_id,
            "gateway_id":     merchant_gateway[merchant_id],
            "txn_date":       str(rand_date()),
            "amount":         money(rand_amount(merchant_id)),
            "payment_method": pm,
            "card_category":  cc,
            "status":         "settled",
            "monthly_gmv":    money(d("0")),
            "notes":          "",
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

    for payment in payments:
        pid    = payment["payment_id"]
        status = payment["status"]
        notes  = payment.get("notes", "")

        # Skip non-settled payments
        if status in ("refunded", "disputed", "pending_settlement"):
            continue

        # ── Fee-side planted cases ──────────────────────────────────────────────────
        if pid == "PAY001":
            rows = settlement_for(payment, rule_override={"mdr_rate": "1.80", "tax_rate": "18.00"})
        elif pid == "PAY002":
            rows = settlement_for(payment, rule_override={"mdr_rate": "1.50", "tax_rate": "18.00"})
        elif pid == "PAY003":
            rows = settlement_for(payment, wrong_tax_base=True)
        elif pid == "PAY004":
            rows = settlement_for(payment, duplicate=True)
        elif pid == "PAY008":
            rows = settlement_for(payment, rule_override={"mdr_rate": "1.80", "tax_rate": "18.00"})

        # ── Systematic pattern injection ───────────────────────────────────────────
        elif "SYSTEMATIC:wrong_mdr" in notes:
            rows = settlement_for(payment, rule_override={"mdr_rate": "1.80", "tax_rate": "18.00"})
        elif "SYSTEMATIC:missed_volume_tier" in notes:
            rows = settlement_for(payment, rule_override={"mdr_rate": "1.50", "tax_rate": "18.00"})

        # ── Bank-side planted cases: fee settlement is CORRECT, bank behavior is wrong ──
        # PAY009: no bank_feed row (handled in build_bank_feed by omission)
        # PAY010: bank_feed row has posting_date = settle+7 (late)
        # PAY011: bank_feed row has posted_amount = net-500 (mismatch)
        # PAY012: two bank_feed rows (posted + reversal_entry)
        # PAY013: two bank_feed rows (held + posted/cleared)
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
                "dispute_id":       f"DIS{p['payment_id'][3:]}",
                "payment_id":       p["payment_id"],
                "merchant_id":      p["merchant_id"],
                "dispute_date":     str(date.fromisoformat(p["txn_date"]) + timedelta(days=10)),
                "disputed_amount":  p["amount"],
                "chargeback_fee":   money(d("250")),
                "outcome":          "merchant_loss",
                "notes":            p.get("notes", ""),
            })
    return disputes


# ═══════════════════════════════════════════════════════════════════════════════
# 10. BANK FEED (NEW — Day 3)
#
#     Schema:
#       bank_txn_id     — globally unique ID (BKTXN{counter:04d})
#       settlement_id   — FK to settlements.csv
#       posted_amount   — credited to merchant's bank account (Decimal, 6dp)
#       posting_date    — date the credit appeared in bank account (YYYY-MM-DD)
#       status          — posted | reversed | reversal_entry | held
#       notes           — empty for clean rows; "PLANTED:..." for anomalies
#
#     Normal posting model:
#       posting_date = settlement_date + random.randint(1, 3)
#       posted_amount = net_settled_amount
#       status = 'posted'
#
#     bank_txn_id uniqueness: globally incrementing counter.
#     The ONLY place two rows share a settlement_id is GT012 (PAY012) —
#     that is intentional and documented.
# ═══════════════════════════════════════════════════════════════════════════════

def build_bank_feed(settlements: list[dict]) -> list[dict]:
    """Generate bank_feed.csv rows for each settlement."""
    bank_feed = []
    counter   = [1]

    def next_bktxn() -> str:
        txn_id = f"BKTXN{counter[0]:04d}"
        counter[0] += 1
        return txn_id

    sel_lookup = {s["settlement_id"]: s for s in settlements}
    handled    = set()

    # GT009: SEL009 intentionally skipped (no bank row = the anomaly)
    handled.add("SEL009")

    # GT010: SEL010 posted 7 days late (SLA=3)
    if "SEL010" in sel_lookup:
        sel = sel_lookup["SEL010"]
        posting_dt = date.fromisoformat(sel["settlement_date"]) + timedelta(days=7)
        bank_feed.append({
            "bank_txn_id":   next_bktxn(),
            "settlement_id": "SEL010",
            "posted_amount": sel["net_settled_amount"],
            "posting_date":  str(posting_dt),
            "status":        "posted",
            "notes":         "PLANTED:settlement_posted_late",
        })
        handled.add("SEL010")

    # GT011: SEL011 posted Rs500 short
    if "SEL011" in sel_lookup:
        sel = sel_lookup["SEL011"]
        posting_dt = date.fromisoformat(sel["settlement_date"]) + timedelta(days=random.randint(1, 3))
        short_amt  = (d(sel["net_settled_amount"]) - d("500")).quantize(SIX_DP, rounding=ROUND_HALF_UP)
        bank_feed.append({
            "bank_txn_id":   next_bktxn(),
            "settlement_id": "SEL011",
            "posted_amount": money(short_amt),
            "posting_date":  str(posting_dt),
            "status":        "posted",
            "notes":         "PLANTED:posted_amount_mismatch",
        })
        handled.add("SEL011")

    # GT012: SEL012 posted then reversed (2 rows, intentional)
    if "SEL012" in sel_lookup:
        sel = sel_lookup["SEL012"]
        posting_dt  = date.fromisoformat(sel["settlement_date"]) + timedelta(days=random.randint(1, 2))
        reversal_dt = posting_dt + timedelta(days=1)
        bank_feed.append({
            "bank_txn_id":   next_bktxn(),
            "settlement_id": "SEL012",
            "posted_amount": sel["net_settled_amount"],
            "posting_date":  str(posting_dt),
            "status":        "reversed",
            "notes":         "PLANTED:posting_reversed|original_post",
        })
        bank_feed.append({
            "bank_txn_id":   next_bktxn(),
            "settlement_id": "SEL012",
            "posted_amount": money(d("0")),
            "posting_date":  str(reversal_dt),
            "status":        "reversal_entry",
            "notes":         "PLANTED:posting_reversed|reversal_debit",
        })
        handled.add("SEL012")

    # GT013: SEL013 held then cleared within 2 days (2 rows, monitoring only)
    if "SEL013" in sel_lookup:
        sel = sel_lookup["SEL013"]
        hold_dt  = date.fromisoformat(sel["settlement_date"]) + timedelta(days=1)
        clear_dt = hold_dt + timedelta(days=2)
        bank_feed.append({
            "bank_txn_id":   next_bktxn(),
            "settlement_id": "SEL013",
            "posted_amount": sel["net_settled_amount"],
            "posting_date":  str(hold_dt),
            "status":        "held",
            "notes":         "PLANTED:hold_placed_then_cleared|hold_event",
        })
        bank_feed.append({
            "bank_txn_id":   next_bktxn(),
            "settlement_id": "SEL013",
            "posted_amount": sel["net_settled_amount"],
            "posting_date":  str(clear_dt),
            "status":        "posted",
            "notes":         "PLANTED:hold_placed_then_cleared|cleared_event",
        })
        handled.add("SEL013")

    # Normal postings: all other settlements
    for sel in settlements:
        sel_id = sel["settlement_id"]
        if sel_id in handled:
            continue
        if sel_id.endswith("_DUP"):
            # DUP rows are internal settlement accounting, not separate bank transfers
            continue
        settle_dt  = date.fromisoformat(sel["settlement_date"])
        posting_dt = settle_dt + timedelta(days=random.randint(1, 3))
        bank_feed.append({
            "bank_txn_id":   next_bktxn(),
            "settlement_id": sel_id,
            "posted_amount": sel["net_settled_amount"],
            "posting_date":  str(posting_dt),
            "status":        "posted",
            "notes":         "",
        })

    return bank_feed


# ═══════════════════════════════════════════════════════════════════════════════
# 11. GROUND TRUTH CSV
#
#     Now tracks 13 planted cases (was 8). New columns for bank-side tracking:
#       bank_case_type     — root cause label for bank cases; empty for fee cases
#       expected_bank_impact — cash impact in rupees; 0 for monitoring cases
#       bank_posting_state — not_applicable | not_posted | posted_late |
#                            amount_mismatch | reversed | hold_cleared
#
#     FINANCIAL MATH CHECKPOINTS:
#       GT009: net = 18000 - 1.60%*18000 - 18%*(1.60%*18000) = 17660.16
#       GT012: net = 11000 - 1.50%*11000 - 18%*(1.50%*11000) = 10805.30
# ═══════════════════════════════════════════════════════════════════════════════

def build_ground_truth() -> list[dict]:
    def ns(amount_str, mdr_rate_str, tax_rate_str) -> Decimal:
        """Compute net settled amount given amount and correct rates."""
        amt     = d(amount_str)
        mdr_fee = pct(mdr_rate_str, amt)
        tax_fee = pct(tax_rate_str, mdr_fee)
        return (amt - mdr_fee - tax_fee).quantize(SIX_DP, rounding=ROUND_HALF_UP)

    NO_BANK = {
        "bank_case_type": "",
        "expected_bank_impact": money(d("0")),
        "bank_posting_state": "not_applicable",
    }

    rows = [
        # ── Original 8 fee-side cases (unchanged logic) ────────────────────────
        {"case_id": "GT001", "payment_id": "PAY001", "case_type": "wrong_mdr",
         "description": "Gateway applied 1.80% instead of contracted 1.50% (CON002)",
         "expected_leakage_inr": money(pct("0.30", d("10000")) + pct("18.00", pct("0.30", d("10000")))),
         "should_flag": "True", **NO_BANK},

        {"case_id": "GT002", "payment_id": "PAY002", "case_type": "missed_volume_tier",
         "description": "Tier rate 1.20% applies (GMV>5L); gateway used 1.50%",
         "expected_leakage_inr": money(pct("0.30", d("15000")) + pct("18.00", pct("0.30", d("15000")))),
         "should_flag": "True", **NO_BANK},

        {"case_id": "GT003", "payment_id": "PAY003", "case_type": "wrong_tax_base",
         "description": "GST 18% on gross Rs20000 instead of on MDR fee",
         "expected_leakage_inr": money(pct("18.00", d("20000")) - pct("18.00", pct("1.60", d("20000")))),
         "should_flag": "True", **NO_BANK},

        {"case_id": "GT004", "payment_id": "PAY004", "case_type": "duplicate_fee",
         "description": "MDR + tax deducted twice in same settlement for Rs8000 debit",
         "expected_leakage_inr": money(pct("0.75", d("8000")) + pct("18.00", pct("0.75", d("8000")))),
         "should_flag": "True", **NO_BANK},

        {"case_id": "GT005", "payment_id": "PAY005", "case_type": "legitimate_refund",
         "description": "Customer refund -- correctly processed, not leakage",
         "expected_leakage_inr": money(d("0")), "should_flag": "False", **NO_BANK},

        {"case_id": "GT006", "payment_id": "PAY006", "case_type": "legitimate_chargeback",
         "description": "Chargeback with contractual dispute fee -- not leakage",
         "expected_leakage_inr": money(d("0")), "should_flag": "False", **NO_BANK},

        {"case_id": "GT007", "payment_id": "PAY007", "case_type": "timing_difference",
         "description": "Mar-30 txn settled Apr-2 -- cross-month lag, not an error",
         "expected_leakage_inr": money(d("0")), "should_flag": "False", **NO_BANK},

        {"case_id": "GT008", "payment_id": "PAY008", "case_type": "contract_version_violation",
         "description": "Feb-15 txn: gateway used stale CON001 (V1) rate 1.80% instead of CON002 rate 1.50%",
         "expected_leakage_inr": money(pct("0.30", d("9000")) + pct("18.00", pct("0.30", d("9000")))),
         "should_flag": "True", **NO_BANK},

        # ── 5 new bank-side cases ────────────────────────────────────────────────
        # GT009: no bank posting at all -- full net amount is at risk
        {"case_id": "GT009", "payment_id": "PAY009", "case_type": "bank_posting_anomaly",
         "description": "SEL009: settlement created, never posted to merchant's bank account",
         "expected_leakage_inr": money(d("0")),   # fee itself is correct
         "should_flag": "True",
         "bank_case_type": "SETTLEMENT_NOT_POSTED",
         "expected_bank_impact": money(ns("18000", "1.60", "18.00")),
         "bank_posting_state": "not_posted"},

        # GT010: posted late -- no permanent financial leakage, just float risk
        {"case_id": "GT010", "payment_id": "PAY010", "case_type": "bank_posting_anomaly",
         "description": "SEL010: settlement posted 7 days after settlement date (SLA=3 days)",
         "expected_leakage_inr": money(d("0")),
         "should_flag": "True",
         "bank_case_type": "SETTLEMENT_POSTED_LATE",
         "expected_bank_impact": money(d("0")),   # money did arrive eventually
         "bank_posting_state": "posted_late"},

        # GT011: posted Rs500 short -- shortfall is the bank impact
        {"case_id": "GT011", "payment_id": "PAY011", "case_type": "bank_posting_anomaly",
         "description": "SEL011: posted Rs500 less than settled net amount",
         "expected_leakage_inr": money(d("0")),
         "should_flag": "True",
         "bank_case_type": "POSTED_AMOUNT_MISMATCH",
         "expected_bank_impact": money(d("500")),
         "bank_posting_state": "amount_mismatch"},

        # GT012: posting reversed -- full net settled amount at risk (zero in account)
        {"case_id": "GT012", "payment_id": "PAY012", "case_type": "bank_posting_anomaly",
         "description": "SEL012: bank posted then reversed -- net merchant cash = Rs0",
         "expected_leakage_inr": money(d("0")),
         "should_flag": "True",
         "bank_case_type": "POSTING_REVERSED",
         "expected_bank_impact": money(ns("11000", "1.50", "18.00")),
         "bank_posting_state": "reversed"},

        # GT013: hold placed, cleared within 2 days -- monitoring only
        {"case_id": "GT013", "payment_id": "PAY013", "case_type": "bank_posting_anomaly",
         "description": "SEL013: hold placed then cleared within 2 days -- within normal window",
         "expected_leakage_inr": money(d("0")),
         "should_flag": "False",
         "bank_case_type": "HOLD_PLACED_THEN_CLEARED",
         "expected_bank_impact": money(d("0")),
         "bank_posting_state": "hold_cleared"},
    ]

    return rows





# ═══════════════════════════════════════════════════════════════════════════════
# 12. CSV WRITER
# ═══════════════════════════════════════════════════════════════════════════════

def write_csv(filename: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  [SKIP] {filename} -- no rows to write")
        return
    path = DATA_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [OK]   {filename:35s}  {len(rows):>5} rows  ->  {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("\n================================================")
    print("  AI Finance Controller - Day 3 Data Generator")
    print("================================================\n")

    payments     = build_payments()
    settlements  = build_settlements(payments)
    refunds      = build_refunds(payments)
    disputes     = build_disputes(payments)
    bank_feed    = build_bank_feed(settlements)
    ground_truth = build_ground_truth()

    print(f"  Payments built:     {len(payments)}")
    print(f"  Settlements built:  {len(settlements)}")
    print(f"  Refunds built:      {len(refunds)}")
    print(f"  Disputes built:     {len(disputes)}")
    print(f"  Bank feed rows:     {len(bank_feed)}")
    print(f"  Ground truth rows:  {len(ground_truth)}")
    print()

    write_csv("merchants.csv",      MERCHANTS)
    write_csv("contracts.csv",      CONTRACTS)
    write_csv("contract_rules.csv", CONTRACT_RULES)
    write_csv("payments.csv",       payments)
    write_csv("settlements.csv",    settlements)
    write_csv("refunds.csv",        refunds)
    write_csv("disputes.csv",       disputes)
    write_csv("bank_feed.csv",      bank_feed)
    write_csv("ground_truth.csv",   ground_truth)

    print("\n------------------------------------------------")
    print("  Sanity checks:")
    _sanity_checks(payments, settlements, bank_feed, ground_truth)
    print("\n  [DONE] Day 3 complete. All CSVs written to data/")
    print("------------------------------------------------\n")


def _sanity_checks(payments, settlements, bank_feed, ground_truth):
    """14 invariants that must all hold. Any failure = data-generation bug."""

    # 1. Payment count
    # 238 = 8 fee-planted + 5 bank-planted + 35 systematic-mdr + 30 systematic-tier + 160 clean
    # PAY014 and PAY015 are reserved but not generated (intentionally)
    assert len(payments) == 238, f"Expected 238 payments, got {len(payments)}"
    print(f"  [checkmark] Payment count = {len(payments)}")


    # 2. All 13 planted IDs present
    all_planted_ids = {c[0] for c in PLANTED_CASES_FEE} | {c[0] for c in PLANTED_CASES_BANK}
    pay_ids  = {p["payment_id"] for p in payments}
    missing  = all_planted_ids - pay_ids
    assert not missing, f"Missing planted IDs: {missing}"
    print(f"  [checkmark] All 13 planted payment IDs present")

    # 3. Ground truth flag counts: 9 True, 4 False
    flagged     = [r for r in ground_truth if r["should_flag"] == "True"]
    not_flagged = [r for r in ground_truth if r["should_flag"] == "False"]
    assert len(flagged)     == 9, f"Expected 9 flag=True, got {len(flagged)}"
    assert len(not_flagged) == 4, f"Expected 4 flag=False, got {len(not_flagged)}"
    print(f"  [checkmark] Ground truth: {len(flagged)} should_flag=True, {len(not_flagged)} should_flag=False")

    # 4. PAY004 still has 2 settlement rows
    dup_rows = [s for s in settlements if s["payment_id"] == "PAY004"]
    assert len(dup_rows) == 2, f"Expected 2 settlement rows for PAY004, got {len(dup_rows)}"
    print(f"  [checkmark] PAY004 has {len(dup_rows)} settlement rows (duplicate confirmed)")

    # 5. SEL009 has NO bank_feed row (GT009 = not_posted)
    sel009_rows = [b for b in bank_feed if b["settlement_id"] == "SEL009"]
    assert len(sel009_rows) == 0, f"SEL009 should have no bank rows, got {len(sel009_rows)}"
    print(f"  [checkmark] SEL009 correctly absent from bank_feed (SETTLEMENT_NOT_POSTED)")

    # 6. SEL010 posted 7 days late (delay > BANK_POSTING_SLA_DAYS=3)
    sel010_bank   = [b for b in bank_feed if b["settlement_id"] == "SEL010"]
    assert len(sel010_bank) == 1, f"SEL010 should have exactly 1 bank row"
    sel010_settle = next(s for s in settlements if s["settlement_id"] == "SEL010")
    delay = (date.fromisoformat(sel010_bank[0]["posting_date"]) -
             date.fromisoformat(sel010_settle["settlement_date"])).days
    assert delay > BANK_POSTING_SLA_DAYS, f"SEL010 delay={delay} should be > {BANK_POSTING_SLA_DAYS}"
    print(f"  [checkmark] SEL010 posting delay = {delay} days (SLA={BANK_POSTING_SLA_DAYS}, correctly late)")

    # 7. SEL011 posted Rs500 short
    sel011_bank   = [b for b in bank_feed if b["settlement_id"] == "SEL011"]
    assert len(sel011_bank) == 1, f"SEL011 should have exactly 1 bank row"
    sel011_settle = next(s for s in settlements if s["settlement_id"] == "SEL011")
    shortfall     = (d(sel011_settle["net_settled_amount"]) - d(sel011_bank[0]["posted_amount"])).quantize(SIX_DP, rounding=ROUND_HALF_UP)
    assert shortfall == d("500"), f"SEL011 shortfall should be Rs500, got {shortfall}"
    print(f"  [checkmark] SEL011 shortfall = Rs{shortfall} (POSTED_AMOUNT_MISMATCH confirmed)")

    # 8. SEL012 has 2 rows: reversed + reversal_entry
    sel012_rows = [b for b in bank_feed if b["settlement_id"] == "SEL012"]
    assert len(sel012_rows) == 2, f"SEL012 should have 2 bank rows, got {len(sel012_rows)}"
    statuses = {r["status"] for r in sel012_rows}
    assert "reversed" in statuses and "reversal_entry" in statuses, \
        f"SEL012 rows should be 'reversed' + 'reversal_entry', got {statuses}"
    print(f"  [checkmark] SEL012 has {len(sel012_rows)} rows: {sorted(statuses)} (POSTING_REVERSED confirmed)")

    # 9. SEL013 has 2 rows: held + posted; hold_duration <= BANK_HOLD_OK_DAYS
    sel013_rows = [b for b in bank_feed if b["settlement_id"] == "SEL013"]
    assert len(sel013_rows) == 2, f"SEL013 should have 2 bank rows, got {len(sel013_rows)}"
    hold_row  = next(r for r in sel013_rows if r["status"] == "held")
    clear_row = next(r for r in sel013_rows if r["status"] == "posted")
    hold_duration = (date.fromisoformat(clear_row["posting_date"]) -
                     date.fromisoformat(hold_row["posting_date"])).days
    assert hold_duration <= BANK_HOLD_OK_DAYS, \
        f"SEL013 hold duration={hold_duration} should be <= {BANK_HOLD_OK_DAYS}"
    print(f"  [checkmark] SEL013 hold_duration = {hold_duration} days <= {BANK_HOLD_OK_DAYS} (monitoring, not exception)")

    # 10. All bank_txn_ids are unique
    all_bktxn = [b["bank_txn_id"] for b in bank_feed]
    assert len(all_bktxn) == len(set(all_bktxn)), "Duplicate bank_txn_ids found!"
    print(f"  [checkmark] All {len(all_bktxn)} bank_txn_ids are unique")

    # 11. Systematic pattern counts
    wrong_mdr_count   = sum(1 for p in payments if "SYSTEMATIC:wrong_mdr" in p.get("notes", ""))
    missed_tier_count = sum(1 for p in payments if "SYSTEMATIC:missed_volume_tier" in p.get("notes", ""))
    assert wrong_mdr_count   == 35, f"Expected 35 systematic wrong_mdr, got {wrong_mdr_count}"
    assert missed_tier_count == 30, f"Expected 30 systematic missed_tier, got {missed_tier_count}"
    print(f"  [checkmark] Systematic patterns: {wrong_mdr_count} wrong_mdr + {missed_tier_count} missed_volume_tier")

    # 12. Decimal spot-check on bank feed
    for b in bank_feed[:20]:
        d(b["posted_amount"])
    print(f"  [checkmark] Bank feed money fields parse cleanly as Decimal (spot-checked 20 rows)")

    # 13. FINANCIAL MATH CHECKPOINT: GT009 bank impact
    # PAY009: Rs18000, MER002/credit, mdr=1.60%, tax=18% on MDR
    # mdr_fee = 18000 * 1.60 / 100 = 288.000000
    # tax_fee = 288 * 18.00 / 100  = 51.840000
    # net     = 18000 - 288 - 51.84 = 17660.160000
    gt009 = next(r for r in ground_truth if r["case_id"] == "GT009")
    assert d(gt009["expected_bank_impact"]) == d("17660.160000"), \
        f"GT009 bank impact mismatch: {gt009['expected_bank_impact']}"
    print(f"  [checkmark] GT009 bank impact hand-checked: Rs{gt009['expected_bank_impact']} (correct)")

    # 14. FINANCIAL MATH CHECKPOINT: GT001 leakage (unchanged)
    # PAY001: Rs10000, overcharge=0.30%, tax=18%: 30 + 5.40 = 35.40
    gt001 = next(r for r in ground_truth if r["case_id"] == "GT001")
    assert d(gt001["expected_leakage_inr"]) == d("35.400000"), \
        f"GT001 leakage mismatch: {gt001['expected_leakage_inr']}"
    print(f"  [checkmark] GT001 leakage hand-checked: Rs{gt001['expected_leakage_inr']} (correct)")


if __name__ == "__main__":
    main()
