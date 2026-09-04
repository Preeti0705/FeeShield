"""
benchmark.py — Ground Truth Evaluator
======================================
Compares run_audit.py output (audit_claims.csv) against ground_truth.csv
and reports precision, recall, and false-positive rate for the full pipeline.

Metrics:
  - True Positive (TP): ground_truth should_flag=True AND audit raised a claim
  - False Negative (FN): ground_truth should_flag=True AND audit did NOT raise a claim
  - False Positive (FP): ground_truth should_flag=False AND audit raised a claim
  - True Negative (TN): ground_truth should_flag=False AND audit did NOT raise a claim

  Precision = TP / (TP + FP)   — of everything flagged, how much was real?
  Recall    = TP / (TP + FN)   — of everything real, how much did we catch?
  FPR       = FP / (FP + TN)   — of all clean cases, how many were wrongly flagged?

Usage:
  py -3.11 benchmark.py
"""

import csv
import sys
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

DATA_DIR = Path(__file__).parent / "data"
TWO_DP = Decimal("0.01")


def load_ground_truth() -> dict[str, dict]:
    """Return {payment_id: row} from ground_truth.csv"""
    path = DATA_DIR / "ground_truth.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run: py -3.11 data/generate_data.py")
        sys.exit(1)
    return {r["payment_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}


def load_claims() -> dict[str, list[dict]]:
    """Return {payment_id: [claim, ...]} from audit_claims.csv"""
    path = DATA_DIR / "audit_claims.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run: py -3.11 run_audit.py")
        sys.exit(1)
    result: dict[str, list[dict]] = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        pid = r["payment_id"]
        result.setdefault(pid, []).append(r)
    return result


def run_benchmark():
    truth = load_ground_truth()
    claims = load_claims()

    TP, FN, FP, TN = 0, 0, 0, 0
    leakage_recovered = Decimal("0")
    leakage_total = Decimal("0")

    print("\n" + "=" * 70)
    print("  AI FINANCE CONTROLLER — BENCHMARK vs GROUND TRUTH")
    print("=" * 70)
    print(f"\n{'Case':<8} {'PayID':<8} {'Expect Flag':<12} {'Raised Claim':<14} {'Result':<4} {'Notes'}")
    print("-" * 70)

    for case_id, gt in sorted(truth.items(), key=lambda x: x[0]):
        pid = gt["payment_id"]
        should_flag = gt["should_flag"] == "True"
        expected_leakage = Decimal(gt["expected_leakage_inr"])
        leakage_total += expected_leakage

        pay_claims = claims.get(pid, [])
        raised_claim = len(pay_claims) > 0

        if should_flag and raised_claim:
            outcome = "TP"
            TP += 1
            # Sum all claim amounts for this payment
            recovered = sum(Decimal(c["claim_amount_inr"].replace(",", "")) for c in pay_claims)
            leakage_recovered += recovered
            label = f"[OK] caught (Rs {recovered:.2f})"
        elif should_flag and not raised_claim:
            outcome = "FN"
            FN += 1
            label = "[!!] MISSED"
        elif not should_flag and raised_claim:
            outcome = "FP"
            FP += 1
            label = f"[!!] FALSE POSITIVE ({[c['root_cause'] for c in pay_claims]})"
        else:
            outcome = "TN"
            TN += 1
            label = "[OK] correctly silent"

        flag_str = "YES" if should_flag else "no"
        claim_str = "YES" if raised_claim else "no"
        print(f"{case_id:<8} {pid:<8} {flag_str:<12} {claim_str:<14} {outcome:<4} {label}")

    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    recovery_rate = float(leakage_recovered / leakage_total * 100) if leakage_total > 0 else 0.0

    print("\n" + "=" * 70)
    print("  METRICS")
    print("=" * 70)
    print(f"  True Positives  (TP): {TP}")
    print(f"  True Negatives  (TN): {TN}")
    print(f"  False Positives (FP): {FP}")
    print(f"  False Negatives (FN): {FN}")
    print()
    print(f"  Precision : {precision:.1%}  (of flagged findings, % that were real)")
    print(f"  Recall    : {recall:.1%}  (of real anomalies, % that were caught)")
    print(f"  FPR       : {fpr:.1%}  (of clean cases, % wrongly flagged)")
    print(f"  F1 Score  : {f1:.1%}")
    print()
    print(f"  Expected Total Leakage   : Rs {float(leakage_total):>10,.2f}")
    print(f"  Claimed Recovery Amount  : Rs {float(leakage_recovered):>10,.2f}")
    print(f"  Recovery Rate            : {recovery_rate:.1f}%")
    print("=" * 70 + "\n")

    return {"precision": precision, "recall": recall, "fpr": fpr, "f1": f1, "TP": TP, "FN": FN, "FP": FP, "TN": TN}


if __name__ == "__main__":
    run_benchmark()
