"""Decide whether an acceptance report is allowed to set m2_ready."""

from __future__ import annotations

REQUIRED_CASES = (
    "T01",
    "T02",
    "T03",
    "T04",
    "T05",
    "T06",
    "T07",
    "T08_tilt",
    "T08_accel",
    "T08_fault",
    "T09",
    "T10_negative",
    "T10_motion",
    "HOLD",
    "T11",
    "T12",
    "T13_T14",
)


def assess(report: dict) -> dict:
    reasons = []
    if report.get("pytest_returncode", 1) != 0:
        reasons.append("pytest failed")
    if report.get("provenance_passed") is not True:
        reasons.append("provenance failed")
    if report.get("hashes_match") is not True:
        reasons.append("hashes changed or do not match")
    cases = report.get("cases") or {}
    for case_id in REQUIRED_CASES:
        row = cases.get(case_id)
        if row is None:
            reasons.append(f"missing {case_id}")
            continue
        if row.get("status") != "passed":
            reasons.append(f"{case_id} {row.get('status')}")
        if "metrics" not in row:
            reasons.append(f"{case_id} has no metrics")
        if not row.get("log_paths") and case_id in ("T08_tilt", "T13_T14", "T10_motion"):
            reasons.append(f"{case_id} has no trajectory")
    if report.get("visual_status") not in ("passed", "unavailable"):
        reasons.append("visual status missing")
    physics = not any(item.startswith("missing") or item.startswith("T") or item == "pytest failed" for item in reasons)
    # physics_passed is false when any required case is missing or failed
    physics_passed = not any(
        item == "pytest failed" or item.startswith("missing") or item.startswith("T0") or item.startswith("T1") or item.startswith("HOLD")
        for item in reasons
    )
    m2_ready = len(reasons) == 0
    if m2_ready:
        status = "passed"
    elif report.get("incomplete"):
        status = "incomplete"
    else:
        status = "failed"
    return {
        "physics_passed": physics_passed and report.get("pytest_returncode", 1) == 0,
        "provenance_passed": report.get("provenance_passed") is True and report.get("hashes_match") is True,
        "visual_status": report.get("visual_status", "unavailable"),
        "m2_ready": m2_ready,
        "overall_status": status,
        "reasons": reasons,
    }
