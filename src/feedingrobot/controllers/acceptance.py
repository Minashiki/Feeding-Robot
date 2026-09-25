"""Shared M2 fix-scope checks. The verifier recomputes these; it does not trust summary flags."""

from __future__ import annotations

REQUIRED_FILES = (
    "report.json",
    "run.json",
    "execution.json",
    "case_manifest.json",
    "input_hash_after.json",
    "baseline.json",
    "m1_regression_binding.json",
    "controller_config.json",
    "pytest.txt",
)

# Node ids the fixes-v1 suite must actually execute. The acceptance config's
# case-name list is not this set.
REQUIRED_NODE_PARTS = (
    "test_m2_math.py",
    "test_m2_tracking.py",
    "test_m2_wrench.py",
    "test_m2_guard.py",
    "test_m2_traj.py",
    "test_m2_fixes.py",
    "test_m2_report.py",
)

REMAINING_FULL_M2 = (
    "T08 spring surface",
    "T09 stiffness trend",
    "T10 blocked release on a physical wall",
    "T11 force pulse recovery",
    "T15 noise and delay on T05/T08",
    "T16 P0 bridge",
    "T17 full-matrix tamper cases beyond fixes-v1",
    "formal seeds 3-8",
    "full A/B/C matrix from M2Plan",
)
