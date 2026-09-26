"""Tamper checks against a finished V3 package. Not part of the core node list."""

import json
import os
from pathlib import Path

from feedingrobot.controllers.v3spec import ADVERSARIAL, run_adversarial

EXPECTED = {
    "f6_limit": {"threshold_mismatch", "reference_error"},
    "f6_nan": {"nonfinite"},
    "f6_missing_rot": {"missing_field"},
    "manifest_empty": {"evidence_manifest"},
    "manifest_drop": {"evidence_manifest"},
    "manifest_dup": {"duplicate_path"},
    "hash_stale": {"hash_mismatch"},
    "release_json": {"wrong_type"},
    "release_two_frames": {"sample_count", "event_missing"},
    "release_gap": {"bad_time"},
    "release_fake_time": {"event_missing"},
    "release_speed": {"release_speed"},
    "missing_A": {"missing_comparison"},
    "comparisons_lie": {"summary_mismatch"},
    "solver_label": {"solver_mismatch"},
    "node_missing": {"nodeid"},
    "teardown_fail": {"teardown"},
    "run_incomplete": {"run_incomplete"},
    "input_changed": {"input_hash"},
    "failed_m1": {"m1_failed"},
}

if os.environ.get("M2_V3_PACKAGE"):

    def test_v3_package_tampers_are_rejected():
        package = Path(os.environ["M2_V3_PACKAGE"])
        rows = run_adversarial(package)
        assert [row["name"] for row in rows] == list(ADVERSARIAL)
        for row in rows:
            assert row["fixes_passed"] is False
            assert EXPECTED[row["name"]] <= set(row["codes"])
        out = os.environ.get("M2_V3_RESULTS")
        if out:
            Path(out).write_text(json.dumps(rows))
