"""Tamper checks against a finished V4 package. Not part of the core node list."""

import hashlib
import inspect
import os
from pathlib import Path

from feedingrobot.controllers.v3spec import ADVERSARIAL, run_adversarial

EXPECTED = {
    "bad_dtype_tick": {"bad_dtype"},
    "bad_offsets": {"bad_offsets"},
    "bad_rotation": {"invalid_rotation"},
    "bad_shape_k": {"bad_shape"},
    "comparisons_lie": {"summary_mismatch"},
    "f6_limit": {"threshold_mismatch", "reference_error"},
    "f6_missing_rot": {"missing_field"},
    "f6_nan": {"nonfinite"},
    "failed_m1": {"m1_failed"},
    "fast_ramp": {"unload_duration"},
    "fast_skip": {"event_mismatch"},
    "fast_speed": {"release_speed"},
    "gain_huge": {"gain_path_mismatch"},
    "gear_fake_free": {"phase_gear_mismatch"},
    "hash_stale": {"hash_mismatch"},
    "input_changed": {"input_hash"},
    "k_nan": {"nonfinite"},
    "manifest_drop": {"evidence_manifest"},
    "manifest_dup": {"duplicate_path"},
    "manifest_empty": {"evidence_manifest"},
    "missing_A": {"missing_comparison"},
    "mode_aborted": {"unexpected_mode"},
    "mode_unknown": {"invalid_enum"},
    "node_missing": {"nodeid"},
    "pause_integral": {"pause_motion"},
    "pause_vref": {"reference_speed"},
    "q_100": {"joint_range"},
    "release_fake_time": {"event_missing"},
    "release_gap": {"bad_time"},
    "release_json": {"wrong_type"},
    "release_reload": {"external_reloaded"},
    "release_speed": {"release_speed"},
    "release_two_frames": {"sample_count", "event_missing"},
    "run_incomplete": {"run_incomplete"},
    "solver_label": {"solver_mismatch"},
    "teardown_fail": {"teardown"},
}


def _digest(root: Path) -> dict:
    rows = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rows[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


if os.environ.get("M2_V4_PACKAGE"):

    def test_v4_package_tampers_are_rejected():
        package = Path(os.environ["M2_V4_PACKAGE"])
        rows = run_adversarial(package)
        assert [row["name"] for row in rows] == list(ADVERSARIAL)
        for row in rows:
            assert row["fixes_passed"] is False
            assert EXPECTED[row["name"]] <= set(row["codes"])
            assert "clean_copy" not in row["codes"]
        path = os.environ.get("M2_V4_RESULTS")
        if path:
            Path(path).write_text(__import__("json").dumps(rows, indent=2))

    def test_v4_copies_do_not_use_hardlinks():
        assert "os.link" not in inspect.getsource(run_adversarial)

    def test_v4_exception_leaves_the_source_package():
        package = Path(os.environ["M2_V4_PACKAGE"])
        before = _digest(package)
        try:
            run_adversarial(package, fail_after="bad_dtype_tick")
        except RuntimeError as exc:
            assert str(exc) == "bad_dtype_tick"
        else:
            raise AssertionError("tamper injection did not raise")
        assert _digest(package) == before
