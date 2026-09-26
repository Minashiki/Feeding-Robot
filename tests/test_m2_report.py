"""F1 and V3: invalid evidence must not pass. Thresholds come from the checker, not the file."""

import json
from pathlib import Path

import numpy as np

from feedingrobot.controllers.acceptance import required_nodeids
from feedingrobot.scripts.verify_m2_report import assess_m2


def _codes(verdict):
    return [item["code"] if isinstance(item, dict) else str(item) for item in verdict["reasons"]]


def _texts(verdict):
    return [item.get("text", "") if isinstance(item, dict) else str(item) for item in verdict["reasons"]]


def test_f1_summary_only_is_rejected(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"incomplete": False, "m3_ready": True, "hybrid_force_status": "disabled", "reasons": []}))
    for scope in ("fixes-v1", "fixes-v2", "full"):
        verdict = assess_m2(tmp_path, scope, check_workspace=True)
        assert verdict["fixes_passed"] is False
        assert verdict["m3_ready"] is False
        assert verdict["reasons"]


def test_f1_empty_speed_array_is_rejected(tmp_path):
    (tmp_path / "cases").mkdir()
    np.savez(tmp_path / "cases" / "F4-speed.npz", t=np.zeros(0), v_hist=np.zeros((0, 6)), p_ref=np.zeros((0, 3)), limit=np.array(0.01))
    verdict = assess_m2(tmp_path, "fixes-v6", False)
    assert "sample_count" in _codes(verdict)
    assert verdict["fixes_passed"] is False


def test_f1_editable_limit_is_rejected(tmp_path):
    (tmp_path / "cases").mkdir()
    speed = np.zeros((2, 6))
    speed[:, 0] = 5.0
    pref = np.array([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
    np.savez(tmp_path / "cases" / "F4-speed.npz", t=np.array([0.0, 0.001]), v_hist=speed, p_ref=pref, limit=np.array(10.0))
    verdict = assess_m2(tmp_path, "fixes-v6", False)
    assert "threshold_mismatch" in _codes(verdict)
    assert "speed_limit" in _codes(verdict)


def test_f1_missing_family_is_rejected(tmp_path):
    (tmp_path / "cases").mkdir()
    np.savez(tmp_path / "cases" / "F4-speed.npz", t=np.array([0.0, 0.001]), v_hist=np.zeros((2, 6)), p_ref=np.zeros((2, 3)), limit=np.array(0.01))
    verdict = assess_m2(tmp_path, "fixes-v6", False)
    assert "missing_case" in _codes(verdict)


def test_f1_short_hash_list_is_rejected(tmp_path):
    for name in ("input_hash_before.json", "input_hash_after.json"):
        (tmp_path / name).write_text(json.dumps({"files": [], "aggregate_sha256": "0"}))
    verdict = assess_m2(tmp_path, "fixes-v6", True)
    assert "input_set" in _codes(verdict)


def test_f1_incomplete_run_is_rejected(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"incomplete": True}))
    (tmp_path / "report.json").write_text(json.dumps({"fixes_passed": True, "evidence_valid": True, "schema_version": "m2-fix-v6", "scope": "fixes_v6", "m3_ready": False, "hybrid_force_status": "disabled"}))
    verdict = assess_m2(tmp_path, "fixes-v6", False)
    assert "run_incomplete" in _codes(verdict)
    assert verdict["fixes_passed"] is False


def test_f1_failed_m1_is_rejected(tmp_path):
    bad = tmp_path / "m1"
    bad.mkdir()
    (bad / "run.json").write_text(json.dumps({"schema_version": "nope", "status": "failed", "incomplete": True}))
    (bad / "report.json").write_text("{}\n")
    import hashlib

    digest = hashlib.sha256((bad / "report.json").read_bytes()).hexdigest()
    (tmp_path / "m1_regression_binding.json").write_text(json.dumps({"regression_dir": str(bad), "baseline_dir": str(bad), "report_sha256": digest}))
    verdict = assess_m2(tmp_path, "fixes-v6", False)
    assert "m1_failed" in _codes(verdict)


def test_f1_missing_nodeid_is_rejected(tmp_path):
    node = required_nodeids()[0]
    (tmp_path / "execution.json").write_text(json.dumps({"exitstatus": 0, "collected": [node], "reports": [{"nodeid": node, "when": "call", "outcome": "passed"}, {"nodeid": node, "when": "setup", "outcome": "passed"}, {"nodeid": node, "when": "teardown", "outcome": "passed"}]}))
    verdict = assess_m2(tmp_path, "fixes-v6", False)
    assert "nodeid" in _codes(verdict)


def test_f1_teardown_failure_is_rejected(tmp_path):
    node = required_nodeids()[0]
    (tmp_path / "execution.json").write_text(
        json.dumps(
            {
                "exitstatus": 1,
                "collected": required_nodeids(),
                "reports": [
                    {"nodeid": node, "when": "call", "outcome": "passed"},
                    {"nodeid": node, "when": "setup", "outcome": "passed"},
                    {"nodeid": node, "when": "teardown", "outcome": "failed"},
                ],
            }
        )
    )
    verdict = assess_m2(tmp_path, "fixes-v6", False)
    assert "teardown" in _codes(verdict)
    assert "exitstatus" in _codes(verdict)


def test_f1_before_after_mismatch_is_rejected(tmp_path):
    (tmp_path / "input_hash_before.json").write_text(json.dumps({"files": [{"path": "requirements.txt", "sha256": "a"}], "aggregate_sha256": "a"}))
    (tmp_path / "input_hash_after.json").write_text(json.dumps({"files": [{"path": "requirements.txt", "sha256": "b"}], "aggregate_sha256": "b"}))
    verdict = assess_m2(tmp_path, "fixes-v6", False)
    assert "input_hash" in _codes(verdict)


def test_f1_bad_sample_is_rejected(tmp_path):
    (tmp_path / "cases").mkdir()
    np.savez(tmp_path / "cases" / "F4-speed.npz", t=np.array([0.001, 0.001]), v_hist=np.zeros((2, 6)), p_ref=np.zeros((2, 3)), limit=np.array(0.01))
    verdict = assess_m2(tmp_path, "fixes-v6", False)
    assert "bad_time" in _codes(verdict)


def test_f1_numeric_tamper_is_rejected(tmp_path):
    (tmp_path / "cases").mkdir()
    (tmp_path / "cases" / "F6-trans.json").write_text(json.dumps({"case_id": "F6-trans", "pos": [0.03, 0.0, 0.0], "p_ref": [0.0, 0.0, 0.0], "correction": [0.0, 0.0, 0.0], "v_ref": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}))
    verdict = assess_m2(tmp_path, "fixes-v6", False)
    assert "reference_error" in _codes(verdict)


def test_f1_summary_mismatch_is_rejected(tmp_path):
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "schema_version": "m2-fix-v6",
                "scope": "fixes_v6",
                "fixes_passed": True,
                "evidence_valid": True,
                "m3_ready": False,
                "hybrid_force_status": "disabled",
            }
        )
    )
    verdict = assess_m2(tmp_path, "fixes-v6", False)
    assert "summary_mismatch" in _codes(verdict)
    assert verdict["fixes_passed"] is False


def test_f1_fixes_v1_is_not_a_v2_pass(tmp_path):
    (tmp_path / "report.json").write_text("{}\n")
    verdict = assess_m2(tmp_path, "fixes-v1", False)
    assert verdict["fixes_passed"] is False
    assert "unsupported_schema" in _codes(verdict)
    assert any("not a V6 certification" in text for text in _texts(verdict))


def test_v3_recorded_contact_peaks_fail():
    from feedingrobot.controllers.v3spec import compare_recorded_peaks, compare_scalar

    left = {"contact_force": 4.176523, "tcp_force": 4.147093, "tcp_torque": 0.01727224}
    right = {"contact_force": 4.761058, "tcp_force": 4.731628, "tcp_torque": 0.01973564}
    assert {item["field"] for item in compare_recorded_peaks(left, right)} == {"contact_force", "tcp_force", "tcp_torque"}
    assert compare_recorded_peaks(left, left) == []
    assert compare_scalar("contact_force", 1.0, 1.1, 0.1, 0.02) is None
    assert compare_scalar("tcp_torque", 0.02, 0.022, 0.1, 0.002) is None
    assert compare_scalar("tcp_torque", 0.01727224, 0.01973564, 0.1, 0.002) is not None
    folder = Path("outputs/m2/acceptance/m2-fix-v2/cases")
    da = np.load(folder / "gear-s0-A.npz")
    db = np.load(folder / "gear-s0-B.npz")
    recorded = compare_recorded_peaks(
        {
            "contact_force": float(np.max(da["contact"])),
            "tcp_force": float(np.max(np.linalg.norm(da["wrench"][:, :3], axis=1))),
            "tcp_torque": float(np.max(np.linalg.norm(da["wrench"][:, 3:], axis=1))),
        },
        {
            "contact_force": float(np.max(db["contact"])),
            "tcp_force": float(np.max(np.linalg.norm(db["wrench"][:, :3], axis=1))),
            "tcp_torque": float(np.max(np.linalg.norm(db["wrench"][:, 3:], axis=1))),
        },
    )
    assert {item["field"] for item in recorded} >= {"contact_force", "tcp_force", "tcp_torque"}
