"""F1: invalid evidence must not pass, including a matching self-declared limit."""

import json

import numpy as np

from feedingrobot.controllers.acceptance import required_nodeids
from feedingrobot.scripts.verify_m2_report import assess_m2


def _touch(path):
    path.write_text("{}\n")


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
    verdict = assess_m2(tmp_path, "fixes-v2", False)
    assert any("missing samples" in item for item in verdict["reasons"])
    assert verdict["fixes_passed"] is False


def test_f1_editable_limit_is_rejected(tmp_path):
    (tmp_path / "cases").mkdir()
    speed = np.zeros((2, 6))
    speed[:, 0] = 5.0
    pref = np.array([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
    np.savez(tmp_path / "cases" / "F4-speed.npz", t=np.array([0.0, 0.001]), v_hist=speed, p_ref=pref, limit=np.array(10.0))
    verdict = assess_m2(tmp_path, "fixes-v2", False)
    assert any("threshold mismatch" in item for item in verdict["reasons"])
    assert any("exceeds" in item for item in verdict["reasons"])


def test_f1_missing_family_is_rejected(tmp_path):
    (tmp_path / "cases").mkdir()
    np.savez(tmp_path / "cases" / "F4-speed.npz", t=np.array([0.0, 0.001]), v_hist=np.zeros((2, 6)), p_ref=np.zeros((2, 3)), limit=np.array(0.01))
    verdict = assess_m2(tmp_path, "fixes-v2", False)
    assert any(item.startswith("missing case F2-") for item in verdict["reasons"])


def test_f1_short_hash_list_is_rejected(tmp_path):
    for name in ("input_hash_before.json", "input_hash_after.json"):
        (tmp_path / name).write_text(json.dumps({"files": [], "aggregate_sha256": "0"}))
    verdict = assess_m2(tmp_path, "fixes-v2", True)
    assert any("input set differs" in item for item in verdict["reasons"])


def test_f1_incomplete_run_is_rejected(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"incomplete": True}))
    (tmp_path / "report.json").write_text(json.dumps({"fixes_passed": True, "evidence_valid": True, "schema_version": "m2-fix-v2", "scope": "fixes_v2", "m3_ready": False, "hybrid_force_status": "disabled"}))
    verdict = assess_m2(tmp_path, "fixes-v2", False)
    assert any("run incomplete" in item for item in verdict["reasons"])
    assert verdict["fixes_passed"] is False


def test_f1_failed_m1_is_rejected(tmp_path):
    bad = tmp_path / "m1"
    bad.mkdir()
    (bad / "run.json").write_text(json.dumps({"schema_version": "nope", "status": "failed", "incomplete": True}))
    (bad / "report.json").write_text("{}\n")
    digest_path = bad / "report.json"
    import hashlib

    digest = hashlib.sha256(digest_path.read_bytes()).hexdigest()
    (tmp_path / "m1_regression_binding.json").write_text(json.dumps({"regression_dir": str(bad), "baseline_dir": str(bad), "report_sha256": digest}))
    verdict = assess_m2(tmp_path, "fixes-v2", False)
    assert any("m1 regression failed" in item for item in verdict["reasons"])


def test_f1_missing_nodeid_is_rejected(tmp_path):
    node = required_nodeids()[0]
    (tmp_path / "execution.json").write_text(json.dumps({"exitstatus": 0, "collected": [node], "reports": [{"nodeid": node, "when": "call", "outcome": "passed"}, {"nodeid": node, "when": "setup", "outcome": "passed"}, {"nodeid": node, "when": "teardown", "outcome": "passed"}]}))
    verdict = assess_m2(tmp_path, "fixes-v2", False)
    assert any("missing nodeid" in item for item in verdict["reasons"])


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
    verdict = assess_m2(tmp_path, "fixes-v2", False)
    assert any("teardown failed" in item for item in verdict["reasons"])
    assert any("exitstatus" in item for item in verdict["reasons"])


def test_f1_before_after_mismatch_is_rejected(tmp_path):
    (tmp_path / "input_hash_before.json").write_text(json.dumps({"files": [{"path": "requirements.txt", "sha256": "a"}], "aggregate_sha256": "a"}))
    (tmp_path / "input_hash_after.json").write_text(json.dumps({"files": [{"path": "requirements.txt", "sha256": "b"}], "aggregate_sha256": "b"}))
    verdict = assess_m2(tmp_path, "fixes-v2", False)
    assert any("input hash changed" in item for item in verdict["reasons"])


def test_f1_bad_sample_is_rejected(tmp_path):
    (tmp_path / "cases").mkdir()
    np.savez(tmp_path / "cases" / "F4-speed.npz", t=np.array([0.001, 0.001]), v_hist=np.zeros((2, 6)), p_ref=np.zeros((2, 3)), limit=np.array(0.01))
    verdict = assess_m2(tmp_path, "fixes-v2", False)
    assert any("F4-speed bad field t" in item for item in verdict["reasons"])


def test_f1_numeric_tamper_is_rejected(tmp_path):
    (tmp_path / "cases").mkdir()
    (tmp_path / "cases" / "F6-trans.json").write_text(json.dumps({"case_id": "F6-trans", "p_ref": [0.0, 0.0, 0.0], "correction": [0.0, 0.0, 0.0], "v_ref": 0.0}))
    (tmp_path / "evidence_manifest.json").write_text(json.dumps({"files": [{"path": "cases/F6-trans.json", "sha256": "updated"}]}))
    verdict = assess_m2(tmp_path, "fixes-v2", False)
    assert any("F6-trans reference error" in item for item in verdict["reasons"])


def test_f1_summary_mismatch_is_rejected(tmp_path):
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "schema_version": "m2-fix-v2",
                "scope": "fixes_v2",
                "fixes_passed": True,
                "evidence_valid": True,
                "m3_ready": False,
                "hybrid_force_status": "disabled",
            }
        )
    )
    verdict = assess_m2(tmp_path, "fixes-v2", False)
    assert any("summary disagrees" in item for item in verdict["reasons"])
    assert verdict["fixes_passed"] is False


def test_f1_fixes_v1_is_not_a_v2_pass(tmp_path):
    _touch(tmp_path / "report.json")
    verdict = assess_m2(tmp_path, "fixes-v1", False)
    assert verdict["fixes_passed"] is False
    assert any("not a V2 certification" in item for item in verdict["reasons"])
