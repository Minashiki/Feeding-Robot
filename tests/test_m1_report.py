"""The report gate must not mark a partial run ready for M2."""

from feedingrobot.sim.report import REQUIRED_CASES, assess


def _good():
    cases = {case_id: {"status": "passed", "metrics": {"ok": True}, "log_paths": ["trace.npz"]} for case_id in REQUIRED_CASES}
    return {
        "pytest_returncode": 0,
        "provenance_passed": True,
        "hashes_match": True,
        "visual_status": "passed",
        "cases": cases,
    }


def test_complete_report_is_ready():
    result = assess(_good())
    assert result["m2_ready"] and result["overall_status"] == "passed"


def test_missing_case_skips_and_bad_hash_are_not_ready():
    missing = _good()
    del missing["cases"]["T10_motion"]
    assert assess(missing)["m2_ready"] is False
    skipped = _good()
    skipped["cases"]["T11"]["status"] = "skipped"
    assert assess(skipped)["m2_ready"] is False
    wrong = _good()
    wrong["hashes_match"] = False
    assert assess(wrong)["m2_ready"] is False
    empty = _good()
    empty["cases"]["T13_T14"]["log_paths"] = []
    assert assess(empty)["m2_ready"] is False
    failed = _good()
    failed["pytest_returncode"] = 1
    assert assess(failed)["m2_ready"] is False
