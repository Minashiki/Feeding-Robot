"""F1: a summary with no trajectory must not pass."""

import json

from feedingrobot.scripts.verify_m2_report import assess_m2


def test_f1_summary_only_is_rejected(tmp_path):
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "incomplete": False,
                "m3_ready": True,
                "hybrid_force_status": "disabled",
                "reasons": [],
            }
        )
    )
    for scope in ("fixes-v1", "full"):
        verdict = assess_m2(tmp_path, scope, check_workspace=True)
        assert verdict["fixes_passed"] is False
        assert verdict["m3_ready"] is False
        assert verdict["reasons"]
