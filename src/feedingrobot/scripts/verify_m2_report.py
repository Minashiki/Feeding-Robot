"""Recompute an M2 run from its files. Does not simulate or edit the report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedingrobot.controllers.acceptance import verify_finished_report
from feedingrobot.sim.model import repo_root


def assess_m2(run: Path, scope: str, check_workspace: bool) -> dict:
    try:
        header = json.loads((run / "run.json").read_text())
    except (OSError, ValueError):
        header = {}
    if header.get("schema_version") in {"m2-full-v1", "m2-full-v2"}:
        from feedingrobot.controllers.full_acceptance import evaluate
        verdict = evaluate(run, check_workspace=check_workspace, calibration=header.get("stage") == "calibration")
        if scope != "full" or header.get("stage") != "full":
            verdict["scope_passed"] = verdict["m3_ready"] = False
            verdict["reasons"].append({"code":"full_scope_required"})
        return verdict
    return verify_finished_report(run, scope, check_workspace)


def main():
    parser = argparse.ArgumentParser(description="Check an M2 run directory without resimulating.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--scope", default="full", choices=("fixes-v1", "fixes-v2", "fixes-v3", "fixes-v4", "fixes-v5", "fixes-v6", "full"))
    parser.add_argument("--check-current-workspace", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_absolute():
        run = repo_root() / run
    verdict = assess_m2(run, args.scope, args.check_current_workspace)
    printable = []
    for item in verdict["reasons"]:
        if isinstance(item, dict):
            printable.append(item.get("text") or item.get("code"))
        else:
            printable.append(str(item))
    print(
        json.dumps(
            {
                "fixes_passed": verdict["fixes_passed"],
                "m3_ready": verdict["m3_ready"],
                "evidence_valid": verdict["evidence_valid"],
                "scope_passed": verdict.get("scope_passed"),
                "hybrid_force_status": verdict["hybrid_force_status"],
                "reasons": printable,
            },
            indent=2,
        )
    )
    if args.scope == "fixes-v6" and verdict.get("scope_passed") and verdict["m3_ready"] is False:
        return
    if args.scope == "full" and verdict.get("scope_passed") and verdict["m3_ready"] is True:
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
