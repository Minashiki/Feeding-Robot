"""Recompute an M2 run from its files. Does not simulate or edit the report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedingrobot.validation.m2.legacy import verify_finished_report
from feedingrobot.validation.m2.spec import SCHEMA
from feedingrobot.sim.model import repo_root


def assess_m2(run: Path, scope: str, check_workspace: bool) -> dict:
    try:
        header = json.loads((run / "run.json").read_text())
    except (OSError, ValueError):
        header = {}
    if header.get("schema_version") == SCHEMA:
        from feedingrobot.validation.m2.package import evaluate
        verdict = evaluate(run, check_workspace=check_workspace, calibration=header.get("stage") == "calibration")
        if scope != ("calibration" if header.get("stage") == "calibration" else "full"):
            verdict["scope_passed"] = verdict["m3_ready"] = verdict["evidence_valid"] = verdict["fixes_passed"] = False
            verdict["reasons"].append({"code":"full_scope_required"})
        return verdict
    if header.get('schema_version') not in {None, 'm2-fix-v6'}:
        return {'fixes_passed': False, 'evidence_valid': False, 'scope_passed': False,
                'm3_ready': False, 'hybrid_force_status': 'disabled',
                'reasons': [{'code': 'unsupported_schema', 'schema': header.get('schema_version')}]}
    return verify_finished_report(run, scope, check_workspace)


def main():
    parser = argparse.ArgumentParser(description="Check an M2 run directory without resimulating.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--scope", default="full", choices=("fixes-v1", "fixes-v2", "fixes-v3", "fixes-v4", "fixes-v5", "fixes-v6", "calibration", "full"))
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
    if args.scope == "calibration" and verdict.get("scope_passed") and verdict["m3_ready"] is False:
        return
    if args.scope == "fixes-v6" and verdict.get("scope_passed") and verdict["m3_ready"] is False:
        return
    if args.scope == "full" and verdict.get("scope_passed") and verdict["m3_ready"] is True:
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
