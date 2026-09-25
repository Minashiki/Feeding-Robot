"""Recompute an M2 run from its files. Does not simulate or edit the report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedingrobot.controllers.acceptance import verify_finished_report
from feedingrobot.sim.model import repo_root


def assess_m2(run: Path, scope: str, check_workspace: bool) -> dict:
    return verify_finished_report(run, scope, check_workspace)


def main():
    parser = argparse.ArgumentParser(description="Check an M2 run directory without resimulating.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--scope", default="full", choices=("fixes-v1", "fixes-v2", "full"))
    parser.add_argument("--check-current-workspace", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_absolute():
        run = repo_root() / run
    verdict = assess_m2(run, args.scope, args.check_current_workspace)
    print(json.dumps({key: verdict[key] for key in ("fixes_passed", "m3_ready", "evidence_valid", "hybrid_force_status", "reasons")}, indent=2))
    if args.scope == "fixes-v2" and verdict["fixes_passed"] and verdict["m3_ready"] is False:
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
