"""Recompute a finished run from its files. Does not simulate or edit the report."""

from __future__ import annotations

import argparse
import json
import sys

from feedingrobot.sim.model import repo_root
from feedingrobot.sim.report import assess_run
from feedingrobot.scripts.validate_m1 import _bundle_hash, _tracked_files


def main():
    parser = argparse.ArgumentParser(description="Check an M1 run directory without resimulating.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--check-current-workspace", action="store_true")
    args = parser.parse_args()
    run = repo_root() / args.run if not args.run.startswith("/") else __import__("pathlib").Path(args.run)
    verdict = assess_run(run)
    if args.check_current_workspace:
        recorded = json.loads((run / "input_hash_after.json").read_text())["aggregate_sha256"]
        config = repo_root() / "configs" / "m1_scene.json"
        current = _bundle_hash(repo_root(), _tracked_files(repo_root(), config))["aggregate_sha256"]
        if current != recorded:
            verdict["reasons"].append("workspace hash differs from the released run")
            verdict["m2_ready"] = False
            verdict["provenance_passed"] = False
            verdict["overall_status"] = "failed"
    print(json.dumps({key: verdict[key] for key in ("physics_passed", "provenance_passed", "visual_status", "m2_ready", "overall_status", "reasons")}, indent=2))
    if not verdict["m2_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
