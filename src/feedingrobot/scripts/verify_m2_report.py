"""Read an M2 report without simulating. Does not rewrite it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedingrobot.sim.model import repo_root


def main():
    parser = argparse.ArgumentParser(description="Check an M2 run directory without resimulating.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--check-current-workspace", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_absolute():
        run = repo_root() / run
    report = json.loads((run / "report.json").read_text())
    reasons = list(report.get("reasons") or [])
    if report.get("incomplete", True):
        reasons.append("run incomplete")
    if report.get("m3_ready") and reasons:
        reasons.append("m3_ready set while failures remain")
    if report.get("hybrid_force_status") not in {"disabled", "passed", "failed"}:
        reasons.append("hybrid status missing")
    if args.check_current_workspace and not (run / "report.json").is_file():
        reasons.append("missing report")
    verdict = {
        "m3_ready": bool(report.get("m3_ready")) and not reasons,
        "hybrid_force_status": report.get("hybrid_force_status"),
        "reasons": reasons,
    }
    print(json.dumps(verdict, indent=2))
    if not verdict["m3_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
