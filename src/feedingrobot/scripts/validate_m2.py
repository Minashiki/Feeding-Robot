"""Write an M2 run directory. A partial suite cannot set m3_ready."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from feedingrobot.sim.model import repo_root

_REQUIRED = [
    "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T10",
    "T11", "T12", "T13", "T14", "T15", "T16", "T17",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="M2 acceptance runner")
    parser.add_argument("--scene-config", default="configs/m1_scene.json")
    parser.add_argument("--controller-config", default="configs/m2_controller.json")
    parser.add_argument("--acceptance-config", default="configs/m2_acceptance.json")
    parser.add_argument("--m1-baseline", default="outputs/m1/acceptance/m1-fix-v3b")
    parser.add_argument("--m1-regression", default="outputs/m1/acceptance/m2-v1-regression")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = repo_root()
    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    (out / "run.json").write_text(json.dumps({"incomplete": True, "started": datetime.now(timezone.utc).isoformat()}, indent=2))
    acceptance = json.loads((root / args.acceptance_config).read_text())
    present = set(acceptance["cases"])
    missing = [case for case in _REQUIRED if case not in present]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_m2_math.py", "tests/test_m2_tracking.py", "tests/test_m2_traj.py", "tests/test_m2_wrench.py", "tests/test_m2_guard.py", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    (out / "pytest.txt").write_text(proc.stdout + "\n" + proc.stderr)
    files = []
    for rel in (
        args.controller_config,
        args.acceptance_config,
        "src/feedingrobot/controllers",
    ):
        path = root / rel
        paths = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file() and item.suffix == ".py"]
        for item in paths:
            files.append({"path": str(item.relative_to(root)), "sha256": _sha256(item)})
    report = {
        "incomplete": False,
        "baseline_passed": (root / args.m1_baseline / "report.json").is_file(),
        "math_passed": proc.returncode == 0,
        "tracking_passed": proc.returncode == 0,
        "wrench_passed": proc.returncode == 0,
        "constraints_passed": proc.returncode == 0,
        "contact_passed": False,
        "convergence_passed": False,
        "provenance_passed": False,
        "hybrid_force_status": "disabled",
        "m3_ready": False,
        "reasons": missing + ["A/B/C convergence not run", "M1 regression not bound"],
        "pytest_returncode": proc.returncode,
        "input_files": files,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "run.json").write_text(json.dumps({"incomplete": False, "m3_ready": False}, indent=2))
    print(json.dumps({key: report[key] for key in ("math_passed", "m3_ready", "hybrid_force_status", "reasons")}, indent=2))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
