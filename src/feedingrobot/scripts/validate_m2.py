"""Build an M2 evidence directory. fixes-v1 cannot set m3_ready."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from feedingrobot.controllers.acceptance import REMAINING_FULL_M2
from feedingrobot.controllers.reference import ReferenceShaper
from feedingrobot.sim.model import repo_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path) -> dict:
    rows = []
    folders = [
        root / "src" / "feedingrobot" / "controllers",
        root / "src" / "feedingrobot" / "scripts",
        root / "tests",
        root / "configs",
    ]
    files = []
    for folder in folders:
        files.extend(path for path in folder.rglob("*") if path.is_file() and path.suffix in {".py", ".json", ".xml", ".md"})
    for rel in ("requirements.txt", "pyproject.toml", "assets/tests/m2_spring_surface.xml"):
        files.append(root / rel)
    for path in sorted(set(files)):
        if path.is_file():
            rows.append({"path": str(path.relative_to(root)), "sha256": _sha256(path)})
    return {"files": rows}


def main():
    parser = argparse.ArgumentParser(description="M2 acceptance runner")
    parser.add_argument("--scope", default="full", choices=("fixes-v1", "full"))
    parser.add_argument("--scene-config", default="configs/m1_scene.json")
    parser.add_argument("--controller-config", default="configs/m2_controller.json")
    parser.add_argument("--acceptance-config", default="configs/m2_acceptance.json")
    parser.add_argument("--m1-baseline", default="outputs/m1/acceptance/m1-fix-v3b")
    parser.add_argument("--m1-regression", default="outputs/m1/acceptance/m2-fix-v1-regression")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.scope != "fixes-v1":
        raise SystemExit("full M2 matrix is not runnable from this entry yet; use --scope fixes-v1")
    root = repo_root()
    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    (out / "cases").mkdir()
    (out / "run.json").write_text(json.dumps({"incomplete": True, "started": datetime.now(timezone.utc).isoformat()}, indent=2))
    execution_path = out / "execution.json"
    env = os.environ.copy()
    env["M1_EXECUTION_PATH"] = str(execution_path)
    env["M1_RUN_ID"] = out.name
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_m2_math.py", "tests/test_m2_tracking.py", "tests/test_m2_traj.py", "tests/test_m2_wrench.py", "tests/test_m2_guard.py", "tests/test_m2_fixes.py", "tests/test_m2_report.py", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
    )
    (out / "pytest.txt").write_text(proc.stdout + "\n" + proc.stderr)
    cfg = json.loads((root / args.controller_config).read_text())
    shaper = ReferenceShaper(cfg)
    shaper.anchor_pose(np.zeros(3), np.eye(3), np.zeros(7))
    shaper.v_lin[:] = [0.05, 0.0, 0.0]
    limits = {"v": 0.01, "w": 0.1, "a": 0.25, "alpha": 1.5, "pos_dev": 0.02, "rot_dev": 0.14}
    row = shaper.shape(np.array([0.05, 0, 0, 0, 0, 0.0]), np.zeros(3), np.eye(3), np.zeros(3), 0.001, limits, 1.0, 1.0, True, False, "run")
    np.savez(out / "cases" / "f4_speed.npz", t=np.array([0.0, 0.001]), v_hist=np.vstack([row["v_hist"], row["v_hist"]]), limit=np.array(0.01))
    hashed = _hash_tree(root)
    (out / "input_hash_after.json").write_text(json.dumps(hashed, indent=2))
    (out / "input_hash_before.json").write_text(json.dumps(hashed, indent=2))
    (out / "controller_config.json").write_text((root / args.controller_config).read_text())
    (out / "acceptance_config.json").write_text((root / args.acceptance_config).read_text())
    baseline = root / args.m1_baseline / "report.json"
    (out / "baseline.json").write_text(json.dumps({"path": str(baseline), "exists": baseline.is_file()}, indent=2))
    regression = root / args.m1_regression / "report.json"
    (out / "m1_regression_binding.json").write_text(json.dumps({"path": str(regression), "sha256": _sha256(regression) if regression.is_file() else None}, indent=2))
    (out / "case_manifest.json").write_text(json.dumps({"cases": ["F2-com", "F3-pulse", "F4-speed", "F5-blend", "F6-saturation", "F7-command"]}, indent=2))
    (out / "environment.json").write_text(json.dumps({"python": sys.version}, indent=2))
    report = {
        "schema_version": "m2-fix-v1",
        "scope": "fixes_v1",
        "incomplete": False,
        "fixes_passed": proc.returncode == 0 and regression.is_file(),
        "m3_ready": False,
        "hybrid_force_status": "disabled",
        "evidence_valid": False,
        "remaining_m2_requirements": list(REMAINING_FULL_M2),
        "pytest_returncode": proc.returncode,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "run.json").write_text(json.dumps({"incomplete": False, "scope": "fixes_v1", "m3_ready": False}, indent=2))
    print(json.dumps({"pytest": proc.returncode, "output": str(out), "m3_ready": False}, indent=2))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
