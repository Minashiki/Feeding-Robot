"""Recompute an M2 run from its files. Does not simulate or edit the report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from feedingrobot.controllers.acceptance import REMAINING_FULL_M2, REQUIRED_FILES, REQUIRED_NODE_PARTS
from feedingrobot.sim.model import repo_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unreadable {path.name}: {exc}") from exc


def assess_m2(run: Path, scope: str, check_workspace: bool) -> dict:
    reasons: list[str] = []
    if scope not in {"fixes-v1", "full"}:
        return {"fixes_passed": False, "m3_ready": False, "reasons": [f"unknown scope {scope}"]}
    missing = [name for name in REQUIRED_FILES if not (run / name).is_file()]
    if missing:
        reasons.append("missing " + missing[0])
        return {"fixes_passed": False, "m3_ready": False, "evidence_valid": False, "reasons": reasons, "scope": scope}
    try:
        report = _load(run / "report.json")
        execution = _load(run / "execution.json")
        manifest = _load(run / "case_manifest.json")
        recorded = _load(run / "input_hash_after.json")
        binding = _load(run / "m1_regression_binding.json")
    except RuntimeError as exc:
        return {"fixes_passed": False, "m3_ready": False, "evidence_valid": False, "reasons": [str(exc)], "scope": scope}
    if report.get("schema_version") != "m2-fix-v1":
        reasons.append("unknown schema")
    if report.get("scope") != "fixes_v1":
        reasons.append("scope mismatch")
    if report.get("hybrid_force_status") != "disabled":
        reasons.append("hybrid status is not disabled")
    if report.get("m3_ready") is True:
        reasons.append("summary claims m3_ready without a full matrix")
    collected = execution.get("collected") or []
    calls = [row for row in execution.get("reports") or [] if row.get("when") == "call"]
    for part in REQUIRED_NODE_PARTS:
        matched = [row for row in calls if part in row.get("nodeid", "")]
        if not any(part in node for node in collected):
            reasons.append(f"missing collected test {part}")
        elif not matched or any(row.get("outcome") != "passed" for row in matched):
            reasons.append(f"test not passed {part}")
    if any(row.get("outcome") in {"skipped", "xfailed"} for row in calls):
        reasons.append("skipped or xfailed test")
    cases = manifest.get("cases") or []
    if "F4-speed" not in cases:
        reasons.append("manifest missing F4-speed")
    trace = run / "cases" / "f4_speed.npz"
    if not trace.is_file():
        reasons.append("missing cases/f4_speed.npz")
    else:
        data = np.load(trace)
        speed = np.asarray(data["v_hist"], dtype=float)
        limit = float(data["limit"])
        if not np.all(np.isfinite(speed)):
            reasons.append("F4 trace non-finite")
        elif np.any(np.linalg.norm(speed.reshape(-1, 6)[:, :3], axis=1) > limit + 1e-9):
            reasons.append("F4 speed recomputation failed")
        if np.any(np.diff(np.asarray(data["t"], dtype=float)) < -1e-12):
            reasons.append("F4 time went backwards")
    if check_workspace:
        root = repo_root()
        current = []
        for rel in recorded.get("files") or []:
            path = root / rel["path"]
            if not path.is_file() or _sha256(path) != rel["sha256"]:
                reasons.append(f"workspace differs {rel['path']}")
                break
            current.append(rel["path"])
        if len(current) != len(recorded.get("files") or []):
            reasons.append("workspace file set differs")
    regression = Path(binding.get("path", ""))
    if not regression.is_file():
        reasons.append("m1 regression report missing")
    elif _sha256(regression) != binding.get("sha256"):
        reasons.append("m1 regression hash mismatch")
    evidence_valid = not reasons
    fixes_passed = evidence_valid and report.get("fixes_passed") is True
    if report.get("fixes_passed") is True and not evidence_valid:
        reasons.append("summary fixes_passed disagrees with recomputation")
        fixes_passed = False
    full_ready = False
    if scope == "full":
        reasons.append("full M2 still missing: " + REMAINING_FULL_M2[0])
        full_ready = False
    return {
        "fixes_passed": fixes_passed if scope == "fixes-v1" else False,
        "m3_ready": full_ready,
        "evidence_valid": evidence_valid,
        "hybrid_force_status": "disabled",
        "remaining_m2_requirements": list(REMAINING_FULL_M2),
        "reasons": reasons,
        "scope": scope,
    }


def main():
    parser = argparse.ArgumentParser(description="Check an M2 run directory without resimulating.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--scope", default="full", choices=("fixes-v1", "full"))
    parser.add_argument("--check-current-workspace", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_absolute():
        run = repo_root() / run
    verdict = assess_m2(run, args.scope, args.check_current_workspace)
    print(json.dumps({key: verdict[key] for key in ("fixes_passed", "m3_ready", "evidence_valid", "hybrid_force_status", "reasons")}, indent=2))
    if args.scope == "fixes-v1":
        if not verdict["fixes_passed"] or verdict["m3_ready"]:
            raise SystemExit(1)
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
