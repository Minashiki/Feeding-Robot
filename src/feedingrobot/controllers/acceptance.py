"""Frozen M2 V2 acceptance spec. Thresholds here override anything stored in a trajectory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from feedingrobot.scripts.validate_m1 import _tracked_files
from feedingrobot.sim.model import repo_root
from feedingrobot.sim.report import assess_run

SCHEMA_VERSION = "m2-fix-v2"
SCOPE_NAME = "fixes_v2"
F4_SPEED_LIMIT_M_S = 0.01
JOINT_SPEED_LIMIT = 0.5
JOINT_PREDICT_LIMIT = 0.4
TAU_RATE_NM_S = 2000.0
POS_DEV_M = 0.02
SUMMARY_ATOL = 1e-9
SUMMARY_RTOL = 1e-7

REQUIRED_FILES = (
    "report.json",
    "run.json",
    "execution.json",
    "environment.json",
    "case_manifest.json",
    "evidence_manifest.json",
    "input_hash_before.json",
    "input_hash_after.json",
    "baseline.json",
    "m1_regression_binding.json",
    "controller_config.json",
    "acceptance_config.json",
    "comparisons.json",
    "pytest.txt",
    "calibration.md",
)

LEGACY_NODEIDS = (
    "tests/test_m2_math.py::test_t01_so3_and_quaternion_roundtrip",
    "tests/test_m2_math.py::test_t01_jacobian_matches_central_difference",
    "tests/test_m2_math.py::test_t01_small_rotation_restores",
    "tests/test_m2_math.py::test_t02_mass_spd_and_leakage",
    "tests/test_m2_math.py::test_t03_torque_and_command_rejection",
    "tests/test_m2_tracking.py::test_t04_hold_three_poses",
    "tests/test_m2_tracking.py::test_t04_position_step_x",
    "tests/test_m2_wrench.py::test_tool_mass_is_80_grams",
    "tests/test_m2_wrench.py::test_t06_unloaded_and_known_force",
    "tests/test_m2_wrench.py::test_t07_food_is_not_zeroed",
    "tests/test_m2_guard.py::test_t12_contact_whitelist_is_ordered",
    "tests/test_m2_guard.py::test_t13_expired_command_latches_and_stops",
    "tests/test_m2_guard.py::test_t13_reset_clears_fault",
    "tests/test_m2_traj.py::test_t05_circle_stays_within_bounds",
    "tests/test_m2_traj.py::test_t14_replay_matches",
    "tests/test_m2_fixes.py::test_f4_new_speed_bound_projects_history",
    "tests/test_m2_fixes.py::test_f4_prohibit_clears_reference_velocity",
    "tests/test_m2_fixes.py::test_f5_interrupted_blend_stays_continuous",
    "tests/test_m2_fixes.py::test_f5_repeat_command_does_not_restart",
    "tests/test_m2_fixes.py::test_f7_rejects_nonfinite_and_keeps_the_live_command",
    "tests/test_m2_fixes.py::test_f2_com_velocity_matches_jacobian",
    "tests/test_m2_fixes.py::test_f3_force_pulse_latches_on_the_post_step_sample",
    "tests/test_m2_fixes.py::test_v2_f5_return_to_free_is_continuous",
    "tests/test_m2_fixes.py::test_v2_f6_pause_still_limits_deviation",
    "tests/test_m2_fixes.py::test_v2_f4_final_candidate_respects_joint_limit",
    "tests/test_m2_fixes.py::test_f6_commit_uses_applied_torque_and_pauses",
    "tests/test_m2_report.py::test_f1_summary_only_is_rejected",
    "tests/test_m2_report.py::test_f1_empty_speed_array_is_rejected",
    "tests/test_m2_report.py::test_f1_editable_limit_is_rejected",
    "tests/test_m2_report.py::test_f1_missing_family_is_rejected",
    "tests/test_m2_report.py::test_f1_short_hash_list_is_rejected",
    "tests/test_m2_report.py::test_f1_incomplete_run_is_rejected",
    "tests/test_m2_report.py::test_f1_failed_m1_is_rejected",
    "tests/test_m2_report.py::test_f1_missing_nodeid_is_rejected",
    "tests/test_m2_report.py::test_f1_teardown_failure_is_rejected",
    "tests/test_m2_report.py::test_f1_before_after_mismatch_is_rejected",
    "tests/test_m2_report.py::test_f1_bad_sample_is_rejected",
    "tests/test_m2_report.py::test_f1_numeric_tamper_is_rejected",
    "tests/test_m2_report.py::test_f1_summary_mismatch_is_rejected",
)

REMAINING_FULL_M2 = (
    "T08 spring surface",
    "T09 stiffness trend",
    "T10 blocked release on a physical wall",
    "T11 force pulse recovery",
    "T15 noise and delay on T05/T08",
    "T16 P0 bridge",
    "T17 full-matrix tamper cases beyond fixes-v2",
    "formal seeds 3-8",
    "full A/B/C matrix from M2Plan",
)

INPUT_PREFIXES = (
    "src/feedingrobot/controllers/",
    "src/feedingrobot/sim/",
    "src/feedingrobot/scripts/",
    "tests/",
    "configs/m1_scene.json",
    "configs/m2_controller.json",
    "configs/m2_acceptance.json",
    "requirements.txt",
    "pyproject.toml",
    "third_party_manifest.json",
    "assets/tests/m2_spring_surface.xml",
)


def f4_joint_case_ids() -> list[str]:
    return [f"F4-j{joint}-{side}-{motion}" for joint in range(7) for side in ("upper", "lower") for motion in ("out", "in", "hist")]


def f5_case_ids() -> list[str]:
    gears = ("FREE", "ACQUIRE", "MOUTH")
    rows = []
    for src in gears:
        for dst in gears:
            if src == dst:
                continue
            for interrupt_ms in (50, 150):
                for dt in (0.001, 0.0005):
                    rows.append(f"F5-{src}-{dst}-{interrupt_ms}-{dt}")
    return rows


def physical_case_ids() -> list[str]:
    families = ("pause", "release", "gear", "wrench", "takeover")
    return [f"{family}-s{seed}-{variant}" for family in families for seed in (0, 1, 2) for variant in ("A", "B", "C")]


def required_case_ids() -> list[str]:
    rows = [
        "F4-speed",
        "F4-pair",
        "F4-recover",
        "F4-scale-0",
        "F4-scale-0.1",
        "F4-scale-1",
        *f4_joint_case_ids(),
        *f5_case_ids(),
        "F5-repeat",
        "F5-return-limits",
        "F6-trans",
        "F6-rot",
        "F6-both",
        "F6-within",
        "F6-pause1000",
        "F6-prohibit",
        "F6-stop",
        "F6-power-on",
        "F2-com-0",
        "F2-com-1",
        "F2-com-2",
        "F3-pulse-A",
        "F3-pulse-B",
        "F7-commands",
        "reset-100",
        "replay-s0",
        "replay-s1",
        "replay-s2",
        *physical_case_ids(),
    ]
    if len(rows) != len(set(rows)):
        raise RuntimeError("duplicate V2 case id")
    return rows


def required_nodeids() -> list[str]:
    nodes = list(LEGACY_NODEIDS)
    for case_id in f4_joint_case_ids():
        nodes.append(f"tests/test_m2_v2.py::test_f4_joint_cell[{case_id}]")
    for case_id in ("F4-speed", "F4-pair", "F4-recover", "F4-scale-0", "F4-scale-0.1", "F4-scale-1"):
        nodes.append(f"tests/test_m2_v2.py::test_f4_case[{case_id}]")
    for case_id in f5_case_ids():
        nodes.append(f"tests/test_m2_v2.py::test_f5_blend[{case_id}]")
    nodes.append("tests/test_m2_v2.py::test_f5_repeat[F5-repeat]")
    nodes.append("tests/test_m2_v2.py::test_f5_return_limits[F5-return-limits]")
    for case_id in ("F6-trans", "F6-rot", "F6-both", "F6-within", "F6-pause1000", "F6-prohibit", "F6-stop", "F6-power-on"):
        nodes.append(f"tests/test_m2_v2.py::test_f6_case[{case_id}]")
    for case_id in ("F2-com-0", "F2-com-1", "F2-com-2"):
        nodes.append(f"tests/test_m2_v2.py::test_f2_pose[{case_id}]")
    nodes.append("tests/test_m2_v2.py::test_f3_pulse[F3-pulse-A]")
    nodes.append("tests/test_m2_v2.py::test_f3_pulse[F3-pulse-B]")
    nodes.append("tests/test_m2_v2.py::test_f7_commands[F7-commands]")
    nodes.append("tests/test_m2_v2.py::test_reset_pollution[reset-100]")
    for case_id in ("replay-s0", "replay-s1", "replay-s2"):
        nodes.append(f"tests/test_m2_v2.py::test_replay[{case_id}]")
    for case_id in physical_case_ids():
        nodes.append(f"tests/test_m2_v2.py::test_physical[{case_id}]")
    return nodes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_inputs(root: Path | None = None) -> dict:
    """Independent input set. Does not read a report's own file list."""
    root = repo_root() if root is None else Path(root)
    files = list(_tracked_files(root, root / "configs" / "m1_scene.json"))
    for folder in (root / "src" / "feedingrobot" / "controllers",):
        found = [path.resolve() for path in folder.rglob("*.py") if path.is_file()]
        if not found:
            raise RuntimeError(f"missing required input directory {folder}")
        files.extend(found)
    for rel in ("configs/m2_controller.json", "configs/m2_acceptance.json", "assets/tests/m2_spring_surface.xml"):
        path = (root / rel).resolve()
        if not path.is_file():
            raise RuntimeError(f"missing required input {rel}")
        files.append(path)
    missing = sorted({str(path.relative_to(root)) for path in files if not path.is_file()})
    if missing:
        raise RuntimeError("missing required input " + missing[0])
    rows = [{"path": str(path.resolve().relative_to(root)), "sha256": _sha256(path.resolve())} for path in sorted(set(files))]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode())
        digest.update(b"\0")
        digest.update(row["sha256"].encode())
        digest.update(b"\n")
    return {"files": rows, "aggregate_sha256": digest.hexdigest()}


def _load(path: Path):
    return json.loads(path.read_text())


def _finite(arr) -> bool:
    return bool(np.all(np.isfinite(np.asarray(arr, dtype=float))))


def _case_path(run: Path, case_id: str) -> Path | None:
    js = run / "cases" / f"{case_id}.json"
    nz = run / "cases" / f"{case_id}.npz"
    if js.is_file() and nz.is_file():
        return None
    if js.is_file():
        return js
    if nz.is_file():
        return nz
    return None


def _check_speed_case(path: Path, reasons: list[str]) -> None:
    data = np.load(path)
    speed = np.asarray(data["v_hist"], dtype=float)
    times = np.asarray(data["t"], dtype=float)
    pref = np.asarray(data["p_ref"], dtype=float)
    if speed.size == 0 or times.size == 0:
        reasons.append("F4-speed missing samples")
        return
    if "limit" in data.files and abs(float(np.asarray(data["limit"]).reshape(-1)[0]) - F4_SPEED_LIMIT_M_S) > 1e-15:
        reasons.append("F4-speed threshold mismatch")
    if speed.ndim != 2 or speed.shape[1] != 6 or len(speed) < 2:
        reasons.append("F4-speed bad field v_hist")
        return
    if not _finite(speed) or not _finite(times) or not _finite(pref):
        reasons.append("F4-speed bad field nonfinite")
        return
    if np.any(np.diff(times) <= 1e-12):
        reasons.append("F4-speed bad field t")
        return
    linear = np.linalg.norm(speed[:, :3], axis=1)
    if np.any(linear > F4_SPEED_LIMIT_M_S + 1e-9):
        reasons.append("F4-speed exceeds fixed limit")
    dt = np.diff(times)
    step = pref[1:] - pref[:-1]
    if not np.allclose(step, speed[:-1, :3] * dt[:, None], atol=SUMMARY_ATOL, rtol=SUMMARY_RTOL):
        reasons.append("F4-speed bad field integration")


def _check_unit_json(case_id: str, row: dict, reasons: list[str]) -> None:
    if row.get("case_id") != case_id:
        reasons.append(f"{case_id} bad field case_id")
        return
    if case_id.startswith("F4-j") or case_id in {"F4-pair", "F4-recover"}:
        outward = float(row["outward_speed"])
        if not np.isfinite(outward) or outward > 1e-9:
            reasons.append(f"{case_id} joint speed")
        if case_id.endswith("-in") or case_id == "F4-recover":
            inward = float(row["inward_speed"])
            if not np.isfinite(inward) or inward <= 1e-6:
                reasons.append(f"{case_id} joint speed")
    elif case_id.startswith("F4-scale-"):
        expected = {"F4-scale-0": 0.0, "F4-scale-0.1": 0.005, "F4-scale-1": 0.05}[case_id]
        if abs(float(row["speed_limit"]) - expected) > 1e-12:
            reasons.append(f"{case_id} threshold mismatch")
        if float(row["actual_speed"]) > expected + 1e-9:
            reasons.append(f"{case_id} speed exceeds fixed limit")
    elif case_id.startswith("F5-") and case_id not in {"F5-repeat", "F5-return-limits"}:
        gains = np.asarray(row["gains"], dtype=float)
        if gains.ndim != 2 or gains.shape[0] < 2 or gains.shape[1] != 4:
            reasons.append(f"{case_id} bad field gains")
            return
        start = np.asarray(row["k_start"], dtype=float)
        target = np.asarray(row["k_target"], dtype=float)
        dt = float(row["dt"])
        jump = np.max(np.abs(np.diff(gains, axis=0)), axis=1)
        allow = np.max(np.abs(target - start)) * dt / 0.2 + 1e-9
        if np.any(jump > allow + 1e-12):
            reasons.append(f"{case_id} gain jump")
    elif case_id == "F5-return-limits":
        if float(row["limit_during"]) > 0.01 + 1e-12 or abs(float(row["limit_after"]) - 0.05) > 1e-12:
            reasons.append("F5-return-limits gain jump")
    elif case_id == "F5-repeat":
        if row.get("finished") is not True:
            reasons.append("F5-repeat gain jump")
    elif case_id == "F6-trans":
        pref = np.asarray(row["p_ref"], dtype=float)
        corr = np.asarray(row["correction"], dtype=float)
        if np.max(np.abs(pref - np.array([0.01, 0.0, 0.0]))) > 1e-9:
            reasons.append("F6-trans reference error")
        if np.max(np.abs(corr - np.array([0.01, 0.0, 0.0]))) > 1e-9:
            reasons.append("F6-trans reference error")
        if abs(float(row["v_ref"])) > 1e-12:
            reasons.append("F6-trans reference error")
    elif case_id in {"F6-rot", "F6-both", "F6-within", "F6-pause1000", "F6-prohibit", "F6-stop", "F6-power-on"}:
        if float(row["error"]) > float(row["limit"]) + 1e-9:
            reasons.append(f"{case_id} reference error")
        if "rot_error" in row and float(row["rot_error"]) > float(row["rot_limit"]) + 1e-9:
            reasons.append(f"{case_id} reference error")
        if case_id == "F6-within" and np.max(np.abs(np.asarray(row["p_ref"], dtype=float))) > 1e-12:
            reasons.append("F6-within reference error")
    elif case_id.startswith("F2-com-"):
        if float(row["vel_error"]) > 1e-8 or float(row["acc_error"]) > 1e-5:
            reasons.append(f"{case_id} bad field error")
    elif case_id.startswith("F3-pulse-"):
        if row.get("reason") != "wrench" or int(row["fault_tick"]) != int(row["sample_tick"]) or row.get("next_mode") != "STOPPING":
            reasons.append(f"{case_id} bad field event")
        if int(row["filter_delta"]) != 2:
            reasons.append(f"{case_id} bad field filter")
    elif case_id == "F7-commands":
        for key in ("nonfinite", "stale", "conflict", "idempotent"):
            if row.get(key) is not True:
                reasons.append("F7-commands bad field event")
                break
    elif case_id == "reset-100":
        if row.get("clean") is not True or int(row.get("n", 0)) != 100:
            reasons.append("reset-100 bad field event")
    elif case_id.startswith("replay-"):
        if float(row["q_error"]) > 1e-9 or float(row["tcp_error"]) > 1e-9 or float(row["wrench_error"]) > 1e-7:
            reasons.append(f"{case_id} bad field error")


def _check_physical(case_id: str, path: Path, reasons: list[str]) -> dict:
    data = np.load(path)
    needed = ("t", "dq", "tau", "tcp", "p_ref", "v_ref", "wrench", "contact")
    for key in needed:
        if key not in data.files:
            reasons.append(f"{case_id} bad field {key}")
            return {}
    t = np.asarray(data["t"], dtype=float)
    dq = np.asarray(data["dq"], dtype=float)
    tau = np.asarray(data["tau"], dtype=float)
    if t.size < 2 or dq.shape[0] != t.size or tau.shape != dq.shape:
        reasons.append(f"{case_id} bad field t")
        return {}
    if not all(_finite(np.asarray(data[key], dtype=float)) for key in needed):
        reasons.append(f"{case_id} bad field nonfinite")
        return {}
    if np.any(np.diff(t) <= 1e-12):
        reasons.append(f"{case_id} bad field t")
        return {}
    dt = float(np.asarray(data["dt"]).reshape(-1)[0])
    if abs(float(np.median(np.diff(t))) - dt) > 1e-9:
        reasons.append(f"{case_id} bad field dt")
    if np.max(np.abs(dq)) > JOINT_SPEED_LIMIT + 1e-9:
        reasons.append(f"{case_id} joint speed")
    slope = np.max(np.abs(np.diff(tau, axis=0)))
    if slope > TAU_RATE_NM_S * dt + 1e-9:
        reasons.append(f"{case_id} torque rate")
    origin = np.asarray(data["tau0"], dtype=float).reshape(7) if "tau0" in data.files else np.zeros(7)
    if np.max(np.abs(tau[0] - origin)) > TAU_RATE_NM_S * dt + 1e-9:
        reasons.append(f"{case_id} torque rate")
    if "k" in data.files:
        gains = np.asarray(data["k"], dtype=float)
        if gains.shape[0] == t.size and gains.shape[0] >= 2:
            jump = np.max(np.abs(np.diff(gains, axis=0)))
            if jump > 200.0 * dt / 0.2 + 1e-9:
                reasons.append(f"{case_id} gain jump")
    err = np.linalg.norm(np.asarray(data["p_ref"], dtype=float) - np.asarray(data["tcp"], dtype=float), axis=1)
    if np.any(err > POS_DEV_M + 1e-9):
        reasons.append(f"{case_id} reference error")
    dist = np.asarray(data["min_dist"], dtype=float) if "min_dist" in data.files else np.zeros(1)
    if np.any(dist < -0.001):
        reasons.append(f"{case_id} bad field contact")
    return {"err": err, "tau": tau, "wrench": np.asarray(data["wrench"], dtype=float), "contact": np.asarray(data["contact"], dtype=float)}


def _abc(metrics: dict, reasons: list[str]) -> dict:
    out = {}
    for family in ("pause", "release", "gear", "wrench", "takeover"):
        for seed in (0, 1, 2):
            base = metrics.get(f"{family}-s{seed}-A")
            if not base:
                continue
            for variant in ("B", "C"):
                other = metrics.get(f"{family}-s{seed}-{variant}")
                if not other:
                    reasons.append(f"{family}-s{seed}-{variant} missing case")
                    continue
                a = float(np.max(base["err"]))
                b = float(np.max(other["err"]))
                limit = max(0.1 * max(a, b), 1e-4)
                delta = abs(a - b)
                out[f"{family}-s{seed}-{variant}"] = delta
                if delta > limit + 1e-12:
                    reasons.append(f"{family}-s{seed}-{variant} reference error")
    return out


def _execution_ok(execution: dict, reasons: list[str]) -> None:
    if int(execution.get("exitstatus", 1)) != 0:
        reasons.append("exitstatus")
    reports = execution.get("reports") or []
    if any(row.get("outcome") in {"skipped", "xfailed"} for row in reports):
        reasons.append("skipped or xfailed test")
    if any(row.get("when") == "teardown" and row.get("outcome") != "passed" for row in reports):
        reasons.append("teardown failed")
    collected = set(execution.get("collected") or [])
    calls = {}
    setups = {}
    teardowns = {}
    for row in reports:
        bucket = {"call": calls, "setup": setups, "teardown": teardowns}.get(row.get("when"))
        if bucket is not None:
            bucket.setdefault(row.get("nodeid"), []).append(row.get("outcome"))
    for node in required_nodeids():
        if node not in collected:
            reasons.append(f"missing nodeid {node}")
            continue
        if calls.get(node) != ["passed"] or setups.get(node) != ["passed"] or teardowns.get(node) != ["passed"]:
            reasons.append(f"missing nodeid {node}")


def _hash_rows(payload: dict) -> dict[str, str]:
    return {row["path"]: row["sha256"] for row in payload.get("files") or []}


def evaluate_evidence(run: Path, check_workspace: bool = False) -> dict:
    """Recompute V2 evidence. Does not read report.fixes_passed."""
    run = Path(run)
    reasons: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (run / name).is_file()]
    if missing:
        reasons.append("missing " + missing[0])
    if not (run / "cases").is_dir():
        reasons.append("missing cases")
    manifest_ids = []
    if (run / "case_manifest.json").is_file():
        manifest_ids = list(_load(run / "case_manifest.json").get("cases") or [])
    expected = required_case_ids()
    if manifest_ids != expected:
        reasons.append("manifest does not match fixed spec")
    present = []
    metrics = {}
    for case_id in expected:
        path = _case_path(run, case_id) if (run / "cases").is_dir() else None
        if path is None:
            reasons.append(f"missing case {case_id}")
            continue
        present.append(case_id)
        try:
            if case_id == "F4-speed":
                _check_speed_case(path, reasons)
            elif path.suffix == ".json":
                _check_unit_json(case_id, _load(path), reasons)
            else:
                metrics[case_id] = _check_physical(case_id, path, reasons)
        except Exception as exc:
            reasons.append(f"{case_id} bad field {exc.__class__.__name__}")
    extra = []
    if (run / "cases").is_dir():
        extra = [path.stem for path in (run / "cases").iterdir() if path.stem not in expected]
    if extra:
        reasons.append("unknown case " + extra[0])
    run_doc = _load(run / "run.json") if (run / "run.json").is_file() else {}
    if run_doc.get("incomplete") is True:
        reasons.append("run incomplete")
    if (run / "execution.json").is_file():
        try:
            _execution_ok(_load(run / "execution.json"), reasons)
        except Exception as exc:
            reasons.append(f"execution bad field {exc.__class__.__name__}")
    before = after = None
    if (run / "input_hash_before.json").is_file() and (run / "input_hash_after.json").is_file():
        before = _load(run / "input_hash_before.json")
        after = _load(run / "input_hash_after.json")
        if _hash_rows(before) != _hash_rows(after) or before.get("aggregate_sha256") != after.get("aggregate_sha256"):
            reasons.append("input hash changed")
        recorded = _hash_rows(after)
        if not recorded:
            reasons.append("input set differs")
        for prefix in INPUT_PREFIXES:
            if not any(path == prefix or path.startswith(prefix) for path in recorded):
                reasons.append("input set differs")
                break
        if check_workspace:
            try:
                current = _hash_rows(collect_inputs())
            except RuntimeError as exc:
                reasons.append(str(exc))
                current = {}
            if current != recorded:
                reasons.append("input set differs")
    binding = _load(run / "m1_regression_binding.json") if (run / "m1_regression_binding.json").is_file() else {}
    regression = Path(binding.get("regression_dir", ""))
    if not regression.is_dir():
        reasons.append("m1 regression failed")
    else:
        verdict = assess_run(regression)
        if not (verdict.get("physics_passed") and verdict.get("provenance_passed") and verdict.get("visual_status") == "passed" and verdict.get("m2_ready")):
            reasons.append("m1 regression failed")
        else:
            shared_name = regression / "input_hash_after.json"
            if shared_name.is_file() and after is not None:
                shared = _hash_rows(_load(shared_name))
                recorded = _hash_rows(after)
                for path, digest in shared.items():
                    if path in recorded and recorded[path] != digest:
                        reasons.append(f"shared input differs {path}")
                        break
    baseline = Path(binding.get("baseline_dir", ""))
    if baseline.is_dir():
        base = assess_run(baseline)
        if not base.get("m2_ready"):
            reasons.append("m1 baseline failed")
    else:
        reasons.append("m1 baseline failed")
    comparisons = _abc(metrics, reasons)
    if (run / "controller_config.json").is_file():
        cfg = _load(run / "controller_config.json")
        if cfg.get("hybrid_normal_force", {}).get("enabled") is not False:
            reasons.append("hybrid status is not disabled")
        if float(cfg.get("tau_rate_nm_s", 0)) != TAU_RATE_NM_S:
            reasons.append("torque rate threshold mismatch")
    structural = [item for item in reasons if _structural(item)]
    return {
        "evidence_valid": not structural,
        "fixes_passed": not reasons,
        "m3_ready": False,
        "hybrid_force_status": "disabled",
        "remaining_m2_requirements": list(REMAINING_FULL_M2),
        "reasons": reasons,
        "comparisons": comparisons,
        "n_cases": len(present),
    }


def _structural(item: str) -> bool:
    markers = ("missing", "unknown", "incomplete", "hash", "input set", "shared input", "manifest", "bad field", "m1 ", "exitstatus", "nodeid", "teardown", "skipped", "unreadable", "schema", "scope mismatch")
    return any(marker in item for marker in markers)


def verify_finished_report(run: Path, scope: str, check_workspace: bool) -> dict:
    if scope == "fixes-v1":
        return {"fixes_passed": False, "m3_ready": False, "evidence_valid": False, "hybrid_force_status": "disabled", "reasons": ["fixes-v1 is not a V2 certification"], "scope": scope, "remaining_m2_requirements": list(REMAINING_FULL_M2)}
    if scope not in {"fixes-v2", "full"}:
        return {"fixes_passed": False, "m3_ready": False, "evidence_valid": False, "hybrid_force_status": "disabled", "reasons": [f"unknown scope {scope}"], "scope": scope, "remaining_m2_requirements": list(REMAINING_FULL_M2)}
    verdict = evaluate_evidence(run, check_workspace=check_workspace)
    reasons = list(verdict["reasons"])
    report = {}
    report_path = Path(run) / "report.json"
    if report_path.is_file():
        try:
            report = _load(report_path)
        except json.JSONDecodeError:
            reasons.append("unreadable report.json")
    if report.get("schema_version") != SCHEMA_VERSION:
        reasons.append("unsupported schema " + str(report.get("schema_version")))
    if report.get("scope") != SCOPE_NAME:
        reasons.append("scope mismatch")
    if report.get("hybrid_force_status") != "disabled":
        reasons.append("hybrid status is not disabled")
    if report.get("m3_ready") is True:
        reasons.append("summary claims m3_ready")
    computed_fixes = not verdict["reasons"]
    computed_evidence = bool(verdict["evidence_valid"]) and report.get("schema_version") == SCHEMA_VERSION and report.get("scope") == SCOPE_NAME
    if report.get("evidence_valid") is not computed_evidence or report.get("fixes_passed") is not computed_fixes:
        reasons.append("summary disagrees")
    if scope == "full":
        reasons.append("full M2 still missing: " + REMAINING_FULL_M2[0])
    return {
        "fixes_passed": scope == "fixes-v2" and not reasons,
        "m3_ready": False,
        "evidence_valid": computed_evidence and "summary disagrees" not in reasons and not any(_structural(item) for item in reasons),
        "hybrid_force_status": "disabled",
        "remaining_m2_requirements": list(REMAINING_FULL_M2),
        "reasons": reasons,
        "scope": scope,
        "comparisons": verdict.get("comparisons", {}),
    }
