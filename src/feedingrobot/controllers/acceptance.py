"""Frozen M2 V2 acceptance spec. Thresholds here override anything stored in a trajectory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from feedingrobot.scripts.validate_m1 import _tracked_files
from feedingrobot.sim.model import repo_root
from feedingrobot.sim.report import assess_run

SCHEMA_VERSION = "m2-fix-v6"
SCOPE_NAME = "fixes_v6"
V6_PACKAGE_NODEID = "tests/test_m2_v6.py::test_v6_package_tampers_are_rejected"
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
    "tests/test_m2_report.py::test_v3_recorded_contact_peaks_fail",
)

REMAINING_FULL_M2 = (
    "T04 full three-pose axis steps, shaping-complete, 1.5 s settle and overshoot",
    "T05 three-pose continuous paths, two loops, orientation sine, start-stop",
    "T06 six unloaded poses and the full wrench matrix",
    "T07 exact_model and estimated_kinematics dynamic matrix",
    "T08 spring surface",
    "T09 stiffness trend",
    "T10 blocked release on a physical wall",
    "T11 force pulse recovery",
    "T12 T13 T14 remaining contact, fault, and replay cells beyond the V6 subpaths",
    "T15 noise and delay on T05/T08",
    "T16 P0 bridge",
    "T17 full-matrix tamper cases beyond fixes-v6",
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


def v6_zero_grid_ids() -> list[str]:
    return [
        f"V6-zero-j{joint}-{side}-{motion}-{variant}"
        for joint in range(7)
        for side in ("upper", "lower")
        for motion in ("linear", "angular", "mixed")
        for variant in ("A", "B")
    ]


def v6_trace_ids() -> list[str]:
    rows = [f"V6-zero-phys-{side}-{variant}" for side in ("upper", "lower") for variant in ("A", "B", "C")]
    rows.extend(f"V6-stop-{name}-{variant}" for name in ("rest", "cross") for variant in ("A", "B", "C"))
    return rows


def physical_case_ids() -> list[str]:
    families = ("pause", "release", "release_fast", "gear", "wrench", "takeover")
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
        *v6_zero_grid_ids(),
        *v6_trace_ids(),
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
    for name in (
        "test_gain_free_to_acquire_first_and_final_step",
        "test_gain_return_after_50ms",
        "test_gain_retarget_restarts_transition",
        "test_same_gear_phase_does_not_restart",
        "test_fake_free_gears_are_rejected",
        "test_release_tail_rejects_reload_and_keeps_boundaries",
    ):
        nodes.append(f"tests/test_m2_v5.py::{name}")
    for dt in ("0.001", "0.0005"):
        nodes.append(f"tests/test_m2_v6.py::test_zero_command_still_projects_outward_history[{dt}]")
        nodes.append(f"tests/test_m2_v6.py::test_singular_region_enters_latched_controlled_stop[{dt}]")
    for name in (
        "test_joint_allow_boundaries",
        "test_zero_far_from_limit_decelerates",
        "test_inward_history_is_not_frozen",
        "test_multi_axis_uses_strictest_beta",
        "test_stop_power_on_and_prohibit_do_not_integrate",
        "test_zero_history_does_not_rewind",
        "test_rho_threshold_edges",
        "test_first_fault_is_kept",
        "test_power_on_blend_yields_to_singularity",
        "test_rho_recovery_stays_latched",
        "test_reset_clears_or_relatches_singularity",
        "test_slow_release_scores_from_unload",
        "test_release_boundaries_and_fast_stable_window",
        "test_adversarial_io_error_is_reported",
    ):
        nodes.append(f"tests/test_m2_v6.py::{name}")
    for case_id in v6_zero_grid_ids():
        nodes.append(f"tests/test_m2_v6.py::test_zero_projection_grid[{case_id}]")
    for case_id in v6_trace_ids():
        nodes.append(f"tests/test_m2_v6.py::test_v6_trace[{case_id}]")
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



def evaluate_evidence(run, check_workspace: bool = False) -> dict:
    from feedingrobot.controllers.v3spec import evaluate_evidence as _evaluate

    return _evaluate(run, check_workspace=check_workspace)


def verify_finished_report(run, scope: str, check_workspace: bool) -> dict:
    from feedingrobot.controllers.v3spec import verify_finished_report as _verify

    return _verify(run, scope, check_workspace)
