"""Phase-driven gain checks and persistent release tails."""

import os
from pathlib import Path

import numpy as np

from feedingrobot.controllers.v3spec import (
    ADVERSARIAL,
    COMMAND_ANG_ZERO,
    COMMAND_LIN_ZERO,
    FORCE_ZERO,
    TORQUE_ZERO,
    check_release_tail,
    evaluate_evidence,
    run_adversarial,
)
from feedingrobot.controllers.v4fields import GEAR_K, check_logged_gains, rebuild_gain_path

EXPECTED = {
    "bad_dtype_tick": {"bad_dtype"},
    "bad_offsets": {"bad_offsets"},
    "bad_rotation": {"invalid_rotation"},
    "bad_shape_k": {"bad_shape"},
    "comparisons_lie": {"summary_mismatch"},
    "f6_limit": {"threshold_mismatch", "reference_error"},
    "f6_missing_rot": {"missing_field"},
    "f6_nan": {"nonfinite"},
    "failed_m1": {"m1_failed"},
    "fast_ramp": {"unload_duration"},
    "fast_skip": {"event_mismatch"},
    "fast_speed": {"release_speed"},
    "gain_huge": {"gain_path_mismatch"},
    "gear_fake_free": {"phase_gear_mismatch"},
    "hash_stale": {"hash_mismatch"},
    "input_changed": {"input_hash"},
    "k_nan": {"nonfinite"},
    "manifest_drop": {"evidence_manifest"},
    "manifest_dup": {"duplicate_path"},
    "manifest_empty": {"evidence_manifest"},
    "missing_A": {"missing_comparison"},
    "mode_aborted": {"unexpected_mode"},
    "mode_unknown": {"invalid_enum"},
    "node_missing": {"nodeid"},
    "pause_integral": {"pause_motion"},
    "pause_vref": {"reference_speed"},
    "q_100": {"joint_range"},
    "release_fake_time": {"event_missing"},
    "release_gap": {"bad_time"},
    "release_json": {"wrong_type"},
    "release_reload": {"external_reloaded"},
    "release_speed": {"release_speed"},
    "release_two_frames": {"sample_count", "event_missing"},
    "run_incomplete": {"run_incomplete"},
    "solver_label": {"solver_mismatch"},
    "teardown_fail": {"teardown"},
}


def _codes(reasons):
    return {row["code"] for row in reasons}


def _phase(parts):
    rows = []
    for name, count in parts:
        rows.extend([name] * count)
    return np.array(rows)


def test_gain_free_to_acquire_first_and_final_step():
    for dt, first in ((0.001, 299.25), (0.0005, 299.625)):
        steps = int(round(0.2 / dt))
        phase = _phase((("TRANSPORT", 20), ("ACQUIRE", steps + 5)))
        built = rebuild_gain_path(phase, dt)
        assert abs(built["K"][20, 0] - first) < 1e-9
        done = built["acquire_done"]
        assert done == 20 + steps - 1
        assert np.allclose(built["K"][done], GEAR_K["ACQUIRE"])
        assert built["trans"][done] == 0 and built["active"][done] == "ACQUIRE"
        assert built["trans"][done - 1] == 1
        assert done < len(phase) - 1
        reasons = []
        check_logged_gains(built["K"], built["D"], built["trans"], built["active"], built["target"], phase, dt, "gain", reasons, require_acquire=True)
        assert reasons == []


def test_gain_return_after_50ms():
    for dt, nxt in ((0.001, 250.25), (0.0005, 250.125)):
        held = int(round(0.05 / dt))
        phase = _phase((("TRANSPORT", 5), ("APPROACH", held), ("TRANSPORT", held + 5)))
        built = rebuild_gain_path(phase, dt)
        assert abs(built["K"][5 + held - 1, 0] - 250.0) < 1e-6
        assert abs(built["K"][5 + held, 0] - nxt) < 1e-6
        assert built["target"][5 + held] == "FREE"


def test_gain_retarget_restarts_transition():
    phase = _phase((("TRANSPORT", 5), ("APPROACH", 50), ("ACQUIRE", 20)))
    built = rebuild_gain_path(phase, 0.001)
    assert abs(built["K"][54, 0] - 250.0) < 1e-6
    assert abs(built["K"][55, 0] - 249.5) < 1e-6
    assert built["target"][55] == "ACQUIRE"
    assert built["trans"][54] == 1 and built["trans"][55] == 1


def test_same_gear_phase_does_not_restart():
    phase = _phase((("TRANSPORT", 5), ("WAIT_READY", 5), ("TRANSPORT", 5)))
    built = rebuild_gain_path(phase, 0.001)
    assert np.all(built["trans"] == 0)
    assert np.allclose(built["K"], GEAR_K["FREE"])
    again = rebuild_gain_path(phase, 0.001)
    assert np.allclose(again["K"], built["K"])


def test_fake_free_gears_are_rejected():
    phase = _phase((("TRANSPORT", 10), ("ACQUIRE", 30)))
    built = rebuild_gain_path(phase, 0.001)
    reasons = []
    free_k = np.tile(GEAR_K["FREE"], (len(phase), 1))
    free_d = np.tile(np.array([49.0, 49.0, 49.0, 0.8, 0.8, 0.8]), (len(phase), 1))
    check_logged_gains(free_k, free_d, np.zeros(len(phase), dtype=int), np.array(["FREE"] * len(phase)), np.array(["FREE"] * len(phase)), phase, 0.001, "gear-s0-A", reasons, require_acquire=True)
    assert "phase_gear_mismatch" in _codes(reasons)
    reasons = []
    target = built["target"].copy()
    target[10] = "FREE"
    check_logged_gains(built["K"], built["D"], built["trans"], built["active"], target, phase, 0.001, "gear", reasons)
    assert _codes(reasons) == {"phase_gear_mismatch"}
    assert built["v_lim"][10] == 0.02
    reasons = []
    early = built["K"].copy()
    early[11] = GEAR_K["ACQUIRE"]
    check_logged_gains(early, built["D"], built["trans"], built["active"], built["target"], phase, 0.001, "gear", reasons)
    assert "gain_path_mismatch" in _codes(reasons)
    reasons = []
    check_logged_gains(GEAR_K["FREE"][None, :].repeat(20, 0), np.tile(np.array([49.0, 49.0, 49.0, 0.8, 0.8, 0.8]), (20, 1)), np.zeros(20, dtype=int), np.array(["FREE"] * 20), np.array(["FREE"] * 20), np.array(["TRANSPORT"] * 20), 0.001, "gear", reasons, require_acquire=True)
    assert "transition_incomplete" in _codes(reasons)


def _quiet(n=800):
    command = np.zeros((n, 6))
    external = np.zeros((n, 6))
    times = np.arange(n + 1) * 0.001
    command[:20, 0] = 0.05
    external[10:30, 0] = -3.0
    return command, external, times


def test_release_tail_rejects_reload_and_keeps_boundaries():
    command, external, times = _quiet()
    reasons = []
    check_release_tail(reasons, "release_fast-s0-A", command, external, times, 30, 30)
    assert reasons == []
    external[500:600, 0] = -3.0
    reasons = []
    check_release_tail(reasons, "release_fast-s0-A", command, external, times, 30, 30)
    assert "external_reloaded" in _codes(reasons)
    assert any(row.get("field") == "external_wrench[500]" for row in reasons)
    external[500:600, 0] = 0.0
    external[400, 0] = -3.0
    reasons = []
    check_release_tail(reasons, "case", command, external, times, 30, 30)
    assert "external_reloaded" in _codes(reasons)
    external[400, 0] = 0.0
    external[-1, 3] = 0.01
    reasons = []
    check_release_tail(reasons, "case", command, external, times, 30, 30)
    assert "external_torque_reloaded" in _codes(reasons)
    external[-1, 3] = 0.0
    external[-1, 1] = -3.0
    reasons = []
    check_release_tail(reasons, "case", command, external, times, 30, 30)
    assert "external_reloaded" in _codes(reasons)
    external[-1, 1] = 0.0
    command[700, 5] = 0.2
    reasons = []
    check_release_tail(reasons, "case", command, external, times, 30, 30)
    assert "angular_command_resumed" in _codes(reasons)
    command[700, 5] = 0.0
    command[700, 0] = 0.05
    reasons = []
    check_release_tail(reasons, "case", command, external, times, 30, 30)
    assert "command_resumed" in _codes(reasons)
    command[700, 0] = 0.0
    external[450, 0] = FORCE_ZERO
    command[450, 0] = COMMAND_LIN_ZERO
    external[450, 3] = TORQUE_ZERO
    command[450, 3] = COMMAND_ANG_ZERO
    reasons = []
    check_release_tail(reasons, "case", command, external, times, 30, 30)
    assert reasons == []
    external[450, 0] = FORCE_ZERO + 1e-8
    reasons = []
    check_release_tail(reasons, "case", command, external, times, 30, 30)
    assert "external_reloaded" in _codes(reasons)
    ramp = np.zeros_like(external)
    ramp[30:80, 0] = np.linspace(-3.0, -1e-3, 50)
    reasons = []
    check_release_tail(reasons, "release-s0-A", command, ramp, times, 30, 80)
    assert reasons == []
    ramp[120, 0] = -3.0
    reasons = []
    check_release_tail(reasons, "release-s0-A", command, ramp, times, 30, 80)
    assert "external_reloaded" in _codes(reasons)


if os.environ.get("M2_V5_PACKAGE"):

    def test_v5_package_tampers_are_rejected():
        package = Path(os.environ["M2_V5_PACKAGE"])
        before = evaluate_evidence(package, check_workspace=True)
        assert before["fixes_passed"] is True
        rows = run_adversarial(package)
        assert [row["name"] for row in rows] == list(ADVERSARIAL)
        for row in rows:
            assert row["fixes_passed"] is False
            assert EXPECTED[row["name"]] <= set(row["codes"])
            assert "clean_copy" not in row["codes"]
        again = evaluate_evidence(package, check_workspace=True)
        assert again["fixes_passed"] is True
        assert again["reasons"] == []
        path = os.environ.get("M2_V5_RESULTS")
        if path:
            Path(path).write_text(__import__("json").dumps(rows, indent=2))
