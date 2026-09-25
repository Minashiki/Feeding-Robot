"""The report gate has to reject incomplete or empty evidence."""

import json
from pathlib import Path

import numpy as np

from feedingrobot.sim.report import KINDS, PAIR_THRESHOLDS, PROCESS_IDS, SEEDS, SOLVER_SPEC, STATIC_IDS, assess_run


def _trace(duration, n=5, peak=0.01, dist=-1e-4, kind="plate"):
    time_s = np.linspace(duration / n, duration, n)
    force = np.zeros((n, 3))
    force[-1, 2] = peak
    food = np.zeros((n, 3))
    event_confirmed = np.zeros(n, dtype=np.int8)
    mouth_contact = np.zeros(n, dtype=np.int8)
    contact_flat = np.zeros((0, 10))
    geom_ids = np.zeros(0, dtype=np.int32)
    geom_names = np.zeros(0, dtype="S64")
    if kind in ("tilt", "accel"):
        food[:, 0] = 1.0
        event_confirmed[:] = 1
    if kind == "lip":
        mouth_contact[2] = 1
        geom_ids = np.asarray([1, 2], dtype=np.int32)
        geom_names = np.asarray([b"mouth_upper", b"bowl_bottom"])
        contact_flat = np.zeros((1, 10))
        contact_flat[0, 0] = 2
        contact_flat[0, 1] = 1
        contact_flat[0, 2] = 2
    return {
        "time_s": time_s,
        "input_interval_s": np.column_stack([time_s - duration / n, time_s]),
        "q": np.zeros((n, 7)),
        "dq": np.full((n, 7), 0.01),
        "qacc": np.zeros((n, 7)),
        "tau_requested": np.zeros((n, 7)),
        "tau_command": np.zeros((n, 7)),
        "tau_actual": np.zeros((n, 7)),
        "tcp_pos": np.zeros((n, 3)),
        "tcp_rot": np.tile(np.eye(3).reshape(9), (n, 1)),
        "mouth_pos": np.zeros((n, 3)),
        "mouth_rot": np.tile(np.eye(3).reshape(9), (n, 1)),
        "food_pos": food,
        "food_quat": np.tile([1, 0, 0, 0], (n, 1)),
        "food_vel": np.zeros((n, 6)),
        "raw_wrench": np.zeros((n, 6)),
        "contact_force_group": force,
        "single_contact_peak": np.linalg.norm(force, axis=1),
        "min_contact_dist": np.full(n, dist),
        "warning_count": np.zeros(n, dtype=np.int32),
        "velocity_fault": np.zeros(n, dtype=np.int8),
        "forbidden_contact": np.zeros(n, dtype=np.int8),
        "event_candidate": event_confirmed,
        "event_confirmed": event_confirmed,
        "mouth_contact": mouth_contact,
        "head_target": np.zeros((n, 4)),
        "head_actual": np.zeros((n, 5)),
        "head_command": np.zeros((n, 5)),
        "contact_flat": contact_flat,
        "geom_ids": geom_ids,
        "geom_name_bytes": geom_names,
    }


def _kind_of(case_id):
    for kind in KINDS:
        if case_id.startswith(kind + "_seed"):
            return kind
    return None


def _metrics_from(arrays, case_id):
    time_s = np.asarray(arrays["time_s"], dtype=float)
    force = np.asarray(arrays["contact_force_group"], dtype=float)
    dt = float(np.median(np.diff(time_s)))
    window = max(1, int(round(0.2 / dt)))
    kind = _kind_of(case_id)
    if kind == "lip":
        event = float(time_s[2])
    elif kind in ("tilt", "accel"):
        event = float(time_s[0])
    else:
        event = None
    return {
        "peak_force_n": float(np.max(np.linalg.norm(force, axis=1))),
        "scalar_impulse_ns": float(np.sum(np.linalg.norm(force, axis=1)) * dt),
        "vector_impulse_ns": (np.sum(force, axis=0) * dt).tolist(),
        "steady_force_n": np.mean(force[-window:], axis=0).tolist(),
        "steady_food_pos_m": np.mean(arrays["food_pos"][-window:], axis=0).tolist(),
        "min_contact_dist_m": float(np.min(arrays["min_contact_dist"])),
        "max_joint_speed_rad_s": float(np.max(np.abs(arrays["dq"]))),
        "n_samples": int(time_s.shape[0]),
        "event_s": event,
    }


def _write_dynamic(case_dir, run_id, case_id, duration=1.0, metrics=None, arrays=None):
    import hashlib
    import os

    kind = _kind_of(case_id)
    arrays = arrays or _trace(duration, kind=kind or "plate")
    npz = case_dir / f"{case_id}.npz"
    np.savez(npz, **arrays)
    digest = hashlib.sha256(npz.read_bytes()).hexdigest()
    metrics = metrics or _metrics_from(arrays, case_id)
    solver = case_id.rsplit("_", 1)[-1] if kind else None
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "case_id": case_id,
        "seed": int(case_id.split("_seed", 1)[1].split("_", 1)[0]) if kind else 0,
        "solver_setting": solver,
        "initial_state_sha256": f"{kind}:{payload_seed(case_id)}" if kind else "process",
        "status": "passed",
        "metrics": metrics,
        "thresholds": PAIR_THRESHOLDS,
        "effective_config": SOLVER_SPEC[solver] if solver in SOLVER_SPEC else {"timestep": 0.001},
        "duration_s": duration,
        "log_paths": [npz.name],
        "log_sha256": digest,
    }
    (case_dir / f"{case_id}.json").write_text(json.dumps(payload))


def payload_seed(case_id):
    if "_seed" not in case_id:
        return 0
    return int(case_id.split("_seed", 1)[1].split("_", 1)[0])


def _comparison(kind, seed, pair):
    metrics = _metrics_from(_trace(1.0, kind=kind), f"{kind}_seed{seed}_A")
    left, right = pair.split("/")
    event = metrics["event_s"]
    needs_event = kind in ("tilt", "accel", "lip")
    return {
        "kind": kind,
        "seed": seed,
        "pair": pair,
        "case_a": f"{kind}_seed{seed}_{left}",
        "case_b": f"{kind}_seed{seed}_{right}",
        "steady_ok": True,
        "peak_ok": True,
        "vector_impulse_ok": True,
        "scalar_impulse_ok": True,
        "event_ok": True if needs_event else "not_applicable",
        "position_ok": "not_applicable" if needs_event else True,
        "event_a": event,
        "event_b": event,
        "delta_event_s": None if event is None else 0.0,
        "position_delta_m": None if needs_event else 0.0,
        "peak_a": metrics["peak_force_n"],
        "peak_b": metrics["peak_force_n"],
        "steady_a": metrics["steady_force_n"],
        "steady_b": metrics["steady_force_n"],
        "vector_a": metrics["vector_impulse_ns"],
        "vector_b": metrics["vector_impulse_ns"],
        "scalar_a": metrics["scalar_impulse_ns"],
        "scalar_b": metrics["scalar_impulse_ns"],
        "thresholds": PAIR_THRESHOLDS,
        "conclusion": "passed",
    }


def _bundle(tmp: Path):
    run_id = "test-run"
    tmp.mkdir()
    (tmp / "run.json").write_text(json.dumps({"schema_version": 2, "run_id": run_id, "status": "completed", "incomplete": False}))
    (tmp / "execution.json").write_text(json.dumps({"run_id": run_id, "reports": [{"nodeid": "tests/test_m1_long.py::test_t13_t14_contact_matrix", "when": "call", "outcome": "passed"}]}))
    (tmp / "input_hash_before.json").write_text(json.dumps({"aggregate_sha256": "abc"}))
    (tmp / "input_hash_after.json").write_text(json.dumps({"aggregate_sha256": "abc"}))
    (tmp / "frames.json").write_text(json.dumps([{"file": "overview_scene.png"}]))
    (tmp / "overview_scene.png").write_bytes(b"\x89PNG" + b"0" * 200)
    case_dir = tmp / "cases"
    case_dir.mkdir()
    for seed in SEEDS:
        for kind in KINDS:
            for solver in ("A", "B", "C"):
                _write_dynamic(case_dir, run_id, f"{kind}_seed{seed}_{solver}")
    _write_dynamic(case_dir, run_id, "tilt_seed4_D")
    for case_id in PROCESS_IDS:
        _write_dynamic(case_dir, run_id, case_id)
    for case_id in STATIC_IDS:
        (case_dir / f"{case_id}.json").write_text(json.dumps({
            "run_id": run_id,
            "case_id": case_id,
            "metrics": {"value": 1.0},
            "thresholds": {"value": 2.0},
            "effective_config": {"config": "configs/m1_scene.json"},
            "duration_s": 1.0,
        }))
    rows = []
    for seed in SEEDS:
        for kind in KINDS:
            for pair in ("A/B", "A/C"):
                rows.append(_comparison(kind, seed, pair))
    rows.append(_comparison("tilt", 4, "A/D"))
    rows.append(_comparison("tilt", 4, "B/D"))
    (tmp / "comparisons.json").write_text(json.dumps(rows))
    return tmp


def test_complete_min_evidence_is_ready(tmp_path):
    result = assess_run(_bundle(tmp_path / "ok"))
    assert result["m2_ready"] is True, result["reasons"]


def test_incomplete_empty_metrics_and_missing_logs_are_not_ready(tmp_path):
    run = _bundle(tmp_path / "bad")
    payload = json.loads((run / "run.json").read_text())
    payload["incomplete"] = True
    payload["status"] = "running"
    (run / "run.json").write_text(json.dumps(payload))
    assert assess_run(run)["m2_ready"] is False
    row = json.loads((run / "cases" / "tilt_seed4_A.json").read_text())
    row["metrics"] = {}
    (run / "cases" / "tilt_seed4_A.json").write_text(json.dumps(row))
    payload["incomplete"] = False
    payload["status"] = "completed"
    (run / "run.json").write_text(json.dumps(payload))
    result = assess_run(run)
    assert result["m2_ready"] is False
    assert any("metrics" in item for item in result["reasons"])
    row["metrics"] = {"peak_force_n": 0.01, "scalar_impulse_ns": 0.01, "min_contact_dist_m": 0.0, "n_samples": 5}
    row["log_paths"] = ["missing.npz"]
    (run / "cases" / "tilt_seed4_A.json").write_text(json.dumps(row))
    assert assess_run(run)["m2_ready"] is False


def test_summary_npz_nan_and_event_gap_are_not_ready(tmp_path):
    run = _bundle(tmp_path / "gap")
    case_dir = run / "cases"
    _write_dynamic(case_dir, "test-run", "tilt_seed4_A", arrays={"n_rows": np.array([25])}, metrics={"peak_force_n": 1, "scalar_impulse_ns": 1, "min_contact_dist_m": 0, "n_samples": 1})
    result = assess_run(run)
    assert result["m2_ready"] is False
    assert any("tilt_seed4_A" in item for item in result["reasons"])
    comparisons = json.loads((run / "comparisons.json").read_text())
    row = next(item for item in comparisons if item["kind"] == "tilt" and item["seed"] == 4 and item["pair"] == "A/B")
    row["event_b"] = row["event_a"] + 0.0405
    row["delta_event_s"] = 0
    (run / "comparisons.json").write_text(json.dumps(comparisons))
    _write_dynamic(case_dir, "test-run", "tilt_seed4_A")
    result = assess_run(run)
    assert result["m2_ready"] is False
    assert any("event" in item for item in result["reasons"])


def test_old_passed_record_does_not_cover_a_skip(tmp_path):
    run = _bundle(tmp_path / "skip")
    execution = json.loads((run / "execution.json").read_text())
    execution["reports"].append({"nodeid": "tests/test_m1_long.py::test_t09_head_tracks_for_10s", "when": "call", "outcome": "skipped"})
    (run / "execution.json").write_text(json.dumps(execution))
    assert assess_run(run)["m2_ready"] is False


def test_bad_arrays_nan_hash_and_matrix_holes_are_not_ready(tmp_path):
    run = _bundle(tmp_path / "holes")
    case_dir = run / "cases"
    backwards = _trace(1.0)
    backwards["time_s"] = backwards["time_s"][::-1].copy()
    _write_dynamic(case_dir, "test-run", "plate_seed0_A", arrays=backwards)
    result = assess_run(run)
    assert result["m2_ready"] is False
    assert any("time" in item for item in result["reasons"])

    nan = _trace(1.0)
    finite = _metrics_from(nan, "plate_seed0_A")
    nan["dq"] = nan["dq"].copy()
    nan["dq"][0, 0] = np.nan
    _write_dynamic(case_dir, "test-run", "plate_seed0_A", arrays=nan, metrics=finite)
    result = assess_run(run)
    assert result["m2_ready"] is False
    assert any("nonfinite" in item for item in result["reasons"])

    deep = _trace(1.0, dist=-0.002)
    _write_dynamic(case_dir, "test-run", "plate_seed0_A", arrays=deep, metrics={**_metrics_from(deep, "plate_seed0_A"), "min_contact_dist_m": -1e-4})
    result = assess_run(run)
    assert result["m2_ready"] is False
    assert any("penetration" in item for item in result["reasons"])

    _write_dynamic(case_dir, "test-run", "plate_seed0_A")
    row = json.loads((case_dir / "plate_seed0_A.json").read_text())
    row["log_sha256"] = "0" * 64
    (case_dir / "plate_seed0_A.json").write_text(json.dumps(row))
    result = assess_run(run)
    assert result["m2_ready"] is False
    assert any("hash" in item for item in result["reasons"])

    _write_dynamic(case_dir, "test-run", "plate_seed0_A")
    (case_dir / "plate_seed0_B.json").unlink()
    result = assess_run(run)
    assert result["m2_ready"] is False
    assert any("plate_seed0_B" in item for item in result["reasons"])

    _write_dynamic(case_dir, "test-run", "plate_seed0_B")
    (case_dir / "tilt_seed4_A.json").unlink()
    result = assess_run(run)
    assert result["m2_ready"] is False
    assert any("tilt_seed4_A" in item for item in result["reasons"])


def test_comparison_numbers_are_recomputed_from_traces(tmp_path):
    run = _bundle(tmp_path / "cmp")
    comparisons = json.loads((run / "comparisons.json").read_text())
    row = next(item for item in comparisons if item["kind"] == "plate" and item["seed"] == 0 and item["pair"] == "A/B")
    row["peak_a"] = 1e6
    row["peak_b"] = 0
    (run / "comparisons.json").write_text(json.dumps(comparisons))
    result = assess_run(run)
    assert result["m2_ready"] is False
    assert any("peak_a" in item and "inconsistent" in item for item in result["reasons"])

    run = _bundle(tmp_path / "impulse")
    comparisons = json.loads((run / "comparisons.json").read_text())
    row = next(item for item in comparisons if item["kind"] == "spoon" and item["seed"] == 1 and item["pair"] == "A/C")
    row["scalar_a"] = 50
    (run / "comparisons.json").write_text(json.dumps(comparisons))
    assert assess_run(run)["m2_ready"] is False

    run = _bundle(tmp_path / "identity")
    comparisons = json.loads((run / "comparisons.json").read_text())
    row = next(item for item in comparisons if item["kind"] == "tilt" and item["seed"] == 4 and item["pair"] == "A/B")
    row["case_a"] = "tilt_seed5_A"
    (run / "comparisons.json").write_text(json.dumps(comparisons))
    result = assess_run(run)
    assert result["m2_ready"] is False
    assert any("identity" in item for item in result["reasons"])

    run = _bundle(tmp_path / "dup")
    comparisons = json.loads((run / "comparisons.json").read_text())
    comparisons.append(dict(comparisons[-1]))
    (run / "comparisons.json").write_text(json.dumps(comparisons))
    result = assess_run(run)
    assert result["m2_ready"] is False
    assert any("duplicate" in item for item in result["reasons"])

    run = _bundle(tmp_path / "hidden")
    arrays = _trace(1.0, kind="plate")
    arrays["geom_ids"] = np.asarray([1, 2], dtype=np.int32)
    arrays["geom_name_bytes"] = np.asarray([b"bowl_bottom", b"plate_bottom"])
    flat = np.zeros((1, 10))
    flat[0, 1] = 1
    flat[0, 2] = 2
    arrays["contact_flat"] = flat
    _write_dynamic(run / "cases", "test-run", "plate_seed0_A", arrays=arrays, metrics=_metrics_from(arrays, "plate_seed0_A"))
    result = assess_run(run)
    assert result["m2_ready"] is False
    assert any("forbidden" in item for item in result["reasons"])
