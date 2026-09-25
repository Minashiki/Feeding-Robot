"""T09 and T12–T14: head motion, nominal run, timestep and solver checks."""

import numpy as np

from feedingrobot.sim.cases import save_case
from feedingrobot.sim.contacts import pair_set
from feedingrobot.sim.model import repo_root
from feedingrobot.sim.scene import FeedingScene


def test_t09_head_tracks_for_10s(m1_config):
    scene = FeedingScene(m1_config)
    scene.reset(seed=3, preset="food_on_plate")
    prev = None
    saturated = 0
    peak_slide = 0.0
    for _ in range(10000):
        state = scene.step_physics(scene.bias_tau(), hold_driver=False)
        assert state["finite"]
        assert sum(state["warnings"]) == 0
        q = np.concatenate([state["head_q"], [state["jaw_q"]]])
        if prev is not None:
            assert np.max(np.abs(q - prev)) < 0.02
        ranges_h = scene.model.jnt_range[scene.index.head_joint_ids]
        jaw_range = scene.model.jnt_range[scene.index.jaw_joint_id]
        assert np.all(state["head_q"] >= ranges_h[:, 0]) and np.all(state["head_q"] <= ranges_h[:, 1])
        assert jaw_range[0] <= state["jaw_q"] <= jaw_range[1]
        assert np.max(np.abs(state["head_error"][:3])) < 0.004
        prev = q
        peak_slide = max(peak_slide, float(np.max(np.abs(state["head_q"][:2]))))
        limits = scene.model.actuator_ctrlrange[scene.index.scene_actuator_ids]
        cmd = state["head_tau"]
        saturated += int(np.any(np.abs(cmd - limits[:, 0]) < 1e-6) or np.any(np.abs(cmd - limits[:, 1]) < 1e-6))
    assert saturated < 200
    t = 10.0
    w = 2 * np.pi * scene.config["head"]["freq_hz"]
    amp = scene.config["head"]["amp_m"]
    # position has moved; final sine is not required to be exact, motion must be visible
    assert peak_slide > 0.5 * amp
    assert abs(state["head_q"][0] - amp * np.sin(w * t)) < 0.003
    ranges = scene.model.jnt_range[scene.index.head_joint_ids]
    assert np.all(state["head_q"] >= ranges[:, 0]) and np.all(state["head_q"] <= ranges[:, 1])
    save_case("T09", {"final_head_error_m": float(np.max(np.abs(state["head_error"][:3]))), "saturated_steps": int(saturated)})


def test_t12_nominal_10s(m1_config):
    scene = FeedingScene(m1_config)
    scene.reset(seed=4, preset="food_on_plate")
    for _ in range(10000):
        state = scene.step_physics(scene.bias_tau(), hold_driver=False)
        assert state["finite"] and sum(state["warnings"]) == 0
        assert min((c["dist"] for c in state["contacts"]), default=0.0) > -1e-3
    ranges = scene.model.jnt_range[scene.index.arm_joint_ids]
    assert np.all(state["q"] > ranges[:, 0] - 1e-4) and np.all(state["q"] < ranges[:, 1] + 1e-4)
    save_case(
        "T12",
        {"final_min_dist_m": float(min(c["dist"] for c in state["contacts"]))},
        thresholds={"min_contact_dist_m": -0.001},
        effective_config={"preset": "food_on_plate", "seed": 4},
        duration_s=10.0,
    )


THRESHOLDS = {
    "event_s": 0.02,
    "steady_force_n": {"floor": 0.001, "frac": 0.1},
    "peak_force_n": {"floor": 0.02, "frac": 0.1},
    "vector_impulse_ns": {"floor": 1e-4, "frac": 0.1},
    "scalar_impulse_ns": {"floor": 1e-4, "frac": 0.1},
    "steady_food_pos_m": 0.002,
    "min_contact_dist_m": -0.001,
}


def _close(a, b, floor, frac):
    return abs(a - b) <= max(floor, frac * max(abs(a), abs(b)))


def _comparison(kind, seed, pair, ref, cmp_, steady_ok, peak_ok, vec_ok, sca_ok):
    event_a = None if ref["event_s"] is None else float(ref["event_s"])
    event_b = None if cmp_["event_s"] is None else float(cmp_["event_s"])
    return {
        "kind": kind,
        "seed": int(seed),
        "pair": pair,
        "steady_ok": bool(steady_ok),
        "peak_ok": bool(peak_ok),
        "vector_impulse_ok": bool(vec_ok),
        "scalar_impulse_ok": bool(sca_ok),
        "event_a": event_a,
        "event_b": event_b,
        "delta_event_s": None if event_a is None or event_b is None else abs(event_a - event_b),
        "peak_a": float(ref["peak_force_n"]),
        "peak_b": float(cmp_["peak_force_n"]),
        "steady_a": ref["steady_force_n"],
        "steady_b": cmp_["steady_force_n"],
        "vector_a": ref["vector_impulse_ns"],
        "vector_b": cmp_["vector_impulse_ns"],
        "scalar_a": float(ref["scalar_impulse_ns"]),
        "scalar_b": float(cmp_["scalar_impulse_ns"]),
        "thresholds": THRESHOLDS,
        "conclusion": "passed" if steady_ok and peak_ok and vec_ok and sca_ok else "failed",
    }


def test_t13_t14_contact_matrix(m1_config):
    import json
    import os
    from pathlib import Path

    import mujoco

    from feedingrobot.sim.cases import save_dynamic
    from feedingrobot.sim.matrix import capture_state, run_case, state_digest

    solvers = {
        "A": {"timestep": 0.001, "iterations": 50, "tolerance_scale": 1.0},
        "B": {"timestep": 0.0005, "iterations": 50, "tolerance_scale": 1.0},
        "C": {"timestep": 0.001, "iterations": 100, "tolerance_scale": 0.1},
        "D": {"timestep": 0.00025, "iterations": 50, "tolerance_scale": 1.0},
    }
    scene = FeedingScene(m1_config)
    base_tol = float(scene.model.opt.tolerance)
    matrix = m1_config["matrix"]
    seeds = list(matrix["seeds"])
    kinds = {
        "plate": float(matrix["plate_duration_s"]),
        "spoon": float(matrix["spoon_duration_s"]),
        "tilt": float(matrix["tilt_duration_s"]),
        "accel": float(matrix["accel_duration_s"]),
        "lip": float(matrix["lip_duration_s"]),
    }
    comparisons = []
    runs = {}
    for seed in seeds:
        for kind, duration in kinds.items():
            preset = "food_on_spoon" if kind in ("spoon", "tilt", "accel") else "food_on_plate" if kind == "plate" else "near_mouth"
            scene.reset(seed=seed, preset=preset, settle_steps=None if kind in ("tilt", "accel") else 0)
            if kind == "lip":
                scene.data.qpos[scene.index.arm_qpos_adr] = np.array(m1_config["q_mouth_out"])
                adr = scene.index.food_qpos_adr
                scene.data.qpos[adr : adr + 7] = [0.8, 0.35, -0.012, 1, 0, 0, 0]
                scene.data.qvel[:] = 0
                mujoco.mj_forward(scene.model, scene.data)
            snap = capture_state(scene)
            digest = state_digest(scene.data)
            names = ("A", "B", "C", "D") if kind == "tilt" and seed == 4 else ("A", "B", "C")
            row = {}
            for name in names:
                spec = solvers[name]
                if kind == "accel":
                    scene.apply_experiment(
                        timestep=spec["timestep"],
                        iterations=spec["iterations"],
                        tolerance=base_tol * spec["tolerance_scale"],
                        speed_limit=float(m1_config["diagnostic_accel"]["speed_limit_rad_s"]),
                    )
                else:
                    scene.apply_experiment(
                        timestep=spec["timestep"],
                        iterations=spec["iterations"],
                        tolerance=base_tol * spec["tolerance_scale"],
                    )
                options = scene.effective_options()
                assert options["timestep"] == spec["timestep"]
                assert int(scene.model.opt.iterations) == spec["iterations"]
                result = run_case(scene, snap, kind, duration)
                case_id = f"{kind}_seed{seed}_{name}"
                status = "passed" if result["metrics"]["ok"] else "failed"
                save_dynamic(
                    case_id,
                    result["metrics"],
                    result["arrays"],
                    seed=seed,
                    solver=name,
                    effective_config=options,
                    thresholds=THRESHOLDS,
                    duration_s=duration,
                    initial_hash=digest,
                    status=status,
                    failure_reason=result["metrics"]["failure_reason"],
                    first_failure_tick=result["metrics"]["first_failure_tick"],
                )
                row[name] = result["metrics"]
                assert result["metrics"]["ok"], (case_id, result["metrics"]["failure_reason"])
            runs[(kind, seed)] = row
    for (kind, seed), row in runs.items():
        for other in ("B", "C"):
            ref, cmp_ = row["A"], row[other]
            if kind in ("tilt", "accel", "lip"):
                assert ref["event_s"] is not None and cmp_["event_s"] is not None, (kind, seed, ref["event_s"], cmp_["event_s"])
                assert abs(ref["event_s"] - cmp_["event_s"]) <= 0.02, (kind, seed, other, ref["event_s"], cmp_["event_s"])
            if kind in ("plate", "spoon"):
                assert np.linalg.norm(np.array(ref["steady_food_pos_m"]) - np.array(cmp_["steady_food_pos_m"])) <= 0.002
            steady_ok = _close(np.linalg.norm(ref["steady_force_n"]), np.linalg.norm(cmp_["steady_force_n"]), 0.001, 0.1)
            peak_ok = _close(ref["peak_force_n"], cmp_["peak_force_n"], 0.02, 0.1)
            vec_ok = _close(np.linalg.norm(ref["vector_impulse_ns"]), np.linalg.norm(cmp_["vector_impulse_ns"]), 1e-4, 0.1)
            sca_ok = _close(ref["scalar_impulse_ns"], cmp_["scalar_impulse_ns"], 1e-4, 0.1)
            comparisons.append(_comparison(kind, seed, f"A/{other}", ref, cmp_, steady_ok, peak_ok, vec_ok, sca_ok))
            assert steady_ok and peak_ok and vec_ok and sca_ok, comparisons[-1]
        if kind == "tilt" and seed == 4:
            for left, right in (("B", "D"), ("A", "D")):
                ref, cmp_ = row[left], row[right]
                assert abs(ref["event_s"] - cmp_["event_s"]) <= 0.02, (left, right, ref["event_s"], cmp_["event_s"])
                steady_ok = _close(np.linalg.norm(ref["steady_force_n"]), np.linalg.norm(cmp_["steady_force_n"]), 0.001, 0.1)
                peak_ok = _close(ref["peak_force_n"], cmp_["peak_force_n"], 0.02, 0.1)
                vec_ok = _close(np.linalg.norm(ref["vector_impulse_ns"]), np.linalg.norm(cmp_["vector_impulse_ns"]), 1e-4, 0.1)
                sca_ok = _close(ref["scalar_impulse_ns"], cmp_["scalar_impulse_ns"], 1e-4, 0.1)
                comparisons.append(_comparison(kind, seed, f"{left}/{right}", ref, cmp_, steady_ok, peak_ok, vec_ok, sca_ok))
                assert steady_ok and peak_ok and vec_ok and sca_ok, comparisons[-1]
    run_dir = os.environ.get("M1_RUN_DIR")
    if run_dir:
        Path(run_dir, "comparisons.json").write_text(json.dumps(comparisons, indent=2))
    save_case(
        "T13_T14",
        {"n_formal_runs": 6 * 5 * 3, "n_seed4_d": 1},
        seeds=seeds,
        thresholds=THRESHOLDS,
        effective_config={"seeds": seeds, "durations_s": kinds},
        duration_s=max(kinds.values()),
        log={"n_formal_runs": np.array([90])},
    )


