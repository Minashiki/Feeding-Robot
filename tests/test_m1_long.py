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
    save_case("T12", {"final_min_dist": float(min(c["dist"] for c in state["contacts"]))})


def _pair_force(contacts, group, partner):
    total = np.zeros(3)
    peak = 0.0
    for row in contacts:
        if {row["group1"], row["group2"]} != {group, partner}:
            continue
        force = row["force_on_geom1_world"] if row["group1"] == group else row["force_on_geom2_world"]
        total = total + force
        peak = max(peak, float(np.linalg.norm(force)))
    return total, peak


def test_t13_t14_contact_matrix(m1_config):
    from feedingrobot.sim.cases import save_case
    from feedingrobot.sim.trajectory import quintic, smooth_pulse
    from feedingrobot.sim.trial import Trial

    configs = {
        "A": {"timestep": 0.001, "iterations": 50, "tolerance": None},
        "B": {"timestep": 0.0005, "iterations": 50, "tolerance": None},
        "C": {"timestep": 0.001, "iterations": 100, "tolerance": 0.1},
    }
    scene = FeedingScene(m1_config)
    base_tol = float(scene.model.opt.tolerance)
    results = []

    def run(seed, kind):
        trial = Trial(scene.speed_limit)
        forces = []
        times = []
        food = []
        event = None
        off = 0
        q0 = scene.data.qpos[scene.index.arm_qpos_adr].copy()
        for k in range(int(round(run.duration / scene.model.opt.timestep))):
            t = k * scene.model.opt.timestep
            if kind == "tilt":
                q_ref, dq_ref = quintic(q0, np.array(m1_config["q_tilt"]), t, 8.0)
                tau = scene.diagnostic_pd_tau(q_ref, 20, 8, dq_ref)
            elif kind == "accel":
                tau = scene.bias_tau()
                tau[5] += smooth_pulse(t, 0.3, 0.4, 8.0)
            elif kind == "lip":
                q_ref, dq_ref = quintic(np.array(m1_config["q_mouth_out"]), np.array(m1_config["q_mouth_lip"]), t, 8.0)
                tau = scene.diagnostic_pd_tau(q_ref, 30, 10, dq_ref)
            else:
                tau = scene.bias_tau()
            state = scene.step_physics(tau, hold_driver=True)
            trial.observe(state)
            dt_now = float(scene.model.opt.timestep)
            if kind in ("plate", "spoon"):
                partner = "plate" if kind == "plate" else "spoon"
                forces.append(_pair_force(state["contacts"], "food", partner)[0])
            elif kind in ("tilt", "accel"):
                forces.append(_pair_force(state["contacts"], "food", "spoon")[0])
                local = state["tcp_mat"].T @ (state["food_pos"] - state["tcp_pos"])
                outside = np.linalg.norm(local[:2]) > 0.02 or local[2] > 0.012 or local[2] < -0.004
                touching = any({c["group1"], c["group2"]} == {"food", "spoon"} for c in state["contacts"])
                off = off + dt_now if outside and not touching else 0.0
                if off >= 0.1 and event is None:
                    event = t
            else:
                forces.append(_pair_force(state["contacts"], "spoon", "mouth")[0])
                if event is None and any("mouth_upper" in (c["geom1"], c["geom2"]) for c in state["contacts"]):
                    event = t
            times.append(t)
            food.append(state["food_pos"].copy())
        forces = np.asarray(forces)
        dt = float(scene.model.opt.timestep)
        window = max(1, int(round(0.2 / dt)))
        steady = forces[-window:]
        impulse = np.sum(forces, axis=0) * dt
        return {
            "ok": trial.ok,
            "failure": None if trial.ok else trial.failure["reason"],
            "steady": np.mean(steady, axis=0),
            "times": np.asarray(times),
            "forces": forces,
            "peak": float(np.max(np.linalg.norm(forces, axis=1))),
            "impulse": float(np.linalg.norm(impulse)),
            "event": event,
            "food": np.mean(np.asarray(food)[-window:], axis=0),
            "options": scene.effective_options(),
            "supported": _supported(kind, state),
        }

    def _supported(kind, state):
        if kind == "plate":
            return any({c["group1"], c["group2"]} == {"food", "plate"} for c in state["contacts"])
        if kind == "spoon":
            return any({c["group1"], c["group2"]} == {"food", "spoon"} for c in state["contacts"])
        return True

    durations = {"plate": 1.0, "spoon": 1.0, "accel": 0.85, "lip": 5.5}
    # Shared by A/B/C. Each seed releases at a different time; the window ends after 100 ms off the spoon and before the plate impact.
    tilt_seeds = (0, 1, 2, 3, 5)
    tilt_end = {0: 3.392, 1: 3.47, 2: 3.406, 3: 3.413, 5: 3.453}
    for seed in tilt_seeds:
        for kind in ("plate", "spoon", "tilt", "accel", "lip"):
            run.duration = tilt_end[seed] if kind == "tilt" else durations[kind]
            row = {"seed": seed, "kind": kind}
            if kind == "accel":
                scene.apply_experiment(speed_limit=6.0)
            for name, cfg in configs.items():
                tol = None if cfg["tolerance"] is None else base_tol * cfg["tolerance"]
                settled = kind in ("tilt", "accel")
                scene.reset(
                    seed=seed,
                    preset="food_on_spoon" if kind in ("spoon", "tilt", "accel") else "food_on_plate" if kind == "plate" else "near_mouth",
                    settle_steps=None if settled else 0,
                )
                if kind == "lip":
                    scene.data.qpos[scene.index.arm_qpos_adr] = np.array(m1_config["q_mouth_out"])
                    adr = scene.index.food_qpos_adr
                    scene.data.qpos[adr : adr + 7] = [0.8, 0.35, -0.012, 1, 0, 0, 0]
                    scene.data.qvel[:] = 0
                if kind == "accel":
                    scene.apply_experiment(timestep=cfg["timestep"], iterations=cfg["iterations"], tolerance=tol, speed_limit=6.0)
                else:
                    scene.apply_experiment(timestep=cfg["timestep"], iterations=cfg["iterations"], tolerance=tol)
                assert scene.effective_options()["timestep"] == cfg["timestep"]
                row[name] = run(seed, kind)
                assert row[name]["ok"], (kind, name, seed, row[name]["failure"])
            results.append(row)

    def close(a, b, floor, frac):
        return abs(a - b) <= max(floor, frac * max(abs(a), abs(b)))

    for row in results:
        for other in ("B", "C"):
            ref, cmp_ = row["A"], row[other]
            if row["kind"] in ("plate", "spoon"):
                assert ref["supported"] and cmp_["supported"]
                assert np.linalg.norm(ref["food"] - cmp_["food"]) <= 0.002
            if row["kind"] in ("tilt", "accel", "lip"):
                assert ref["event"] is not None and cmp_["event"] is not None, (row["kind"], row["seed"], ref["event"], cmp_["event"])
                assert abs(ref["event"] - cmp_["event"]) <= 0.02, (row["kind"], row["seed"], ref["event"], cmp_["event"])
            if row["kind"] in ("tilt", "accel"):
                t_anchor = min(ref["event"], cmp_["event"])
                def _before(run):
                    sel = (run["times"] >= t_anchor - 0.3) & (run["times"] < t_anchor - 0.1)
                    assert np.any(sel)
                    return np.mean(run["forces"][sel], axis=0)
                ref_steady, cmp_steady = _before(ref), _before(cmp_)
            else:
                ref_steady, cmp_steady = ref["steady"], cmp_["steady"]
            assert close(np.linalg.norm(ref_steady), np.linalg.norm(cmp_steady), 0.001, 0.1), (
                row["kind"], row["seed"], other, np.linalg.norm(ref_steady), np.linalg.norm(cmp_steady)
            )
            assert close(ref["peak"], cmp_["peak"], 0.02, 0.1), (row["kind"], row["seed"], other, ref["peak"], cmp_["peak"])
            assert close(ref["impulse"], cmp_["impulse"], 1e-4, 0.1), (row["kind"], row["seed"], other, ref["impulse"], cmp_["impulse"])
    save_case(
        "T13_T14",
        {"n_rows": len(results)},
        seeds=list(tilt_seeds),
        log={"n_rows": np.array([len(results)])},
    )
