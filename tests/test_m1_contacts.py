"""T07, T08, T10 and the latched head hold."""

import numpy as np

from feedingrobot.sim.cases import save_case
from feedingrobot.sim.mouth import passage, to_mouth
from feedingrobot.sim.scene import FeedingScene
from feedingrobot.sim.trajectory import quintic, smooth_pulse
from feedingrobot.sim.trial import Trial


def _food_on(contacts, group):
    return any({c["group1"], c["group2"]} == {"food", group} for c in contacts)


def _named(contacts, name):
    return any(name in (c["geom1"], c["geom2"]) for c in contacts)


def test_t07_plate_and_spoon_support(m1_config):
    scene = FeedingScene(m1_config)
    speeds = []
    for preset in ("food_on_plate", "food_on_spoon"):
        for seed in range(10):
            state = scene.reset(seed=seed, preset=preset)
            trial = Trial(scene.speed_limit)
            tail = []
            for _ in range(2000):
                state = scene.step_physics(scene.bias_tau(), hold_driver=True)
                trial.observe(state)
                tail.append(np.linalg.norm(state["food_vel"]))
            assert trial.ok, trial.failure
            assert max(tail[-200:]) < 0.01
            if preset == "food_on_plate":
                assert _food_on(state["contacts"], "plate")
                local = to_mouth(state["food_pos"], scene.data.site_xpos[scene.index.site_ids["plate_frame"]], np.eye(3))
                assert np.linalg.norm(local[:2]) < 0.08
            else:
                local = state["tcp_mat"].T @ (state["food_pos"] - state["tcp_pos"])
                assert abs(local[0]) < 0.025 and abs(local[1]) < 0.016
                assert _food_on(state["contacts"], "spoon")
            speeds.append(trial.max_speed)
    save_case("T07", {"max_speed": max(speeds)}, seeds=list(range(10)))


def test_t08_tilt_drop(m1_config):
    scene = FeedingScene(m1_config)
    duration = float(m1_config["tilt"]["duration_s"])
    dropped = []
    max_speed = 0.0
    for seed in range(5):
        scene.reset(seed=seed, preset="food_on_spoon")
        q0 = scene.data.qpos[scene.index.arm_qpos_adr].copy()
        q1 = np.array(m1_config["q_tilt"], dtype=float)
        trial = Trial(scene.speed_limit)
        off = 0
        when = None
        angles = []
        for k in range(int((duration + 1.0) / scene.dt)):
            q_ref, dq_ref = quintic(q0, q1, k * scene.dt, duration)
            state = scene.step_physics(
                scene.diagnostic_pd_tau(q_ref, m1_config["tilt"]["kp"], m1_config["tilt"]["kd"], dq_ref),
                hold_driver=True,
            )
            trial.observe(state)
            touching = _food_on(state["contacts"], "spoon")
            local = state["tcp_mat"].T @ (state["food_pos"] - state["tcp_pos"])
            outside = abs(local[0]) > 0.03 or abs(local[1]) > 0.02 or local[2] > 0.02
            off = off + 1 if (outside and not touching) else 0
            angles.append(float(state["tcp_mat"][2, 2]))
            if off >= 100 and when is None:
                when = state["episode_time"]
                break
        assert trial.ok, trial.failure
        assert when is not None, seed
        assert angles[-1] < angles[0]
        dropped.append(when)
        max_speed = max(max_speed, trial.max_speed)
    save_case("T08_tilt", {"drop_time_s": dropped, "max_speed": max_speed}, seeds=list(range(5)), log={"drop_time_s": np.array(dropped)})


def test_t08_accel_drop(m1_config):
    spec = m1_config["diagnostic_accel"]
    scene = FeedingScene(m1_config)
    times = []
    max_speed = 0.0
    for seed in range(5):
        scene.reset(seed=10 + seed, preset="food_on_spoon")
        scene.apply_experiment(speed_limit=spec["speed_limit_rad_s"])
        trial = Trial(scene.speed_limit)
        off = 0
        when = None
        acc = []
        prev_v = None
        for k in range(int(1.6 / scene.dt)):
            t = k * scene.dt
            tau = scene.bias_tau()
            tau[int(spec["joint"])] += smooth_pulse(t, spec["start_s"], spec["duration_s"], spec["amplitude_nm"])
            state = scene.step_physics(tau, hold_driver=True)
            trial.observe(state)
            vel = state["qacc_arm"] * 0
            tcp_v = state["tcp_mat"][:, 0] * 0
            if prev_v is not None:
                acc.append(np.linalg.norm(state["food_vel"] - prev_v) / scene.dt)
            prev_v = state["food_vel"].copy()
            local = state["tcp_mat"].T @ (state["food_pos"] - state["tcp_pos"])
            outside = np.linalg.norm(local[:2]) > 0.02 or local[2] > 0.012 or local[2] < -0.004
            off = off + 1 if outside and not _food_on(state["contacts"], "spoon") else 0
            if off >= 100 and when is None:
                when = t
                break
        assert trial.ok, trial.failure
        assert when is not None, seed
        assert scene.speed_limit == spec["speed_limit_rad_s"]
        assert scene.speed_limit > m1_config["velocity_soft_limit_rad_s"]
        times.append(when)
        max_speed = max(max_speed, trial.max_speed)
    save_case(
        "T08_accel",
        {"drop_time_s": times, "max_speed": max_speed, "limit": spec["speed_limit_rad_s"], "reason": spec["reason"]},
        seeds=list(range(10, 15)),
    )


def test_t08_fault_sticks(m1_config):
    scene = FeedingScene(m1_config)
    scene.reset(seed=0, preset="food_on_plate")
    scene.apply_experiment(speed_limit=1e-6)
    trial = Trial(scene.speed_limit)
    q1 = scene.data.qpos[scene.index.arm_qpos_adr].copy()
    q1[0] += 0.4
    for k in range(400):
        q_ref, dq_ref = quintic(scene.config["q_carry"], q1, k * scene.dt, 0.3)
        state = scene.step_physics(scene.diagnostic_pd_tau(q_ref, 40, 8, dq_ref), hold_driver=True)
        trial.observe(state)
    assert scene.velocity_fault
    assert trial.failure is not None
    assert trial.failure["tick"] < state["tick"]
    save_case("T08_fault", {"first_tick": trial.failure["tick"], "max_speed": trial.max_speed}, status="passed")


def test_hold_latches_and_recovers(m1_config):
    scene = FeedingScene(m1_config)
    scene.reset(seed=1, preset="food_on_plate")
    assert scene._hold_latched is False
    state = scene.step_physics(scene.bias_tau(), hold_driver=True)
    target = state["hold_target"].copy()
    for _ in range(2000):
        state = scene.step_physics(scene.bias_tau(), hold_driver=True)
    assert np.allclose(state["hold_target"], target)
    assert np.max(np.abs(state["head_error"][:3])) <= 0.001
    spec = m1_config["hold_recovery"]
    scene.set_external_wrench(spec["force_n"], [0, 0, 0], state["mouth_pos"], body_name="head")
    for _ in range(int(spec["duration_s"] / scene.dt)):
        scene.step_physics(scene.bias_tau(), hold_driver=True)
    scene.clear_external_wrench()
    for _ in range(int(spec["recover_s"] / scene.dt)):
        state = scene.step_physics(scene.bias_tau(), hold_driver=True)
    assert np.max(np.abs(state["head_error"][:3])) <= spec["trans_tol_m"]
    assert abs(state["head_error"][3]) <= spec["ang_tol_rad"]
    assert np.allclose(state["hold_target"], target)
    scene.reset(seed=1, preset="food_on_plate")
    assert scene._hold_head_target is None and scene._hold_latched is False
    save_case("HOLD", {"recover_error_m": float(np.max(np.abs(state["head_error"][:3])))})


def test_t10_geometry_negatives(m1_config):
    # T10a: old world-x test would pass, mouth-frame x is still outside.
    mouth_pos = np.array([0.57717, 0.12, 0.35])
    tcp = np.array([0.55, 0.12, 0.35])
    rot = np.eye(3)
    info = passage(tcp, rot, mouth_pos, rot, 0.1)
    assert tcp[0] > 0.54
    assert info["tcp_local"][0] < -0.02
    assert not info["entered"]
    # T10b: tcp inside, bowl shifted out of the opening, and a rolled spoon.
    inside = mouth_pos + np.array([0.01, 0.0, 0.0])
    shifted = inside + np.array([0.0, 0.05, 0.0])
    assert not passage(shifted, rot, mouth_pos, rot, 0.1)["entered"]
    roll = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    assert not passage(inside, roll, mouth_pos, rot, 0.1)["entered"]
    # T10g: a shared yaw and translation does not change the decision.
    yaw = 0.4
    c, s = np.cos(yaw), np.sin(yaw)
    r_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    shift = np.array([0.2, -0.3, 0.05])
    base = passage(inside, rot, mouth_pos, rot, 0.1)["entered"]
    moved = passage(r_yaw @ inside + shift, r_yaw @ rot, r_yaw @ mouth_pos + shift, r_yaw @ rot, 0.1)["entered"]
    assert base and moved
    save_case("T10_negative", {"false_entry_x": float(info["tcp_local"][0])})


def test_t10_enter_exit_and_receiver(m1_config):
    scene = FeedingScene(m1_config)
    duration = float(m1_config["mouth_move"]["duration_s"])
    depths = []
    for seed, delta in enumerate((0.0, 0.04, -0.04, 0.06, -0.03)):
        scene.reset(seed=seed, preset="near_mouth", settle_steps=0)
        adr = scene.index.food_qpos_adr
        scene.data.qpos[adr : adr + 7] = [0.8, 0.35, -0.012, 1.0, 0.0, 0.0, 0.0]
        scene.data.qvel[scene.index.food_dof_adr : scene.index.food_dof_adr + 6] = 0
        q0 = np.array(m1_config["q_mouth_out"], dtype=float)
        q0[6] += delta
        q1 = np.array(m1_config["q_mouth_in"], dtype=float)
        scene.data.qpos[scene.index.arm_qpos_adr] = q0
        scene.data.qvel[:] = 0
        trial = Trial(scene.speed_limit)
        inside = 0
        entered = False
        steps = int((duration + 0.3) / scene.dt)
        for k in range(steps):
            q_ref, dq_ref = quintic(q0, q1, k * scene.dt, duration)
            state = scene.step_physics(
                scene.diagnostic_pd_tau(q_ref, m1_config["mouth_move"]["kp"], m1_config["mouth_move"]["kd"], dq_ref),
                hold_driver=True,
            )
            trial.observe(state)
            info = passage(state["tcp_pos"], state["tcp_mat"], state["mouth_pos"], state["mouth_mat"], state["jaw_q"])
            inside = inside + 1 if info["entered"] else 0
            if inside >= 100:
                entered = True
                depths.append(float(info["tcp_local"][0]))
                break
        assert trial.ok, (seed, trial.failure)
        assert entered, seed
        q_back = q0
        outside = 0
        clear = 0
        exited = False
        for k in range(steps):
            q_ref, dq_ref = quintic(q1, q_back, k * scene.dt, duration)
            state = scene.step_physics(
                scene.diagnostic_pd_tau(q_ref, m1_config["mouth_move"]["kp"], m1_config["mouth_move"]["kd"], dq_ref),
                hold_driver=True,
            )
            trial.observe(state)
            info = passage(state["tcp_pos"], state["tcp_mat"], state["mouth_pos"], state["mouth_mat"], state["jaw_q"])
            touching = any({c["group1"], c["group2"]} == {"spoon", "mouth"} for c in state["contacts"])
            outside = outside + 1 if info["exited"] else 0
            clear = clear + 1 if not touching else 0
            if outside >= 100 and clear >= 100:
                exited = True
                break
        assert trial.ok and exited, seed
    # receiver support is its own reset, not a transfer from the spoon
    scene.reset(seed=0, preset="near_mouth", settle_steps=0)
    scene.data.qpos[scene.index.arm_qpos_adr] = np.array(m1_config["q_mouth_out"])
    adr = scene.index.food_qpos_adr
    scene.data.qpos[adr : adr + 7] = [0.57, 0.12, 0.36, 1, 0, 0, 0]
    scene.data.qvel[:] = 0
    trial = Trial(scene.speed_limit)
    for _ in range(1200):
        state = scene.step_physics(scene.bias_tau(), hold_driver=True)
        trial.observe(state)
    assert trial.ok, trial.failure
    assert _named(state["contacts"], "jaw_floor")
    assert np.linalg.norm(state["food_vel"]) < 0.01
    local = to_mouth(state["food_pos"], state["mouth_pos"], state["mouth_mat"])
    assert 0.005 < local[0] < 0.038 and abs(local[1]) < 0.02
    # lip touch from a non-contact start
    scene.reset(seed=2, preset="near_mouth", settle_steps=0)
    adr = scene.index.food_qpos_adr
    scene.data.qpos[adr : adr + 7] = [0.8, 0.35, -0.012, 1.0, 0.0, 0.0, 0.0]
    scene.data.qvel[scene.index.food_dof_adr : scene.index.food_dof_adr + 6] = 0
    q0 = np.array(m1_config["q_mouth_out"], dtype=float)
    q1 = np.array(m1_config["q_mouth_lip"], dtype=float)
    scene.data.qpos[scene.index.arm_qpos_adr] = q0
    scene.data.qvel[:] = 0
    trial = Trial(scene.speed_limit)
    hit = None
    for k in range(int(3.2 / scene.dt)):
        q_ref, dq_ref = quintic(q0, q1, k * scene.dt, 3.0)
        state = scene.step_physics(
            scene.diagnostic_pd_tau(q_ref, m1_config["mouth_move"]["kp"], m1_config["mouth_move"]["kd"], dq_ref),
            hold_driver=True,
        )
        trial.observe(state)
        for row in state["contacts"]:
            names = {row["geom1"], row["geom2"]}
            if "mouth_upper" in names and any(name.startswith("bowl_") for name in names):
                hit = {"dist": row["dist"], "pos": row["pos"].tolist(), "force": row["force_on_geom2_world"].tolist()}
                break
        if hit:
            break
    assert hit is not None and hit["dist"] > -1e-3
    assert trial.ok, trial.failure
    save_case("T10_motion", {"entry_depth_m": depths, "lip_dist": hit["dist"], "max_speed": trial.max_speed}, seeds=list(range(5)), log={"entry_depth_m": np.array(depths)})
