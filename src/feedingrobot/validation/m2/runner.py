"""Physical M2 cases, all executed through the public torque-control step."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import mujoco
import numpy as np

from feedingrobot.controllers.cartesian_impedance import CartesianImpedance, control_step
from feedingrobot.controllers.guard import Guard
from feedingrobot.controllers.so3 import orientation_error, so3_exp
from feedingrobot.validation.m2.spec import Case, VARIANTS, SCHEMA
from feedingrobot.sim.scene import FeedingScene
from feedingrobot.sim.model import load_config
from feedingrobot.validation.m2.provenance import identity, clean_config, solver_options


def smooth(t, duration):
    u = np.clip(t / duration, 0., 1.)
    return float(10*u**3 - 15*u**4 + 6*u**5)


def tcp_twist(scene):
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(scene.model, scene.data, mujoco.mjtObj.mjOBJ_SITE,
                           scene.index.site_ids["tcp"], velocity, 0)
    return np.r_[velocity[3:], velocity[:3]]


def _ik_reset(scene, position, rotation):
    """Bounded deterministic reset-only IK; never used during an episode."""
    ids, cols = scene.index.arm_qpos_adr, scene.index.arm_dof_adr
    ranges = scene.model.jnt_range[scene.index.arm_joint_ids]
    for _ in range(300):
        mujoco.mj_forward(scene.model, scene.data)
        state = scene.snapshot()
        error = np.r_[position - state["tcp_pos"], orientation_error(rotation, state["tcp_mat"])]
        if np.linalg.norm(error) < 1e-8:
            return
        jp, jr = np.zeros((3, scene.model.nv)), np.zeros((3, scene.model.nv))
        mujoco.mj_jacSite(scene.model, scene.data, jp, jr, scene.index.site_ids["tcp"])
        jac = np.vstack([jp[:, cols], jr[:, cols]])
        delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5*np.eye(6), error)
        scene.data.qpos[ids] = np.clip(scene.data.qpos[ids] + np.clip(delta, -.03, .03), ranges[:, 0]+.02, ranges[:, 1]-.02)
    raise RuntimeError("reset IK did not converge")


def prepare(case: Case, seed: int, controller_config: dict):
    cfg = clean_config(controller_config)
    scene_cfg = load_config("configs/m1_scene.json")
    fixture = case.fixture
    if fixture:
        scene_cfg["model"] = "assets/tests/m2_full_wall.xml" if case.family == "wall" else "assets/tests/m2_full_surface.xml"
        cfg["press_test"] = True
    scene = FeedingScene(scene_cfg)
    preset = "food_on_spoon" if case.family == "carry" or case.context == "food" else "near_mouth" if case.family == "mouth" else "food_on_plate"
    state = scene.reset(seed=seed, preset=preset, settle_steps=0)
    reset_qpos = scene.data.qpos.copy()
    food_placement_attempt = None
    if case.family not in {"carry", "plate", "mouth"} and case.context != "food":
        q = np.array(scene.config["q_torque_poses"][case.pose % 3], dtype=float)
        if case.pose >= 3:
            q[6] += np.pi / 2
        q += np.random.default_rng(seed).uniform(-.01, .01, 7)
        scene.data.qpos[scene.index.arm_qpos_adr] = q
    if case.family == "mouth":
        _ik_reset(scene, np.array([.518, .12, .336]), state["tcp_mat"])
    mujoco.mj_forward(scene.model, scene.data)
    if case.family == "plate":
        _ik_reset(scene, np.array([.49, -.22, .030]), so3_exp(np.array([0., .5, 0.])) @ state["tcp_mat"])
        # The M1 reset checked food against the preset arm, before M2's IK move.
        # Use the same eight proposals, now checked against the actual start pose.
        placement_rng = np.random.default_rng(seed)
        for attempt in range(8):
            position = np.r_[np.array([.45, -.18]) + placement_rng.uniform(-.025, .025, 2), .03]
            scene._set_food(position, [1., 0., 0., 0.])
            mujoco.mj_forward(scene.model, scene.data)
            if not scene._illegal():
                food_placement_attempt = attempt
                break
        if food_placement_attempt is None:
            raise RuntimeError(f"no legal food placement after M2 IK: {case.key}, seed={seed}")
    if case.event == "singularity":
        from feedingrobot.validation.m2.spec import SINGULAR_Q, REST_DQ
        scene.data.qpos[scene.index.arm_qpos_adr] = SINGULAR_Q
        scene.data.qvel[scene.index.arm_dof_adr] = REST_DQ
        mujoco.mj_forward(scene.model,scene.data)
        scene.last_tau[:] = scene.data.qfrc_bias[scene.index.arm_dof_adr]
    if fixture:
        mount = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "test_mount")
        bowls = [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in cfg["contact"]["bowl_geoms"]]
        bottom = bowls[0]
        front = bowls[-1]
        normal = scene.data.geom_xpos[front] - scene.data.geom_xpos[bottom]
        normal[2] = 0.
        normal /= np.linalg.norm(normal)
        leading = max(scene.data.geom_xpos[i] @ normal + np.abs(normal @ scene.data.geom_xmat[i].reshape(3, 3)) @ scene.model.geom_size[i] for i in bowls)
        scene.model.body_pos[mount] = scene.data.geom_xpos[bottom] + normal*(leading - scene.data.geom_xpos[bottom] @ normal + .0015)
        angle = np.arctan2(normal[0], -normal[1])
        scene.model.body_quat[mount] = [np.cos(angle/2), 0., 0., np.sin(angle/2)]
        scene.fixture_normal = normal
        slide = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, "spring_slide")
        stiffness = case.stiffness if case.family == "spring" else 500.
        scene.model.jnt_stiffness[slide] = stiffness
        scene.model.dof_damping[scene.model.jnt_dofadr[slide]] = 2*np.sqrt(stiffness*.1)
        if case.family == "wall":
            scene.model.jnt_range[slide] = [0., 1e-6]
    if case.family in {"stiffness", "press"}:
        cfg["gears"]["FREE"]["kp"] = case.stiffness
    cfg["delay_s"] = case.delay
    cfg["noise"]["enable"] = case.noise
    cfg["noise"]["seed"] = seed * 10 + (int(case.event[-1]) if case.event.startswith("sensor") else 0)
    mujoco.mj_forward(scene.model, scene.data)
    state = scene.snapshot()
    if scene._illegal():
        raise RuntimeError(f"illegal initial state: {case.key}, seed={seed}")
    source = {'case': case.key, 'seed': seed, 'preset': preset,
              'food_placement_attempt': food_placement_attempt,
              'reset_qpos': reset_qpos.tolist(), 'pre_settle_qpos': scene.data.qpos.tolist(),
              'pre_settle_qvel': scene.data.qvel.tolist()}
    ctl = CartesianImpedance(scene, cfg)
    guard = Guard(cfg, cfg["joint_speed_limit_rad_s"])
    ctl.guard = guard
    ctl.reset(state, np.zeros(7))
    # One A-timebase settlement is shared by all variants. Hold cases explicitly test takeover from zero.
    if case.family != "hold" and case.event != "singularity":
        for k in range(1000):
            if k % 50 == 0:
                ctl.set_command(np.zeros(6), ctl.now(), ctl.now()+.1, "TRANSPORT")
            state, info = control_step(ctl, guard, hold_driver=case.family == "mouth")
            if guard.failure:
                raise RuntimeError(f"settlement fault {case.key}: {guard.failure}")
    snapshot = {name: np.array(getattr(scene.data, name), copy=True) for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")}
    snapshot["time"] = float(scene.data.time)
    snapshot["tau"] = scene.last_tau.copy()
    source['settled'] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in snapshot.items()}
    scene.m2_initial_source = source
    return scene, cfg, snapshot


def target(case, t):
    """Six-dimensional displacement from the frozen start reference."""
    x = np.zeros(6)
    if case.family == "step":
        x[case.axis] = case.sign * (.01 if case.axis < 3 else np.deg2rad(5)) * smooth(t-1., 1.)
    elif case.family in {"circle", "loaded_circle"}:
        u = np.clip(t-1., 0., 8.)
        x[:2] = [.01*(np.cos(2*np.pi*u/4)-1), .01*np.sin(2*np.pi*u/4)]
    elif case.family == "sine":
        u = np.clip(t-1., 0., 8.)
        x[3+case.axis] = np.deg2rad(3)*np.sin(2*np.pi*u/4)
    elif case.family in {"spring", "press"}:
        distance = .0065
        u = t-1.
        x[1] = -min(max(u, 0.)*.002, distance) + min(max(u-5.25, 0.)*.002, distance)
    elif case.family == "carry":
        x[0] = .01*(smooth(t-1., 1.)-smooth(t-4., 1.))
    elif case.family == "plate":
        x[2] = -.008*(smooth(t-1., 12.)-smooth(t-15., 12.))
    elif case.family == "mouth":
        x[0] = .006*(smooth(t-1., 12.)-smooth(t-15., 12.))
    return x


def command(case, t):
    if case.family == "wall":
        return np.array([0, -.02 if 1. <= t < 3. else 0., 0, 0, 0, 0])
    if case.family == "stop":
        x = np.zeros(6)
        if .8 <= t < 1.42:
            x[case.axis] = case.sign * (.04 if case.axis < 3 else .2)
        return x
    return (target(case, t+.05) - target(case, t))/.05


def run_variant(case, seed, variant, scene, cfg, initial):
    for name, value in initial.items():
        if name not in {"time", "tau"}:
            getattr(scene.data, name)[:] = value
    scene.data.time = initial["time"]
    scene._clear_python_state()
    scene.time_offset = initial["time"]
    scene.dt, scene.model.opt.iterations, scene.model.opt.tolerance = VARIANTS[variant]
    scene.model.opt.timestep = scene.dt
    options_before = solver_options(scene.model)
    assert (options_before['dt'], options_before['iterations'], options_before['tolerance']) == VARIANTS[variant]
    execution_id = uuid.uuid4().hex
    scene.data.xfrc_applied[:] = 0
    scene.data.qfrc_applied[:] = 0
    mujoco.mj_forward(scene.model, scene.data)
    ctl = CartesianImpedance(scene, cfg)
    guard = Guard(cfg, cfg["joint_speed_limit_rad_s"])
    ctl.guard = guard
    ctl.reset(scene.snapshot(), initial["tau"])
    initial_measurement = ctl.wrench._cache
    state = scene.snapshot()
    origin = state["tcp_pos"].copy()
    rotation = state["tcp_mat"].copy()
    states, infos, external, supplied = [state], [], [], []
    velocities = [tcp_twist(scene)]
    physical_dq = [scene.data.qvel[scene.index.arm_dof_adr].copy()]
    fixture_rows, timing = [], []
    references_before, command_times, tool_acceleration, arm_acceleration = [], [], [], []
    external_com_before, applied_wrenches = [], []
    import time
    snapshot_fn = scene.snapshot
    fault_tick = round(1.42 / scene.dt) + 1
    def injected_snapshot():
        s = snapshot_fn()
        if case.family == "stop" and scene.tick == fault_tick:
            if case.event in {"nan", "inf"}:
                s["raw_wrench_sensor"][0] = np.nan if case.event == "nan" else np.inf
            elif case.event == "warning":
                s["warnings"][0] = 1
            elif case.event in {"contact", "penetration"}:
                s["contacts"].append({"geom1": "tool_handle", "geom2": "plate_bottom", "geom1_id": mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_GEOM,"tool_handle"), "geom2_id": mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_GEOM,"plate_bottom"), "frame": np.eye(3).ravel(), "force_contact": np.zeros(3), "group1": "spoon", "group2": "plate", "dist": -.002 if case.event == "penetration" else 0., "force_on_geom2_world": np.zeros(3), "force_on_geom1_world": np.zeros(3), "pos": s["tcp_pos"].copy()})
        if case.family == "stop" and case.event == "speed" and fault_tick <= scene.tick <= round(1.422/scene.dt):
            s["dq"][0] = .51
        return s
    scene.snapshot = injected_snapshot
    try:
        for k in range(round(case.duration/scene.dt)):
            t = k*scene.dt
            scene.clear_external_wrench()
            force = np.zeros(6)
            point = snapshot_fn()["tcp_pos"].copy()
            if case.family == "static" and t >= 1.:
                force[case.axis] = case.sign * case.load
                if case.event == "lever":
                    point[0] += .03
            elif case.family in {"stiffness", "loaded_circle"} and t >= 1.:
                force[case.axis] = case.sign * case.load
            elif case.family == "pulse" and 1.5 <= t < 1.65:
                force[case.axis] = case.sign * (1. if case.axis < 3 else .02)
            elif case.family == "stop" and case.event == "wrench" and abs(t-1.42) < scene.dt/2:
                force[0] = 8.1
            if np.any(force):
                scene.set_external_wrench(force[:3], force[3:], point)
            if k % round(.05/scene.dt) == 0 and guard.status == "RUNNING":
                if not (case.family == "stop" and t >= 1.4-1e-10):
                    phase = "ACQUIRE" if case.family == "plate" and 0.5 <= t < 28. else "APPROACH" if case.family == "mouth" else "TRANSPORT"
                    requested = command(case, t)
                    if case.fixture:
                        requested[:3] = -requested[1] * scene.fixture_normal
                    ctl.set_command(requested, ctl.now(), ctl.now()+.1, phase)
            if case.family == "stop" and abs(t-1.4) < scene.dt/2:
                requested = command(case, t)
                if case.fixture:
                    requested[:3] = -requested[1] * scene.fixture_normal
                ctl.set_command(requested, ctl.now(), ctl.now()+.02 if case.event == "expired" else ctl.now()+.1, "TRANSPORT")
            if case.family == "stop" and abs(t-1.42) < scene.dt/2 and case.event in {"STOP", "RECOVER"}:
                ctl.set_command(np.zeros(6), ctl.now(), ctl.now()+.1, case.event)
            if case.family == "wall":
                aid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "test_retract_driver")
                scene.data.ctrl[aid] = .04 * smooth(t-3., .5)
            supplied.append(np.zeros(6) if ctl._command is None else ctl._command["twist"].copy())
            external.append(np.r_[force, point])
            reference_before = (ctl.reference.p_ref.copy(), ctl.reference.r_ref.copy())
            force_body_com = scene.data.xipos[scene.index.tool_body_id].copy()
            command_time = ([ctl._command['command_time'], ctl._command['valid_until']]
                            if ctl._command is not None else [-1., -1.])
            started = time.perf_counter()
            state, info = control_step(ctl, guard, hold_driver=case.family == "mouth")
            timing.append(time.perf_counter()-started)
            if not info["apply"]:
                supplied.pop(); external.pop(); timing.pop()
                break
            references_before.append(reference_before)
            external_com_before.append(force_body_com)
            applied_wrenches.append(scene.data.xfrc_applied[scene.index.tool_body_id].copy())
            command_times.append(command_time)
            accelerations = []
            mujoco.mj_rnePostConstraint(scene.model, scene.data)
            for body in ctl.wrench.body_ids:
                acceleration = np.zeros(6)
                mujoco.mj_objectAcceleration(scene.model, scene.data, mujoco.mjtObj.mjOBJ_BODY, body, acceleration, 0)
                acceleration[3:] += scene.model.opt.gravity
                accelerations.append(acceleration)
            tool_acceleration.append(accelerations)
            arm_acceleration.append(scene.data.qacc[scene.index.arm_dof_adr].copy())
            for contact_id, contact in enumerate(state['contacts']):
                wrench = np.zeros(6)
                if contact_id < scene.data.ncon:
                    mujoco.mj_contactForce(scene.model, scene.data, contact_id, wrench)
                contact['wrench_contact'] = wrench
            states.append(state); infos.append(info); velocities.append(tcp_twist(scene))
            physical_dq.append(scene.data.qvel[scene.index.arm_dof_adr].copy())
            if case.fixture:
                jid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, "spring_slide")
                rjid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, "test_retract")
                qa, da = scene.model.jnt_qposadr[jid], scene.model.jnt_dofadr[jid]
                retract = [0., 0., 0.]
                if rjid >= 0:
                    ra, rd = scene.model.jnt_qposadr[rjid], scene.model.jnt_dofadr[rjid]
                    retract = [scene.data.qpos[ra], scene.data.qvel[rd], scene.data.qacc[rd]]
                fixture_rows.append([scene.data.qpos[qa], scene.data.qvel[da], scene.data.qacc[da], *retract])
            if guard.status == "ABORTED":
                break
    finally:
        scene.snapshot = snapshot_fn
    n = len(infos)
    arrays = {key: np.array([s[key] for s in states]) for key in ("q", "dq", "tcp_pos", "tcp_mat", "food_pos", "food_quat", "raw_wrench_sensor", "qfrc_actuator", "warnings", "finite")}
    arrays.update({key: np.array([i[key] for i in infos]) for key in (
        "p_ref", "r_ref", "v_ref", "twist_command", "twist_limited", "tau_cmd", "tau_raw", "tau_bias", "tau_task", "tau_null", "tau_before", "k", "d", "phase_k", "rho", "execution", "status", "power_on", "blocked", "reference_correction_pos", "reference_correction_rot", "candidate_twist", "beta", "v_hist", "qdot_pred", "scale", "reanchored",
        "ft_compensated_wrench_tcp", "ft_delivered_wrench_tcp", "ft_estimated_kinematics", "ft_estimated_valid", "ft_valid", "ft_sample_time", "ft_sample_tick", "ft_filtered_wrench_tcp", "ft_tool_load_predicted", "ft_tcp_wrench_world", "ft_bias_wrench_tcp", "ft_tool_velocity", "ft_tool_com", "ft_tool_inertia", "ft_tcp_position")})
    arrays["physical_dq"] = np.array(physical_dq)
    arrays["blocked_now"] = np.array([i["blocked_now"] for i in infos])
    arrays["block_progress"] = np.array([i["block_progress"] for i in infos])
    arrays["block_correction_pos"] = np.array([i["block_correction_pos"] for i in infos])
    arrays["tool_acceleration"] = np.array(tool_acceleration)
    arrays["arm_acceleration"] = np.array(arm_acceleration)
    arrays["external_body_com_before"] = np.array(external_com_before)
    arrays["applied_wrench_com"] = np.array(applied_wrenches)
    arrays["p_ref_before"] = np.array([row[0] for row in references_before])
    arrays["r_ref_before"] = np.array([row[1] for row in references_before])
    arrays["command_times"] = np.array(command_times)
    arrays["correction_reason"] = np.array([i["correction_reason"] or "" for i in infos])
    arrays["constraint_reset"] = np.array([i["constraint_reset"] or "" for i in infos])
    arrays["initial_tool_velocity"] = initial_measurement["tool_velocity"]
    arrays["initial_compensated"] = initial_measurement["compensated_wrench_tcp"]
    arrays.update(t=np.array([s["episode_time"] for s in states]), tick=np.array([s["tick"] for s in states]), tcp_twist=np.array(velocities), external=np.array(external), supplied=np.array(supplied), origin=origin, rotation=rotation, initial_qpos=initial["qpos"], initial_qvel=initial["qvel"], initial_tau=initial["tau"], fixture=np.array(fixture_rows).reshape(-1, 6), wall_seconds=np.array(timing))
    arrays["age"] = np.array([np.nan if i["ft_age_s"] is None else i["ft_age_s"] for i in infos])
    offsets, contacts = [0], []
    for s in states[1:]:
        contacts.extend(s["contacts"]); offsets.append(len(contacts))
    arrays["contact_offsets"] = np.array(offsets)
    for name in ("geom1", "geom2", "dist"):
        arrays["contact_"+name] = np.array([c[name] for c in contacts], dtype=str if name != "dist" else float)
    arrays["contact_force"] = np.array([c["force_on_geom2_world"] for c in contacts]).reshape(-1, 3)
    for key, width in (("pos",3),("frame",9),("force_contact",3),("wrench_contact",6)):
        arrays["contact_"+key] = np.array([c[key] for c in contacts]).reshape(-1,width)
    frame = arrays['contact_frame'].reshape(-1, 3, 3)
    local = arrays['contact_wrench_contact']
    arrays['contact_wrench_world'] = np.concatenate([
        np.einsum('nji,nj->ni', frame, local[:, :3]),
        np.einsum('nji,nj->ni', frame, local[:, 3:]),
    ], axis=1)
    for key in ("geom1_id","geom2_id"):
        arrays["contact_"+key] = np.array([c[key] for c in contacts],dtype=int)
    abort_probe = None
    if guard.status == "ABORTED":
        tick, now = scene.tick, scene.data.time
        applied = [control_step(ctl,guard)[1]["apply"] for _ in range(3)]
        abort_probe = {"tick_before":tick,"tick_after":scene.tick,"time_before":now,"time_after":scene.data.time,"apply":applied}
    metadata = {**case.manifest(seed, variant), "schema_version": SCHEMA, "fault": guard.failure, "events": guard.events,
                "stopped_ok": guard.stopped_ok, "actual_duration": n*scene.dt, "abort_probe":abort_probe,
                "effective_controller": cfg, "joint_ranges": ctl.ranges.tolist(),
                "normal": scene.fixture_normal.tolist() if case.fixture else None,
                "mount": scene.model.body_pos[mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "test_mount")].tolist() if case.fixture else None}
    options_after = solver_options(scene.model)
    if options_after != options_before:
        raise RuntimeError('solver changed during execution')
    metadata.update(initial_source=scene.m2_initial_source, geometry_registry=guard.registry,
                    execution={'id': execution_id, 'case': case.manifest(seed, variant),
                               'solver_before': options_before, 'solver_after': options_after})
    arrays.update(execution_id=np.array(execution_id), case_identity=np.array(identity(case.manifest(seed, variant))),
                  config_identity=np.array(identity(cfg)), solver_identity=np.array(identity(options_before)),
                  initial_source_identity=np.array(identity(scene.m2_initial_source)))
    return arrays, metadata


def run_group(case: Case, seed: int, config: dict, output: str | None = None):
    scene, cfg, initial = prepare(case, seed, config)
    results = []
    for variant in VARIANTS:
        arrays, metadata = run_variant(case, seed, variant, scene, cfg, initial)
        if output is not None:
            path = Path(output) / metadata["case_id"]
            temporary = Path(str(path) + ".npz.partial")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(str(path) + ".npz")
            temporary = Path(str(path) + ".json.partial")
            temporary.write_text(json.dumps(metadata, indent=2))
            temporary.replace(str(path) + ".json")
            from feedingrobot.validation.m2.provenance import digest
            receipt = {'case': case.manifest(seed, variant), 'execution': metadata['execution'],
                       'initial_source_identity': str(arrays['initial_source_identity'].item()),
                       'config_identity': str(arrays['config_identity'].item()),
                       'npz_sha256': digest(str(path)+'.npz'), 'json_sha256': digest(str(path)+'.json'), 'complete': True}
            temporary = Path(str(path)+'.execution.json.partial')
            temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False))
            temporary.replace(str(path)+'.execution.json')
        results.append((arrays, metadata))
    return results
