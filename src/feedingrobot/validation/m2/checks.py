"""Read-only scoring of the full matrix; no controller calls or physics steps."""
from __future__ import annotations

import numpy as np

from feedingrobot.validation.m2.spec import Case, VARIANTS, SCHEMA, TAU_MAX, RANGES, GEARS
from feedingrobot.validation.m2.contact_checks import recorded_contact_allowed
from feedingrobot.controllers.contracts import GEAR_OF
from feedingrobot.validation.m2.reference_checks import expected_stimulus



def angles(a, b):
    return np.arccos(np.clip((np.sum(a*b, axis=(-2, -1))-1)/2, -1., 1.))


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def envelope(values, floor):
    values = np.asarray(values)
    centered = values - np.mean(values, axis=0)
    windows = np.array_split(centered, 4)
    return rms(windows[-1]) <= 1.1*rms(windows[0]) + floor



def score(case: Case, seed, variant, data, meta, stop_damping=None, base_config=None):
    d = {key: np.asarray(value) for key, value in data.items()}
    failures = []
    seen = set()
    def describe(rows):
        return [{'case_id': case.manifest(seed, variant)['case_id'], 'seed': seed, 'variant': variant,
                 'field': row['code'].split(':', 1)[-1], **row} for row in rows]
    def check(ok, code, observed=None):
        if not bool(ok) and code not in seen:
            seen.add(code)
            failures.append({"code": code, "observed": observed})
    expected = case.manifest(seed, variant)
    check(meta.get('schema_version') == SCHEMA, 'trace_schema')
    for key, value in expected.items():
        check(meta.get(key) == value, "identity:"+key)
    from feedingrobot.validation.m2.provenance import check_identity
    failures.extend(check_identity(case, seed, variant, d, meta, base_config))
    if stop_damping is None:
        from feedingrobot.sim.model import load_config
        stop_damping = load_config('configs/m2_controller.json')['stop']['damping_nm_s_per_rad']
    dt = VARIANTS[variant][0]
    n = len(d.get("tau_cmd", []))
    from feedingrobot.validation.m2.contact_checks import check_contacts
    contact_failures = check_contacts(d)
    if contact_failures:
        return describe(failures + contact_failures), {}
    abort = case.family == "stop" and case.event in {"nan", "inf", "warning"}
    expected_n = round(1.42/dt)+1 if abort else round(case.duration/dt)
    check(n == expected_n, "sample_count", n)
    if not n:
        return describe(failures), {}
    state_shapes = {"q": (n+1, 7), "dq": (n+1, 7), "tcp_pos": (n+1, 3), "tcp_mat": (n+1, 3, 3),
                    "physical_dq": (n+1,7),
                    "raw_wrench_sensor": (n+1, 6), "qfrc_actuator": (n+1, 7), "tcp_twist": (n+1, 6),
                    "food_pos": (n+1, 3), "food_quat": (n+1, 4), "t": (n+1,), "tick": (n+1,)}
    interval_shapes = {"p_ref": (n,3), "r_ref": (n,3,3), "v_ref": (n,6), "external": (n,9), "supplied": (n,6),
                       "blocked_now": (n,), "block_progress": (n,6), "block_correction_pos": (n,3),
                       "p_ref_before": (n,3), "r_ref_before": (n,3,3), "command_times": (n,2),
                       "arm_acceleration": (n,7), "candidate_twist": (n,6), "beta": (n,),
                       "external_body_com_before": (n,3), "applied_wrench_com": (n,6),
                       "v_hist": (n,6), "qdot_pred": (n,7), "scale": (n,), "reanchored": (n,),
                       "reference_correction_rot": (n,), "correction_reason": (n,), "constraint_reset": (n,),
                       "tau_before": (n,7), "tau_cmd": (n,7), "tau_raw": (n,7), "tau_task": (n,7), "tau_null": (n,7), "tau_bias": (n,7),
                       "ft_compensated_wrench_tcp": (n,6), "ft_estimated_kinematics": (n,6), "ft_delivered_wrench_tcp": (n,6),
                       "ft_valid": (n,), "ft_estimated_valid": (n,), "ft_sample_time": (n,), "ft_sample_tick": (n,),
                       "k": (n,6), "d": (n,6), "phase_k": (n,), "execution": (n,), "status": (n,), "twist_limited": (n,6)}
    for key, shape in {**state_shapes, **interval_shapes}.items():
        check(key in d and d[key].shape == shape, "shape:"+key)
    if any(f["code"].startswith("shape:") for f in failures):
        return describe(failures), {}
    for key in ("ft_tcp_wrench_world","ft_bias_wrench_tcp","ft_tool_load_predicted","ft_tool_velocity","ft_tool_com","ft_tool_inertia","ft_tcp_position","initial_tool_velocity","initial_compensated","contact_pos","contact_frame","contact_force_contact","contact_geom1_id","contact_geom2_id"):
        check(key in d,"missing:"+key)
    if failures:
        return describe(failures), {}
    from feedingrobot.validation.m2.contact_checks import check_tool_balance
    failures.extend(check_tool_balance(d, meta['geometry_registry']))
    fault_row = round(1.42/dt)
    if case.family == 'stop' and case.event in {'contact', 'penetration'}:
        start, end = d['contact_offsets'][fault_row:fault_row+2]
        injected = [j for j in range(start, end)
                    if d['contact_geom1'][j] == 'tool_handle' and d['contact_geom2'][j] == 'plate_bottom']
        check(len(injected) == 1, 'contact_stimulus', len(injected))
        if len(injected) == 1:
            j = injected[0]
            ids = {row['name']: row['id'] for row in meta['geometry_registry']}
            check(d['contact_geom1_id'][j] == ids['tool_handle']
                  and d['contact_geom2_id'][j] == ids['plate_bottom']
                  and abs(d['contact_dist'][j] - (-.002 if case.event == 'penetration' else 0.)) < 1e-12
                  and np.array_equal(d['contact_frame'][j], np.eye(3).ravel())
                  and not np.any(d['contact_wrench_contact'][j])
                  and np.allclose(d['contact_pos'][j], d['tcp_pos'][fault_row+1], atol=1e-12, rtol=0),
                  'contact_stimulus')
    expected_command, expected_external, expected_phase = expected_stimulus(case,np.arange(n)*dt)
    if case.fixture:
        normal = np.asarray(meta.get("normal",[]))
        check(normal.shape == (3,) and abs(np.linalg.norm(normal)-1.) < 1e-9 and abs(normal[2])<1e-12, "fixture_normal")
        if normal.shape == (3,): expected_command[:,:3] = -expected_command[:,1,None]*normal
    check(np.allclose(d["supplied"], expected_command,atol=1e-10,rtol=0), "command_stimulus")
    check(np.allclose(d["external"][:,:6],expected_external,atol=1e-10,rtol=0), "load_stimulus")
    points = d["tcp_pos"][:-1].copy()
    if case.event == "lever": points[:,0] += .03
    check(np.allclose(d["external"][:,6:][np.any(expected_external!=0,axis=1)],points[np.any(expected_external!=0,axis=1)],atol=1e-9), "load_point")
    check(np.array_equal(d["phase_k"],expected_phase), "phase_stimulus")
    from feedingrobot.validation.m2.reference_checks import rebuild_gain_path, check_analytic_motion
    failures.extend(check_analytic_motion(case, d))
    gain = rebuild_gain_path(expected_phase,dt)
    if "K" in gain:
        expected_k, expected_d = gain["K"],gain["D"]
        if case.family in {"press","stiffness"}: expected_k[:,:3] = case.stiffness
        stopped = d["execution"] == "stop"
        expected_k[stopped],expected_d[stopped] = 0.,0.
        expected_d[d["execution"] == "power_on"] *= 8.
        check(np.allclose(d["k"],expected_k,atol=1e-9,rtol=0),"gain_k")
        check(np.allclose(d["d"],expected_d,atol=1e-9,rtol=0),"gain_d")
    for key, array in d.items():
        if array.dtype.kind not in "fiub":
            continue
        finite = np.isfinite(array)
        if key in {"ft_delivered_wrench_tcp", "age"}:
            # Only delay warmup lacks a delivered sample. Filter warmup is finite.
            finite[d["t"][1:] < case.delay-1e-12] = True
            if case.family == "stop" and case.event in {"nan", "inf"}:
                finite[fault_row] = True
        elif case.family == "stop" and case.event in {"nan", "inf"}:
            if key == "raw_wrench_sensor":
                finite[fault_row+1, 0] = True
            elif key == "ft_compensated_wrench_tcp":
                finite[fault_row] = True
            elif key == "ft_estimated_kinematics":
                finite[fault_row] = True
            elif key == "ft_tcp_wrench_world":
                finite[fault_row] = True
        check(np.all(finite), "nonfinite:"+key)
    check(d["tick"].dtype.kind in "iu" and np.array_equal(d["tick"], np.arange(n+1)), "tick")
    check(np.allclose(d["t"], np.arange(n+1)*dt, atol=1e-9, rtol=0), "time")
    check(np.allclose(d["ft_sample_time"], d["t"][1:], atol=1e-9, rtol=0), "sample_time")
    check(np.array_equal(d["ft_sample_tick"], d["tick"][1:]), "sample_tick")
    for key in ("tcp_mat", "r_ref"):
        rr = d[key]
        check(np.allclose(rr @ np.swapaxes(rr, -1, -2), np.eye(3), atol=1e-7) and np.allclose(np.linalg.det(rr), 1., atol=1e-7), "rotation:"+key)
    check(np.all(d["q"] >= RANGES[:,0]-1e-8) and np.all(d["q"] <= RANGES[:,1]+1e-8), "joint_range")
    speed = np.max(np.abs(d["dq"]), axis=1)
    speed_check = speed.copy()
    if case.family == "stop" and case.event == "speed":
        check(abs(d["dq"][fault_row+1, 0]-.51) <= 1e-12, "speed_stimulus")
        last = fault_row+round(.002/dt)+1
        check(np.all(d["dq"][fault_row+1:last,0] == .51), "speed_stimulus_duration")
        speed_check[fault_row+1:last] = np.max(np.abs(d["dq"][fault_row+1:last,1:]),axis=1)
    check(np.max(speed_check) <= .5+1e-9, "joint_speed", float(np.max(speed_check)))
    check(np.all(np.abs(d["tau_cmd"]) <= TAU_MAX+1e-9), "torque_limit")
    previous = np.vstack([d["initial_tau"], d["tau_cmd"][:-1]])
    check(np.allclose(d["tau_before"], previous, atol=1e-9, rtol=0), "torque_history")
    check(np.max(np.abs(d["tau_cmd"]-previous)) <= 2000*dt+1e-9, "torque_rate")
    limited = np.clip(d["tau_raw"], np.maximum(-TAU_MAX, previous-2000*dt), np.minimum(TAU_MAX, previous+2000*dt))
    check(np.allclose(d["tau_cmd"], limited, atol=1e-9, rtol=1e-7), "torque_chain")
    check(np.allclose(d["qfrc_actuator"][1:], d["tau_cmd"], atol=1e-8, rtol=0), "actuator_torque")
    warnings = d["warnings"].copy()
    if case.family == "stop" and case.event == "warning":
        check(warnings[fault_row+1,0] == 1, "warning_stimulus")
        warnings[fault_row+1,0] = 0
    check(not np.any(warnings), "warning")
    check(np.all(d["finite"]), "state_finite")
    pe = np.linalg.norm(d["p_ref"]-d["tcp_pos"][1:], axis=1)
    re = angles(d["r_ref"], d["tcp_mat"][1:])
    force = np.linalg.norm(d["ft_compensated_wrench_tcp"][:,:3], axis=1)
    moment = np.linalg.norm(d["ft_compensated_wrench_tcp"][:,3:], axis=1)
    speeds = np.linalg.norm(d["tcp_twist"][1:,:3], axis=1)
    spins = np.linalg.norm(d["tcp_twist"][1:,3:], axis=1)
    offsets = d["contact_offsets"]
    check(offsets.dtype.kind in "iu" and offsets.shape == (n+1,) and offsets[0] == 0 and np.all(np.diff(offsets)>=0) and offsets[-1] == len(d["contact_dist"]), "contact_offsets")
    if any(f["code"] == "contact_offsets" for f in failures):
        return describe(failures), {}
    contact_ticks = []
    for k in range(n):
        phase = str(d["phase_k"][k])
        try:
            limits = GEARS[GEAR_OF[phase]]
        except KeyError:
            check(False, "phase"); continue
        stopping = d["execution"][k] == "stop"
        check(np.linalg.norm(d["twist_limited"][k,:3]) <= (0. if stopping else limits[0])+1e-9, "command_speed")
        check(np.linalg.norm(d["twist_limited"][k,3:]) <= (0. if stopping else limits[1])+1e-9, "command_angular_speed")
        check(np.linalg.norm(d["p_ref"][k]-d["tcp_pos"][k]) <= limits[2]+1e-9, "reference_deviation")
        check(angles(d["r_ref"][k],d["tcp_mat"][k]) <= limits[3]+1e-8,"reference_rotation_deviation")
        transient = case.family in {"stop", "pulse"}
        check(speeds[k] <= (.10 if transient else 1.2*limits[0])+1e-9, "tcp_speed", float(speeds[k]))
        check(spins[k] <= (.5 if transient else 1.2*limits[1])+1e-9, "tcp_angular_speed", float(spins[k]))
        injected_wrench = case.family == "stop" and case.event in {"nan", "inf", "wrench"} and k == fault_row
        if not injected_wrench:
            check(force[k] <= limits[4]+1e-9 and moment[k] <= limits[5]+1e-9, "wrench_limit")
        for j in range(int(offsets[k]), int(offsets[k+1])):
            a, b = str(d["contact_geom1"][j]), str(d["contact_geom2"][j])
            injected_contact = case.family == "stop" and case.event in {"contact", "penetration"} and k == fault_row and {a,b} == {"tool_handle","plate_bottom"}
            if injected_contact:
                continue
            check(d["contact_dist"][j] >= -.001, "penetration")
            row = {'geom1': None if a == 'None' else a, 'geom2': None if b == 'None' else b,
                   'geom1_id': d['contact_geom1_id'][j], 'geom2_id': d['contact_geom2_id'][j]}
            check(recorded_contact_allowed(phase, row, {"press_test": case.fixture}, meta['geometry_registry']), "forbidden_contact")
            if (a.startswith("bowl_") or b.startswith("bowl_")) and ({a,b} & {"spring_pad","plate_bottom","jaw_lip"}):
                contact_ticks.append(k)
    valid = d["ft_valid"].astype(bool)
    for key in ("ft_valid", "ft_estimated_valid"):
        check(d[key].dtype.kind in "biu" and np.all(np.isin(d[key], [0,1])), "boolean:"+key)
    estimated_valid = np.ones(n,dtype=bool)
    if case.family == "stop" and case.event in {"nan", "inf"}:
        estimated_valid[-1] = False
        raw = d["raw_wrench_sensor"][-1,0]
        check(np.isnan(raw) if case.event == "nan" else np.isposinf(raw), "nonfinite_stimulus")
    check(np.array_equal(d["ft_estimated_valid"],estimated_valid), "estimated_validity")
    check(np.all(np.isfinite(d["ft_delivered_wrench_tcp"][valid])), "valid_nan")
    check(np.allclose(d["age"][valid], case.delay, atol=dt+1e-12, rtol=0), "age")
    expected_valid = d["t"][1:] - case.delay >= 1/30.-1e-12
    if case.family == "stop" and case.event in {"nan", "inf"}:
        expected_valid[-1] = False
    check(np.array_equal(valid, expected_valid), "validity")
    events = {}
    if contact_ticks:
        events["contact"] = (min(contact_ticks)+1)*dt
        events["last_contact"] = (max(contact_ticks)+1)*dt
    if case.family != "stop":
        check(meta.get("fault") is None, "unexpected_fault", meta.get("fault"))
        check(not np.any(np.isin(d["status"], ["STOPPING","STOPPED","ABORTED"])), "unexpected_stop")
    else:
        start = 0 if case.event == "singularity" else round(1.42/dt) + (0 if case.event in {"expired","STOP","RECOVER"} else 1)
        events["stop_trigger"] = start*dt
        if case.event in {"STOP","RECOVER"}:
            check(meta.get("fault") is None, "fabricated_fault")
            requests = [e for e in meta.get("events",[]) if e.get("kind") == "stop_request"]
            check(len(requests) == 1 and requests[0]["tick"] == start, "stop_request")
        else:
            reason = {"expired":"command_expired", "contact":"forbidden_contact", "nan":"non-finite-wrench", "inf":"non-finite-wrench"}.get(case.event, case.event)
            fault = meta.get("fault") or {}
            check(fault.get("reason") == reason and fault.get("tick") == start and abs(fault.get("time",-1)-start*dt) < 1e-9, "first_fault", fault)
        if abort:
            check(d["ft_valid"][-1] == (case.event == "warning"), "abort_validity")
            probe=meta.get("abort_probe") or {}
            check(probe.get("apply")==[False,False,False] and probe.get("tick_before")==n and probe.get("tick_after")==n and probe.get("time_before")==probe.get("time_after"),"abort_advanced")
        else:
            check(np.all(d["execution"][start:] == "stop"), "late_stop")
            check(not np.any(d["supplied"][start+1:]), "auto_resume")
            check(np.allclose(d["tau_raw"][start:], d["tau_bias"][start:]-stop_damping*d["dq"][start:n], atol=1e-8), "stop_torque")
            check(not np.any(d["tau_task"][start:]) and not np.any(d["tau_null"][start:]), "stop_task")
            count = round(.2/dt)+1
            held = next((k for k in range(start, min(n-count+2, start+round(.5/dt)+1)) if np.all(speed[k:k+count] <= .02+1e-12)), None)
            check(held is not None, "stop_window")
            if held is not None:
                events["stop_speed"] = held*dt
                events["stop_confirm"] = (held+count-1)*dt
    t = d["t"][1:]
    tail = t >= case.duration-1.-1e-9
    moving = (t >= 1.) & (t <= 9.)
    if case.family in {'loaded_circle', 'spring', 'press', 'stiffness'}:
        check(not np.any(d['blocked_now']), 'normal_motion_blocked')
    if case.family == "hold":
        check(rms(pe[tail]) <= .001 and rms(re[tail]) <= np.deg2rad(.5), "hold_error")
    elif case.family == "step":
        amplitude = .01 if case.axis < 3 else np.deg2rad(5)
        if case.axis < 3:
            progress = case.sign*(d["tcp_pos"][1:,case.axis]-d["origin"][case.axis])
            reference = case.sign*(d["p_ref"][:,case.axis]-d["origin"][case.axis])
        else:
            from feedingrobot.controllers.so3 import so3_log
            progress = case.sign*np.array([so3_log(r @ d["rotation"].T)[case.axis-3] for r in d["tcp_mat"][1:]])
            reference = case.sign*np.array([so3_log(r @ d["rotation"].T)[case.axis-3] for r in d["r_ref"]])
        check(abs(reference[-1]-amplitude) <= (1e-5 if case.axis < 3 else 1e-4), "step_amplitude", float(reference[-1]))
        complete = np.flatnonzero((t>=2.) & (np.abs(reference-amplitude) <= (1e-5 if case.axis < 3 else 1e-4)))
        check(len(complete)>0, "shaping_complete")
        if len(complete):
            end = complete[0]+round(1.5/dt)
            check(np.all(pe[end:end+round(.5/dt)] <= .002) and np.all(re[end:end+round(.5/dt)] <= np.deg2rad(1)), "step_settle")
            events["shaping_complete"] = float(t[complete[0]])
        check(np.max(progress)-amplitude <= max(.2*amplitude, .0005 if case.axis<3 else np.deg2rad(.2)), "overshoot")
    elif case.family in {"circle", "sine", "loaded_circle"}:
        saturated=np.any(np.abs(d["tau_cmd"]-d["tau_raw"])>1e-8,axis=1)
        longest=current=0
        for active in saturated:
            current=current+1 if active else 0
            longest=max(longest,current)
        check(longest*dt<.02-1e-12,"saturation_duration")
        if case.family != "loaded_circle":
            check(rms(pe[moving]) <= .002 and np.max(pe[moving]) <= .005, "tracking_position")
            check(rms(re[moving]) <= np.deg2rad(1) and np.max(re[moving]) <= np.deg2rad(3), "tracking_rotation")
        expected_wrench = np.zeros((n,6))
        if case.family == "loaded_circle":
            expected_wrench[t > 1., case.axis] = case.sign*case.load
        residual = d["ft_compensated_wrench_tcp"]-expected_wrench
        estimated = d["ft_estimated_kinematics"]-expected_wrench
        ev = d["ft_estimated_valid"].astype(bool)
        for label, array, mask, limits in (("exact",residual,np.ones(n,dtype=bool),(.05,.005,.15,.015)), ("estimated",estimated,ev,(.1,.01,.3,.03))):
            check(np.any(mask), label+"_missing")
            if np.any(mask):
                ff,mm = np.linalg.norm(array[mask,:3],axis=1), np.linalg.norm(array[mask,3:],axis=1)
                check(rms(ff)<=limits[0] and rms(mm)<=limits[1] and np.max(ff)<=limits[2] and np.max(mm)<=limits[3], label+"_wrench")
    elif case.family == "static":
        expected_wrench = np.zeros(6); expected_wrench[case.axis] = case.sign*case.load
        if case.event == "lever":
            expected_wrench[3:] += np.cross(np.array([.03,0,0]),expected_wrench[:3])
        error = np.mean(d["ft_compensated_wrench_tcp"][tail],axis=0)-expected_wrench
        limit = np.maximum(np.array([.02]*3+[.002]*3), .02*np.abs(expected_wrench))
        check(np.all(np.abs(error)<=limit), "static_wrench")
        if case.load == 0:
            check(np.linalg.norm(error[:3])<=.02 and np.linalg.norm(error[3:])<=.002, "unloaded_bias")
    elif case.family in {"spring","press"}:
        hold = (t >= 5.25) & (t <= 6.25)
        check(len(contact_ticks)>0 and max(contact_ticks)*dt > 5.25, "contact_missing")
        normal = np.asarray(meta["normal"])
        contact_force = np.zeros(n)
        for k in range(n):
            for j in range(offsets[k], offsets[k+1]):
                if d["contact_geom2"][j] == "spring_pad": contact_force[k] += d["contact_force"][j] @ normal
                elif d["contact_geom1"][j] == "spring_pad": contact_force[k] -= d["contact_force"][j] @ normal
        x = d["fixture"]
        stiffness = case.stiffness if case.family == "spring" else 500.
        reconstructed = stiffness*x[:,0] + 2*np.sqrt(.1*stiffness)*x[:,1] + .1*(x[:,2]+x[:,5])
        mean = float(np.mean(contact_force[hold]))
        check(mean > .01 and np.max(x[:,0]) > 1e-5, "spring_response")
        check(np.max(x[:,0]) < .0199, "spring_stop")
        check(abs(np.mean(reconstructed[hold])-mean) <= max(.05,.1*abs(mean)), "spring_force_balance")
        check(rms(contact_force[hold]-mean) <= max(.05,.1*abs(mean)), "spring_oscillation")
        check(envelope(contact_force[hold],.01), "spring_envelope")
        check(np.max(force)<=2., "spring_peak")
        check(not any(k*dt>=9.5 for k in contact_ticks), "spring_exit")
    elif case.family == "stiffness":
        displacement = np.mean(d["tcp_pos"][1:][tail]-d["p_ref"][tail],axis=0)[case.axis]*case.sign
        prediction = case.load/case.stiffness
        check(displacement>0 and abs(displacement-prediction) <= .3*prediction, "stiffness_prediction", float(displacement))
        check(not np.any(d["reference_correction_pos"]), "stiffness_clipped")
    elif case.family == "wall":
        check(len(contact_ticks)>0 and np.any(d["blocked"]), "wall_blocked")
        confirmed = np.linalg.norm(d['block_correction_pos'], axis=1) > 1e-12
        check(np.any(confirmed & d['blocked_now'].astype(bool)), 'wall_progress_blocked')
        check(np.max(d["fixture"][:,3]) >= .035, "wall_not_retracted")
        release = round(3./dt)
        span = slice(release,release+round(.5/dt)+1)
        check(np.max(speeds[span])<=.03, "release_speed")
        extra = (d["tcp_pos"][1:][span]-d["tcp_pos"][release]) @ np.asarray(meta["normal"])
        check(np.max(extra)<=.005, "release_distance")
        stable = (t>=3.8)&(t<=4.)
        check(np.max(speeds[stable])<=.005 and np.max(spins[stable])<=.05, "release_settle")
        events["release"] = 3.
    elif case.family == "pulse":
        recovery = (t>=3.15)&(t<=3.65)
        check(np.all(pe[recovery]<=.002) and np.all(re[recovery]<=np.deg2rad(1)), "pulse_recovery")
        check(envelope(pe[tail],.0001) and envelope(re[tail],np.deg2rad(.05)), "pulse_envelope")
        events["release"] = 1.65
    elif case.family == "carry":
        local = np.einsum("ni,nij->nj",d["food_pos"]-d["tcp_pos"],d["tcp_mat"])
        supported = (np.abs(local[:,:2])<=.02).all(axis=1)&(local[:,2]>=-.004)&(local[:,2]<=.012)
        longest = current = 0
        for good in supported:
            current = 0 if good else current+1; longest=max(longest,current)
        check(longest*dt < .1, "food_lost")
        check(abs((d["p_ref"][:,0]-d["origin"][0]).max()-.01)<.0001, "carry_distance")
        check(np.linalg.norm(d["tcp_pos"][-1]-d["origin"])<=.002, "carry_return")
        check(np.mean(force[tail])>.02,"food_load_zeroed")
    elif case.family in {"plate","mouth"}:
        check(len(contact_ticks)>0, "bridge_contact_missing")
        check(np.max(force) > .01, "bridge_contact_force")
        check(not any(k*dt>=28. for k in contact_ticks), "bridge_exit")
    finite_force = np.nan_to_num(force)
    if not any(row["code"].startswith(("nonfinite:","shape:","rotation:")) for row in failures):
        from feedingrobot.validation.m2.physics_checks import check_physics,check_sensor_chain,check_tool_dynamics
        from feedingrobot.validation.m2.reference_checks import check_reference_path
        try:
            failures.extend(check_physics(d,case))
            failures.extend(check_sensor_chain(d,case,seed,dt))
            failures.extend(check_tool_dynamics(d))
            failures.extend(check_reference_path(d,meta['effective_controller'],dt))
        except (ValueError,KeyError,IndexError,TypeError,np.linalg.LinAlgError) as error:
            check(False,"physical_evidence",str(error))
    metrics = {"position_rms": rms(pe[tail]) if np.any(tail) else rms(pe), "rotation_rms": rms(re[tail]) if np.any(tail) else rms(re),
               "force_peak": float(np.max(finite_force)), "force_mean": float(np.mean(finite_force[tail])) if np.any(tail) else 0.,
               "moment_peak": float(np.nanmax(moment)), "tau_peak": np.max(np.abs(d["tau_cmd"]),axis=0).tolist(),
               "impulse": float(np.sum(finite_force)*dt), "events": events,
               "initial_qpos": d["initial_qpos"].tolist(), "initial_qvel": d["initial_qvel"].tolist()}
    metrics.update(initial_source_identity=str(d['initial_source_identity'].item()),
                   execution_id=str(d['execution_id'].item()), variant=variant)
    metrics["tracking_position_rms"]=rms(pe[moving]) if np.any(moving) else rms(pe)
    from feedingrobot.validation.m2.contact_checks import tool_contact_wrench
    try:
        contact_wrench = tool_contact_wrench(d, meta['geometry_registry'])
        contact_force = np.linalg.norm(contact_wrench[:, :3], axis=1)
        metrics.update(contact_force_peak=float(np.max(contact_force)),
                       contact_moment_peak=float(np.max(np.linalg.norm(contact_wrench[:, 3:], axis=1))),
                       contact_impulse=float(np.sum(contact_force) * dt))
    except ValueError:
        check(False, 'contact_geom_identity')
    metrics["tracking_rotation_rms"]=rms(re[moving]) if np.any(moving) else rms(re)
    if case.family=="stiffness":metrics["stiffness_displacement"]=float(displacement)
    if case.family in {"spring","press"}:metrics["contact_force_mean"]=mean
    # Keep the first occurrence of each failed predicate, with its measured value.
    unique = {row["code"]:row for row in reversed(failures)}
    return describe(list(unique.values())), metrics
