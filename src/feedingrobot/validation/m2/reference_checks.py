"""Independent reference and gain reconstruction."""
import numpy as np
from feedingrobot.controllers.contracts import GEAR_OF

from feedingrobot.validation.m2.spec import BLEND_S, BLEND_TIME_TOL_S, GEAR_K, GEAR_D, GEAR_VW

def rebuild_gain_path(phase, dt: float, initial_phase: str = "TRANSPORT") -> dict:
    """Expected gains from the case's initial phase and the logged phase. Logged gears are not inputs."""
    phase = np.asarray(phase).astype(str)
    n = int(phase.shape[0])
    init = GEAR_OF[str(initial_phase)]
    expected_k = GEAR_K[init].copy()
    expected_d = GEAR_D[init].copy()
    completed = init
    desired_prev = init
    in_transition = False
    start_k = expected_k.copy()
    start_d = expected_d.copy()
    elapsed = 0.0
    out_k = np.zeros((n, 6))
    out_d = np.zeros((n, 6))
    out_active = np.empty(n, dtype=object)
    out_target = np.empty(n, dtype=object)
    out_trans = np.zeros(n, dtype=int)
    v_lim = np.zeros(n)
    w_lim = np.zeros(n)
    acquire_done = None
    for k in range(n):
        name = str(phase[k])
        if name not in GEAR_OF:
            return {"error": name, "index": k}
        desired = GEAR_OF[name]
        if desired == "STOP":
            expected_k = np.zeros(6)
            expected_d = np.zeros(6)
            completed = "STOP"
            desired_prev = "STOP"
            in_transition = False
            elapsed = BLEND_S
            out_k[k] = expected_k
            out_d[k] = expected_d
            out_active[k] = completed
            out_target[k] = desired
            out_trans[k] = 0
            v_lim[k], w_lim[k] = GEAR_VW["STOP"]
            continue
        if desired != desired_prev:
            start_k = expected_k.copy()
            start_d = expected_d.copy()
            elapsed = 0.0
            in_transition = True
        if in_transition:
            elapsed = min(BLEND_S, elapsed + float(dt))
            if elapsed >= BLEND_S - BLEND_TIME_TOL_S:
                expected_k = GEAR_K[desired].copy()
                expected_d = GEAR_D[desired].copy()
                completed = desired
                in_transition = False
                if desired == "ACQUIRE" and acquire_done is None:
                    acquire_done = k
            else:
                alpha = elapsed / BLEND_S
                expected_k = (1.0 - alpha) * start_k + alpha * GEAR_K[desired]
                expected_d = (1.0 - alpha) * start_d + alpha * GEAR_D[desired]
        else:
            expected_k = GEAR_K[desired].copy()
            expected_d = GEAR_D[desired].copy()
            completed = desired
        out_k[k] = expected_k
        out_d[k] = expected_d
        out_active[k] = completed
        out_target[k] = desired
        out_trans[k] = 1 if in_transition else 0
        left = GEAR_VW[completed]
        right = GEAR_VW[desired]
        v_lim[k] = min(left[0], right[0]) if in_transition else left[0]
        w_lim[k] = min(left[1], right[1]) if in_transition else left[1]
        desired_prev = desired
    return {
        "K": out_k,
        "D": out_d,
        "active": out_active,
        "target": out_target,
        "trans": out_trans,
        "v_lim": v_lim,
        "w_lim": w_lim,
        "acquire_done": acquire_done,
    }



def check_analytic_motion(case, data):
    """Measure the prescribed two-cycle motion independently of reported references."""
    if case.family not in {'circle', 'loaded_circle', 'sine'}:
        return []
    t = data['t'][1:]
    moving = (t >= 1.) & (t <= 9.)
    angle = .5 * np.pi * np.clip(t - 1., 0., 8.)
    failures = []
    def check_error(error, code, rms_limit, max_limit):
        rms = float(np.sqrt(np.mean(error**2)))
        maximum = float(np.max(error))
        if rms > rms_limit or maximum > max_limit:
            metric = 'rms' if rms > rms_limit else 'maximum'
            failures.append({'code': code, 'metric': metric,
                             'observed': rms if metric == 'rms' else maximum,
                             'limit': rms_limit if metric == 'rms' else max_limit,
                             'rms': rms, 'maximum': maximum,
                             'rms_limit': rms_limit, 'max_limit': max_limit})
    if case.family in {'circle', 'loaded_circle'}:
        target = data['origin'] + np.column_stack((
            .01 * (np.cos(angle) - 1.), .01 * np.sin(angle), np.zeros(len(t))))
        error = np.linalg.norm(data['p_ref'] - target, axis=1)[moving]
        check_error(error, 'reference:circle_target', .002, .005)
        actual_target = target.copy()
        if case.family == 'loaded_circle':
            actual_target[t > 1., case.axis] += case.sign * case.load / GEAR_K['FREE'][case.axis]
        error = np.linalg.norm(data['tcp_pos'][1:] - actual_target, axis=1)[moving]
        check_error(error, 'motion:' + case.family + '_target', .002, .005)
        if case.family == 'loaded_circle':
            from feedingrobot.controllers.so3 import so3_log
            error = np.array([np.linalg.norm(so3_log(r @ data['rotation'].T))
                              for r in data['tcp_mat'][1:][moving]])
            check_error(error, 'motion:loaded_circle_rotation', np.deg2rad(1), np.deg2rad(3))
    else:
        from feedingrobot.controllers.so3 import so3_log
        target = np.pi / 60 * np.sin(angle)
        for field, code in [('r_ref', 'reference:sine_target'), ('tcp_mat', 'motion:sine_target')]:
            rotations = data[field] if field == 'r_ref' else data[field][1:]
            displacement = np.array([so3_log(r @ data['rotation'].T) for r in rotations])
            expected = np.zeros_like(displacement)
            expected[:, case.axis] = target
            error = np.linalg.norm(displacement - expected, axis=1)[moving]
            check_error(error, code, np.deg2rad(1), np.deg2rad(3))
    return failures


def check_reference_path(data, config, dt):
    """Return failures and independently reconstructed first-event times at t_k."""
    from feedingrobot.controllers.so3 import so3_exp, so3_log
    def bound(value, limit):
        norm = np.linalg.norm(value)
        return value * min(1., limit / max(norm, 1e-30))
    failures = {}
    events = {}
    def equal(field, k, expected, tolerance=1e-8):
        if not np.allclose(data[field][k], expected, atol=tolerance, rtol=0):
            failures.setdefault(field, {'code': 'reference:' + field, 'tick': k,
                                       'time': float(data['t'][k])})
    gain = rebuild_gain_path(data['phase_k'], dt)
    if 'error' in gain:
        return [{'code': 'reference:phase'}], events
    velocity = np.zeros(6)
    block_time = dev_time = 0.
    blocked = False
    previous_target = 'FREE'
    stopped = False
    origin = data['origin']
    block_deficit = 0.
    progress_start = None
    progress_direction = None
    block_spec = config['blocked']
    window_steps = int(np.ceil(block_spec['progress_window_s'] / dt - 1e-12))
    for k in range(len(data['tau_cmd'])):
        p = data['p_ref'][k - 1].copy() if k else origin.copy()
        r = data['r_ref'][k - 1].copy() if k else data['rotation'].copy()
        equal('p_ref_before', k, p)
        equal('r_ref_before', k, r)
        actual_p, actual_r = data['tcp_pos'][k], data['tcp_mat'][k]
        stop = data['execution'][k] == 'stop'
        power = data['execution'][k] == 'power_on'
        target = GEAR_OF[str(data['phase_k'][k])]
        specs = [config['gears'][str(gain['active'][k])], config['gears'][target]]
        limits = {key: min(spec[key] for spec in specs) for key in ('v', 'w', 'a', 'alpha', 'pos_dev_m', 'rot_dev_rad')}
        if stop:
            limits = dict(config['gears']['STOP'])
        changed = target != previous_target
        if changed:
            progress_start = progress_direction = None
            block_time = block_deficit = 0.
        new = config['gears'][target]
        gain_anchor = changed and (np.linalg.norm(p - actual_p) > new['pos_dev_m'] or np.linalg.norm(so3_log(r @ actual_r.T)) > new['rot_dev_rad'])
        if (stop and not stopped) or (not stop and gain_anchor):
            p, r, velocity = actual_p.copy(), actual_r.copy(), np.zeros(6)
            block_time = block_deficit = dev_time = 0.
            progress_start = progress_direction = None
        stopped |= stop
        previous_target = target
        scale = float(np.clip((data['rho'][k] - .01) / .04, 0., 1.))
        equal('scale', k, scale)
        command_time, valid_until = data['command_times'][k]
        present = command_time >= 0
        if present and (command_time > data['t'][k] + 1e-9 or valid_until <= command_time or valid_until - command_time > .1 + 1e-9):
            failures.setdefault('command_times', {'code': 'reference:command_times', 'tick': k})
        twist = data['supplied'][k].copy()
        if power or stop or not present or data['t'][k] >= valid_until - 1e-12:
            twist[:] = 0.
        equal('twist_command', k, twist)
        saturated = k > 0 and np.any(np.abs(data['tau_cmd'][k - 1] - data['tau_raw'][k - 1]) > 1e-8)
        integrate = not (power or stop or scale <= 0 or saturated)
        command_norm = np.linalg.norm(twist[:3])
        average = np.zeros(3)
        span = 0.
        if not integrate or command_norm <= 1e-6:
            progress_start = progress_direction = None
            block_time = block_deficit = 0.
        else:
            direction = twist[:3] / command_norm
            if progress_start is None or (progress_direction is not None and direction @ progress_direction <= 1e-12):
                progress_start = k
                block_time = block_deficit = 0.
            progress_direction = direction
            start = max(progress_start, k - window_steps)
            span = (k - start) * dt
            if span > 1e-12:
                average = (actual_p - data['tcp_pos'][start]) / span
        candidate = np.zeros(6)
        reanchor = False
        blocked_now = False
        block_correction = np.zeros(3)
        if integrate:
            for sl, speed, acceleration in [(slice(0, 3), 'v', 'a'), (slice(3, 6), 'w', 'alpha')]:
                history = bound(velocity[sl], limits[speed] * scale)
                requested = bound(twist[sl], limits[speed] * scale)
                candidate[sl] = bound(history + bound(requested - history, limits[acceleration] * dt), limits[speed] * scale)
            norm = np.linalg.norm(twist[:3])
            force = data['ft_compensated_wrench_tcp'][k - 1, :3] if k else data['initial_compensated'][:3]
            direction = twist[:3] / max(norm, 1e-30)
            normal = -force / max(np.linalg.norm(force), 1e-30)
            requested = float(twist[:3] @ normal)
            stalled = (span >= block_spec['progress_window_s'] - 1e-12 and requested > 1e-6
                       and average @ normal <= block_spec['progress_ratio'] * requested)
            opposing = (norm > 1e-6 and -force @ direction > block_spec['force_n']
                        and (p - actual_p) @ direction > block_spec['error_m'] and stalled)
            block_time = block_time + dt if opposing else 0.
            block_deficit = block_deficit + (requested - average @ normal) * dt if opposing else 0.
            deficit_confirmed = (block_time >= block_spec['deficit_confirm_s'] - 1e-15
                                 and block_deficit >= block_spec['deficit_m'] - 1e-15)
            if block_time >= block_spec['confirm_s'] - 1e-15 or deficit_confirmed:
                events.setdefault('blocked_confirm', float(data['t'][k]))
                candidate[:3] -= normal * max(float(candidate[:3] @ normal), 0.)
                block_correction = -normal * max(float((p - actual_p) @ normal) - block_spec['error_m'], 0.)
                p += block_correction
                if np.linalg.norm(block_correction) > 1e-15:
                    reanchor = True
                    progress_start = progress_direction = None
                    average, span, block_time, block_deficit = np.zeros(3), 0., 0., 0.
                blocked = blocked_now = True
            cap = np.linalg.norm(p - actual_p) >= limits['pos_dev_m'] - 1e-6 or np.linalg.norm(so3_log(r @ actual_r.T)) >= limits['rot_dev_rad'] - 1e-5
            dev_time = dev_time + dt if cap else 0.
            if dev_time >= .1 - 1e-15:
                p, r = actual_p.copy(), actual_r.copy()
                candidate[:] = 0.
                blocked = reanchor = True
                blocked_now = True
                block_time = block_deficit = dev_time = 0.
                progress_start = progress_direction = None
                average, span = np.zeros(3), 0.
            box = config['workspace_box_m']
            for axis in range(3):
                if abs(p[axis] + candidate[axis] * dt - origin[axis]) > box and abs(candidate[axis]) > 1e-15:
                    candidate[axis] = np.sign(candidate[axis]) * max(box - abs(p[axis] - origin[axis]), 0.) / dt
        equal('candidate_twist', k, candidate)
        limited = candidate * data['beta'][k]
        equal('twist_limited', k, limited)
        before_p, before_r = p.copy(), r.copy()
        if integrate:
            p += limited[:3] * dt
            r = so3_exp(limited[3:] * dt) @ r
        uncorrected = p.copy()
        p = actual_p + bound(p - actual_p, limits['pos_dev_m'])
        correction = p - uncorrected
        rotation_error = so3_log(r @ actual_r.T)
        angle = np.linalg.norm(rotation_error)
        rotation_correction = max(angle - limits['rot_dev_rad'], 0.)
        if rotation_correction > 1e-12:
            r = so3_exp(bound(rotation_error, limits['rot_dev_rad'])) @ actual_r
        equal('reference_correction_pos', k, correction)
        equal('reference_correction_rot', k, rotation_correction)
        reanchor |= np.linalg.norm(correction) > 1e-15 or rotation_correction > 1e-15
        if np.linalg.norm(correction) > 1e-15 or rotation_correction > 1e-15:
            progress_start = progress_direction = None
            average, span, block_time, block_deficit = np.zeros(3), 0., 0., 0.
        velocity = np.zeros(6) if reanchor or not integrate else limited
        v_ref = np.zeros(6) if reanchor or not integrate else np.r_[(p - before_p) / dt, so3_log(r @ before_r.T) / dt]
        equal('v_ref', k, v_ref)
        equal('v_hist', k, velocity)
        equal('p_ref', k, p)
        equal('r_ref', k, r)
        equal('blocked', k, blocked)
        equal('blocked_now', k, blocked_now)
        equal('block_progress', k, np.r_[average, span, block_time, block_deficit])
        equal('block_correction_pos', k, block_correction)
    return list(failures.values()), events


def expected_stimulus(case, times):
    """Independent analytic schedule, not imported from the command generator."""
    published = np.floor((times+1e-10)/.05)*.05
    def displacement(t):
        out = np.zeros((len(t),6))
        def blend(start, duration):
            u = np.clip((t-start)/duration,0,1)
            return u*u*u*(10-15*u+6*u*u)
        if case.family == "step":
            out[:,case.axis] = case.sign*(.01 if case.axis<3 else np.pi/36)*blend(1,1)
        elif case.family in {"circle","loaded_circle"}:
            a = .5*np.pi*np.clip(t-1,0,8)
            out[:,0],out[:,1] = .01*(np.cos(a)-1), .01*np.sin(a)
        elif case.family == "sine":
            out[:,3+case.axis] = np.pi/60*np.sin(.5*np.pi*np.clip(t-1,0,8))
        elif case.family in {"spring","press"}:
            out[:,1] = -np.clip((t-1)*.002,0,.0065)+np.clip((t-6.25)*.002,0,.0065)
        elif case.family == "carry": out[:,0] = .01*(blend(1,1)-blend(4,1))
        elif case.family == "plate": out[:,2] = -.008*(blend(1,12)-blend(15,12))
        elif case.family == "mouth": out[:,0] = .006*(blend(1,12)-blend(15,12))
        return out
    supplied = (displacement(published+.05)-displacement(published))/.05
    external = np.zeros_like(supplied)
    phase = np.full(len(times),"TRANSPORT",dtype="U16")
    if case.family == "wall": supplied[:,1] = np.where((published>=1)&(published<3),-.02,0)
    if case.family == "stop":
        end = 1.42 if case.event in {"STOP","RECOVER"} else 1.42+(times[1]-times[0])*(.5 if case.event == "expired" else 1.5)
        supplied[:,case.axis] = np.where((published>=.8)&(times<end),case.sign*(.04 if case.axis<3 else .2),0)
        if case.event == "singularity":supplied[:]=0
        if case.event in {"STOP","RECOVER"}: phase[times>=1.42-1e-10]=case.event
        if case.event == "wrench": external[np.abs(times-1.42)<1e-10,0] = 8.1
    if case.family in {"static","loaded_circle","stiffness"}:
        external[times>=1.-1e-10,case.axis] = case.sign*case.load
    if case.family == "pulse":
        external[(times>=1.5-1e-10)&(times<1.65-1e-10),case.axis] = case.sign*(1. if case.axis<3 else .02)
    if case.family == "plate": phase[(published>=.5)&(published<28.)]="ACQUIRE"
    if case.family == "mouth": phase[:]="APPROACH"
    return supplied, external, phase
