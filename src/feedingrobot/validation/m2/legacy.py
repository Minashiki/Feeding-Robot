"""Read-only fixes-v6 boundary and binding of generator-produced counterexamples."""
from pathlib import Path
import json

from feedingrobot.validation.m2.provenance import digest, identity, read


def rebuild_progress_events(data, config, dt):
    """Current blocked contract on legacy physical arrays; old packages keep their rule."""
    import numpy as np
    from feedingrobot.controllers.contracts import GEAR_OF
    from feedingrobot.controllers.so3 import so3_log
    n = len(data['command'])
    events = np.zeros(n, dtype=bool)
    for name, shape in [('block_progress', (n, 6)), ('block_correction_pos', (n, 3))]:
        if name not in data or data[name].shape != shape or not np.all(np.isfinite(data[name])):
            return events, [name]
    spec = config['blocked']
    window = int(np.ceil(spec['progress_window_s'] / dt - 1e-12))
    start = previous_direction = None
    previous_gear = GEAR_OF[str(data['phase'][0])]
    timer = deficit = dev_time = 0.
    errors = set()
    for k in range(n):
        gear = GEAR_OF[str(data['phase'][k])]
        command = data['command'][k, :3]
        magnitude = np.linalg.norm(command)
        pos = data['tcp_pos'][k]
        ref = data['p_ref_before'][k].copy()
        average, span = np.zeros(3), 0.
        correction = np.zeros(3)
        held = bool(data['pause_applied'][k]) or str(data['execution'][k]) not in {'run', 'zero'}
        if gear != previous_gear:
            start = previous_direction = None
            timer = deficit = 0.
        previous_gear = gear
        if held or magnitude <= 1e-6:
            start = previous_direction = None
            timer = deficit = 0.
        else:
            direction = command / magnitude
            if start is None or (previous_direction is not None and direction @ previous_direction <= 1e-12):
                start, timer, deficit = k, 0., 0.
            previous_direction = direction
            left = max(start, k - window)
            span = (k - left) * dt
            if span > 1e-12:
                average = (pos - data['tcp_pos'][left]) / span
            force = data['force_used'][k]
            normal = -force / max(np.linalg.norm(force), 1e-30)
            requested = command @ normal
            stalled = (span >= spec['progress_window_s'] - 1e-12 and requested > 1e-6
                       and average @ normal <= spec['progress_ratio'] * requested)
            eligible = -force @ direction > spec['force_n'] and (ref - pos) @ direction > spec['error_m']
            timer = timer + dt if eligible and stalled else 0.
            deficit = deficit + (requested - average @ normal) * dt if eligible and stalled else 0.
            confirmed_distance = timer >= spec['deficit_confirm_s'] - 1e-15 and deficit >= spec['deficit_m'] - 1e-15
            if timer >= spec['confirm_s'] - 1e-15 or confirmed_distance:
                events[k] = True
                correction = -normal * max(float((ref - pos) @ normal) - spec['error_m'], 0.)
                ref += correction
                if np.linalg.norm(correction) > 1e-15:
                    start = previous_direction = None
                    average, span, timer, deficit = np.zeros(3), 0., 0., 0.
        if not held:
            limit = config['gears'][gear]
            capped = (np.linalg.norm(ref - pos) >= limit['pos_dev_m'] - 1e-6
                      or np.linalg.norm(so3_log(data['r_ref_before'][k] @ data['tcp_rot'][k].T)) >= limit['rot_dev_rad'] - 1e-5)
            dev_time = dev_time + dt if capped else 0.
            if dev_time >= spec['dev_s'] - 1e-15:
                start = previous_direction = None
                average, span, timer, deficit, dev_time = np.zeros(3), 0., 0., 0., 0.
        if np.linalg.norm(data['reference_correction_pos'][k]) > 1e-15:
            start = previous_direction = None
            average, span, timer, deficit = np.zeros(3), 0., 0., 0.
        if not np.allclose(data['block_progress'][k], np.r_[average, span, timer, deficit], atol=1e-8, rtol=0):
            errors.add('block_progress')
        if not np.allclose(data['block_correction_pos'][k], correction, atol=1e-8, rtol=0):
            errors.add('block_correction_pos')
    return events, sorted(errors)


def record_identity(run):
    run = Path(run)
    return identity({str(p.relative_to(run)): digest(p)
                     for p in sorted(run.rglob('*')) if p.is_file()
                     and p.name not in {'adversarial_binding.json', 'adversarial_binding.json.tmp'}})


def seal_execution(run):
    """Called by the generator only after real counterexample execution completes."""
    run = Path(run)
    execution = read(run/'adversarial_execution.json')
    from feedingrobot.controllers.v3spec import check_second_stage
    errors = []
    check_second_stage(execution, errors)
    if errors or execution.get('run_id') != run.name + '-adversarial':
        raise ValueError('legacy counterexample execution is incomplete')
    payload = {'source': record_identity(run),
               'results': digest(run/'adversarial_results.json'),
               'execution': digest(run/'adversarial_execution.json')}
    temporary = run/'adversarial_binding.json.tmp'
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(run/'adversarial_binding.json')


def verify_finished_report(run, scope, check_workspace):
    from feedingrobot.controllers.v3spec import verify_finished_report as verify
    result = verify(run, scope, check_workspace, rerun_adversarial=False)
    if scope != 'fixes-v6':
        return result
    run = Path(run)
    try:
        binding = read(run/'adversarial_binding.json')
        expected = {'source': record_identity(run),
                    'results': digest(run/'adversarial_results.json'),
                    'execution': digest(run/'adversarial_execution.json')}
        execution = read(run/'adversarial_execution.json')
        valid = binding == expected and execution.get('run_id') == run.name + '-adversarial'
    except (OSError, ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        result['reasons'].append({'code': 'legacy_execution_binding'})
        result.update(scope_passed=False, fixes_passed=False, evidence_valid=False, m3_ready=False)
    return result
