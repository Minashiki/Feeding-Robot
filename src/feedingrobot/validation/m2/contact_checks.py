"""Independent contact evidence reconstruction, in world coordinates."""
import numpy as np


def check_contacts(data):
    n = len(data.get('tau_cmd', []))
    count = len(data.get('contact_dist', []))
    shapes = {'contact_offsets': (n + 1,), 'contact_force': (count, 3),
              'contact_force_contact': (count, 3), 'contact_frame': (count, 9),
              'contact_pos': (count, 3), 'contact_geom1': (count,),
              'contact_geom2': (count,), 'contact_geom1_id': (count,),
              'contact_geom2_id': (count,), 'contact_wrench_contact': (count, 6),
              'contact_wrench_world': (count, 6)}
    failures = [{'code': 'shape:' + key} for key, shape in shapes.items()
                if key not in data or data[key].shape != shape]
    if failures:
        return failures
    offsets = data['contact_offsets']
    if (offsets.dtype.kind not in 'iu' or offsets[0] != 0 or offsets[-1] != count
            or np.any(np.diff(offsets) < 0)):
        return [{'code': 'contact_offsets'}]
    for key in ('contact_geom1_id', 'contact_geom2_id'):
        if data[key].dtype.kind not in 'iu':
            failures.append({'code': 'dtype:' + key})
    for key in ('contact_dist', 'contact_frame', 'contact_force', 'contact_force_contact', 'contact_pos', 'contact_wrench_contact', 'contact_wrench_world'):
        if data[key].dtype.kind not in 'fi':
            failures.append({'code': 'dtype:' + key})
    if failures:
        return failures
    frame = data['contact_frame'].reshape(-1, 3, 3)
    checks = {
        'contact_frame': np.all(np.isfinite(frame)) and np.allclose(
            frame @ frame.transpose(0, 2, 1), np.eye(3), atol=1e-9, rtol=0)
            and np.allclose(np.linalg.det(frame), 1., atol=1e-9, rtol=0),
        'contact_force_transform': np.allclose(
            np.einsum('nji,nj->ni', frame, data['contact_force_contact']),
            data['contact_force'], atol=1e-9, rtol=1e-7),
    }
    failures.extend({'code': code} for code, ok in checks.items() if not ok)
    return failures


def check_tool_balance(data, registry):
    """Sum only contacts crossing the tool boundary, with moments about TCP."""
    count = len(data['contact_dist'])
    wrench = data.get('contact_wrench_contact')
    if wrench is None or wrench.shape != (count, 6):
        return [{'code': 'shape:contact_wrench_contact'}]
    if not np.allclose(wrench[:, :3], data['contact_force_contact'], atol=1e-9, rtol=0):
        return [{'code': 'contact_wrench_force'}]
    frame = data['contact_frame'].reshape(-1, 3, 3)
    world = np.concatenate([np.einsum('nji,nj->ni', frame, wrench[:, :3]),
                            np.einsum('nji,nj->ni', frame, wrench[:, 3:])], axis=1)
    if 'contact_wrench_world' not in data or data['contact_wrench_world'].shape != (count, 6):
        return [{'code': 'shape:contact_wrench_world'}]
    if not np.allclose(world, data['contact_wrench_world'], atol=1e-9, rtol=1e-7):
        return [{'code': 'contact_wrench_transform'}]
    try:
        net = tool_contact_wrench(data, registry)
    except ValueError:
        return [{'code': 'contact_geom_identity'}]
    external = data['external']
    expected_applied = external[:, :6].copy()
    expected_applied[:, 3:] += np.cross(external[:, 6:] - data['external_body_com_before'], external[:, :3])
    applied = data['applied_wrench_com']
    if not np.allclose(applied, expected_applied, atol=1e-9, rtol=0):
        return [{'code': 'applied_wrench'}]
    net[:, :3] += applied[:, :3]
    net[:, 3:] += applied[:, 3:] + np.cross(data['ft_tool_com'][:, 0] - data['tcp_pos'][1:], applied[:, :3])
    residual = data['ft_compensated_wrench_tcp'] + data['ft_bias_wrench_tcp'] - net
    valid = np.all(np.isfinite(residual), axis=1)
    failures = []
    # Exact, simultaneous MuJoCo channels: no empirical tolerance expansion.
    for part, columns, limit in [('force', slice(0, 3), 1e-7), ('moment', slice(3, 6), 1e-7)]:
        error = np.linalg.norm(residual[valid, columns], axis=1)
        if len(error) and np.max(error) > limit:
            failures.append({'code': 'contact_balance:' + part, 'observed': float(np.max(error)), 'limit': limit})
    return failures


def tool_contact_wrench(data, registry):
    wrench = data['contact_wrench_contact']
    frame = data['contact_frame'].reshape(-1, 3, 3)
    world_force = np.einsum('nji,nj->ni', frame, wrench[:, :3])
    world_moment = np.einsum('nji,nj->ni', frame, wrench[:, 3:])
    net = np.zeros((len(data['tau_cmd']), 6))
    for k, (start, end) in enumerate(zip(data['contact_offsets'][:-1], data['contact_offsets'][1:])):
        tcp = data['tcp_pos'][k + 1]
        for j in range(start, end):
            ids = [int(data['contact_geom1_id'][j]), int(data['contact_geom2_id'][j])]
            if any(gid < 0 or gid >= len(registry) for gid in ids):
                raise ValueError('invalid contact geometry identity')
            tool = [registry[gid]['group'] == 'spoon' for gid in ids]
            sign = int(tool[1]) - int(tool[0])
            net[k, :3] += sign * world_force[j]
            net[k, 3:] += sign * (world_moment[j] + np.cross(data['contact_pos'][j] - tcp, world_force[j]))
    return net


def recorded_contact_allowed(phase, row, context, registry):
    """Offline permission predicate from frozen ownership and allowed surfaces."""
    from feedingrobot.controllers.contracts import BOWL_GEOMS, PLATE_GEOMS, MOUTH_GEOMS, MOUTH_PHASES
    ids = [row.get('geom1_id'), row.get('geom2_id')]
    if any(not isinstance(gid, (int, np.integer)) or gid < 0 or gid >= len(registry) for gid in ids):
        return False
    owners = [registry[int(gid)] for gid in ids]
    if any(row.get('geom' + str(side)) != owner['name'] for side, owner in enumerate(owners, 1)):
        return False
    names = {owner['name'] or '' for owner in owners}
    groups = {owner['group'] for owner in owners}
    if any(name.startswith(('spring_', 'test_')) for name in names):
        return bool(context.get('press_test') and 'spring_pad' in names and names.intersection(BOWL_GEOMS))
    if 'unknown' in groups:
        return False
    if groups in ({'food', 'spoon'}, {'food', 'plate'}):
        return True
    if groups == {'food', 'mouth'}:
        return phase in MOUTH_PHASES or bool(context.get('mouth_case'))
    if 'food' in groups and groups <= {'food', 'table', 'floor', 'plate'}:
        return True
    if not groups.intersection({'arm', 'spoon'}):
        return True
    if groups == {'spoon', 'plate'}:
        return phase == 'ACQUIRE' and bool(names.intersection(BOWL_GEOMS)) and bool(names.intersection(PLATE_GEOMS))
    if groups == {'spoon', 'mouth'}:
        return phase in MOUTH_PHASES and bool(names.intersection(BOWL_GEOMS)) and bool(names.intersection(MOUTH_GEOMS))
    return False
