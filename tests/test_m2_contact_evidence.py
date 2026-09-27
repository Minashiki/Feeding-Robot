"""Analytic and adversarial checks for independent contact evidence."""
import numpy as np
import pytest
from feedingrobot.validation.m2.contact_checks import check_contacts, check_tool_balance


def sample(reverse=False):
    frame = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    local = np.array([2., 3., 4., .1, .2, .3])
    force, moment = frame.T @ local[:3], frame.T @ local[3:]
    point = np.array([.1, .2, .3])
    sign = -1 if reverse else 1
    return dict(tau_cmd=np.zeros((1, 7)), contact_dist=np.zeros(1), contact_offsets=np.array([0, 1]),
                contact_force=force[None], contact_force_contact=local[None, :3].copy(),
                contact_wrench_contact=local[None].copy(), contact_wrench_world=np.r_[force, moment][None], contact_frame=frame.reshape(1, 9),
                contact_pos=point[None], contact_geom1=np.array(['tool' if reverse else 'plate']),
                contact_geom2=np.array(['plate' if reverse else 'tool']),
                contact_geom1_id=np.array([1 if reverse else 0]), contact_geom2_id=np.array([0 if reverse else 1]),
                tcp_pos=np.zeros((2, 3)), external=np.zeros((1, 9)), ft_bias_wrench_tcp=np.zeros((1, 6)),
                external_body_com_before=np.zeros((1, 3)), applied_wrench_com=np.zeros((1, 6)),
                ft_tool_com=np.zeros((1, 1, 3)),
                ft_compensated_wrench_tcp=sign * np.r_[force, moment + np.cross(point, force)][None])


@pytest.mark.parametrize('reverse', [False, True])
def test_contact_frame_side_and_moment_arm(reverse):
    data = sample(reverse)
    assert check_contacts(data) == []
    assert check_tool_balance(data, [{'group': 'plate'}, {'group': 'spoon'}]) == []


@pytest.mark.parametrize('field', ['contact_frame', 'contact_force', 'contact_force_contact'])
def test_corrupt_contact_channel(field):
    data = sample()
    data[field][:] = 0 if field == 'contact_frame' else 1000
    assert check_contacts(data)


@pytest.mark.parametrize('change', ['side', 'moment_arm', 'both_forces', 'delete'])
def test_balanced_channel_tampering(change):
    data = sample()
    if change == 'side':
        data['contact_geom1_id'][:], data['contact_geom2_id'][:] = 1, 0
    elif change == 'moment_arm':
        data['contact_pos'][:] = 0
    elif change == 'both_forces':
        data['contact_wrench_world'][:, :3] *= 1000
        data['contact_force'] *= 1000
        data['contact_force_contact'] *= 1000
        data['contact_wrench_contact'][:, :3] *= 1000
    else:
        for key in tuple(data):
            if key.startswith('contact_') and key != 'contact_offsets':
                data[key] = data[key][:0]
        data['contact_offsets'][:] = 0
    assert check_contacts(data) == []
    assert check_tool_balance(data, [{'group': 'plate'}, {'group': 'spoon'}])


@pytest.mark.parametrize('phase,names,groups,allowed', [
    ('ACQUIRE', ('bowl_front', 'plate_bottom'), ('spoon', 'plate'), True),
    ('TRANSPORT', ('bowl_front', 'plate_bottom'), ('spoon', 'plate'), False),
    ('TRANSFER', ('bowl_front', 'jaw_lip'), ('spoon', 'mouth'), True),
    ('ACQUIRE', ('tool_handle', 'plate_bottom'), ('spoon', 'plate'), False),
    ('TRANSFER', ('bowl_front', None), ('spoon', 'arm'), False),
    ('TRANSFER', ('bowl_front', 'unknown'), ('spoon', 'unknown'), False),
    ('TRANSPORT', ('food_0', 'bowl_front'), ('food', 'spoon'), True),
])
@pytest.mark.parametrize('reverse', [False, True])
def test_independent_contact_permissions(phase, names, groups, allowed, reverse):
    from feedingrobot.validation.m2.contact_checks import recorded_contact_allowed
    registry = [{'id': i, 'name': name, 'group': group} for i, (name, group) in enumerate(zip(names, groups))]
    ids = [1, 0] if reverse else [0, 1]
    row = {'geom1_id': ids[0], 'geom2_id': ids[1], 'geom1': names[ids[0]], 'geom2': names[ids[1]]}
    assert recorded_contact_allowed(phase, row, {}, registry) == allowed


def test_applied_wrench_uses_sampled_com_not_old_world_point():
    data = sample()
    data['external'][0] = [1., 0., 0., 0., 0., 0., 0., 0., 0.]
    data['external_body_com_before'][0] = [0., .1, 0.]
    data['ft_tool_com'][0, 0] = [0., .1001, 0.]
    data['applied_wrench_com'][0] = [1., 0., 0., 0., 0., .1]
    data['ft_compensated_wrench_tcp'][0] += [1., 0., 0., 0., 0., -.0001]
    registry = [{'group': 'plate'}, {'group': 'spoon'}]
    assert check_tool_balance(data, registry) == []
    data['applied_wrench_com'][0, 5] = 0.
    assert check_tool_balance(data, registry) == [{'code': 'applied_wrench'}]
