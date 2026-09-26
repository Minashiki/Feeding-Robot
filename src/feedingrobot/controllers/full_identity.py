"""Full-matrix provenance checks; reconstruction never advances physics."""
import copy
import hashlib
import json

import mujoco
import numpy as np

from feedingrobot.controllers.full_spec import VARIANTS
from feedingrobot.controllers.guard import geometry_registry
from feedingrobot.sim.model import load_config, repo_root
from feedingrobot.sim.scene import FeedingScene


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def clean_config(config):
    return {k: copy.deepcopy(v) for k, v in config.items() if k != '_config_path'}


def expected_config(case, seed, base):
    cfg = clean_config(base)
    if case.fixture:
        cfg['press_test'] = True
    if case.family in {'stiffness', 'press'}:
        cfg['gears']['FREE']['kp'] = case.stiffness
    cfg['delay_s'] = case.delay
    cfg['noise']['enable'] = case.noise
    cfg['noise']['seed'] = seed*10 + (int(case.event[-1]) if case.event.startswith('sensor') else 0)
    return cfg


def solver_options(model):
    return {'dt': float(model.opt.timestep), 'iterations': int(model.opt.iterations),
            'tolerance': float(model.opt.tolerance), 'solver': int(model.opt.solver),
            'integrator': int(model.opt.integrator)}


def check_identity(case, seed, variant, data, meta, base=None):
    failures = []
    def check(ok, code):
        if not ok:
            failures.append({'code': code})
    try:
        if base is None:
            base = json.loads((repo_root()/'configs/m2_controller.json').read_text())
        effective = expected_config(case, seed, base)
        check(clean_config(meta['effective_controller']) == effective, 'effective_config')
        execution = meta['execution']
        check(execution['case'] == case.manifest(seed, variant), 'execution_identity')
        check(str(data['execution_id'].item()) == execution['id'] and bool(execution['id']), 'execution_id')
        check(str(data['case_identity'].item()) == identity(case.manifest(seed, variant)), 'trace_identity')
        check(str(data['config_identity'].item()) == identity(effective), 'config_identity')
        scene_cfg = load_config('configs/m1_scene.json')
        if case.fixture:
            scene_cfg['model'] = 'assets/tests/m2_full_wall.xml' if case.family == 'wall' else 'assets/tests/m2_full_surface.xml'
        scene = FeedingScene(scene_cfg)
        dt, iterations, tolerance = VARIANTS[variant]
        options = {**solver_options(scene.model), 'dt': dt, 'iterations': iterations, 'tolerance': tolerance}
        check(execution['solver_before'] == options == execution['solver_after'], 'solver_identity')
        check(str(data['solver_identity'].item()) == identity(options), 'solver_identity')
        registry = geometry_registry(scene.model)
        check(meta['geometry_registry'] == registry, 'geometry_registry')
        source = meta['initial_source']
        check(source['case'] == case.key and source['seed'] == seed, 'initial_source')
        check(str(data['initial_source_identity'].item()) == identity(source), 'initial_source')
        settled = source['settled']
        for field in ('qpos', 'qvel', 'tau'):
            check(np.array_equal(data['initial_'+field], settled[field]), 'initial_state')
        ix = scene.index
        check(np.array_equal(data['q'][0], np.asarray(settled['qpos'])[ix.arm_qpos_adr]), 'initial_state')
        check(np.array_equal(data['physical_dq'][0], np.asarray(settled['qvel'])[ix.arm_dof_adr]), 'initial_state')
        preset = 'food_on_spoon' if case.family == 'carry' or case.context == 'food' else 'near_mouth' if case.family == 'mouth' else 'food_on_plate'
        check(source['preset'] == preset, 'initial_source')
        reset = np.asarray(source['reset_qpos'])
        pre = np.asarray(source['pre_settle_qpos'])
        check(reset.shape == pre.shape == (scene.model.nq,), 'initial_shape')
        if reset.shape != (scene.model.nq,) or pre.shape != reset.shape:
            return failures
        q = np.asarray(scene_cfg['presets'][preset]['qpos'])
        check(np.array_equal(reset[ix.arm_qpos_adr], q), 'reset_arm')
        # Independent reconstruction of the declared food RNG stream (M1 allows eight attempts).
        rng = np.random.default_rng(seed)
        if preset == 'near_mouth':
            candidates = [np.array([.45, -.18, .03])]
        elif preset == 'food_on_plate':
            candidates = [np.r_[np.array([.45, -.18])+rng.uniform(-.025, .025, 2), .03] for _ in range(8)]
        else:
            scene.data.qpos[:] = reset
            mujoco.mj_forward(scene.model, scene.data)
            tcp = ix.site_ids['tcp']
            p, r = scene.data.site_xpos[tcp], scene.data.site_xmat[tcp].reshape(3, 3)
            candidates = [p+r@np.r_[rng.uniform(-.002, .002, 2), .007] for _ in range(8)]
        food = reset[ix.food_qpos_adr:ix.food_qpos_adr+3]
        check(any(np.allclose(food, candidate, atol=1e-12, rtol=0) for candidate in candidates), 'seed_initialization')
        expected_q = q.copy()
        if case.event == 'singularity':
            from feedingrobot.controllers.v6checks import SINGULAR_Q
            expected_q = SINGULAR_Q
        elif case.family not in {'carry', 'plate', 'mouth'} and case.context != 'food':
            expected_q = np.asarray(scene_cfg['q_torque_poses'][case.pose % 3]).copy()
            if case.pose >= 3:
                expected_q[6] += np.pi/2
            expected_q += np.random.default_rng(seed).uniform(-.01, .01, 7)
        if case.family not in {'plate', 'mouth'}:
            check(np.allclose(pre[ix.arm_qpos_adr], expected_q, atol=1e-12, rtol=0), 'seed_initialization')
        else:
            scene.data.qpos[:] = pre
            mujoco.mj_forward(scene.model, scene.data)
            target = [.49, -.22, .030] if case.family == 'plate' else [.518, .12, .336]
            check(np.allclose(scene.data.site_xpos[ix.site_ids['tcp']], target, atol=1e-8, rtol=0), 'reset_ik')
        other = np.ones(scene.model.nq, dtype=bool)
        other[ix.arm_qpos_adr] = False
        check(np.array_equal(pre[other], reset[other]), 'reset_nonarm')
    except (KeyError, ValueError, TypeError, IndexError, AttributeError) as error:
        failures.append({'code': 'identity_structure', 'detail': str(error)})
    return failures


def check_receipt(case, seed, variant, data, meta, receipt, npz_hash, meta_hash):
    expected = {'case': case.manifest(seed, variant), 'execution': meta['execution'],
                'initial_source_identity': str(data['initial_source_identity'].item()),
                'config_identity': str(data['config_identity'].item()),
                'npz_sha256': npz_hash, 'json_sha256': meta_hash, 'complete': True}
    return [] if receipt == expected else [{'code': 'execution_receipt'}]
