"""Full-matrix provenance checks; reconstruction never advances physics."""
import copy
import hashlib
import json
from pathlib import Path
from collections import Counter

import mujoco
import numpy as np

from feedingrobot.validation.m2.spec import VARIANTS
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
            from feedingrobot.validation.m2.spec import SINGULAR_Q
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
        if case.family == 'plate':
            selected = None
            scene.data.qpos[:] = pre
            for attempt, candidate in enumerate(candidates):
                scene.data.qpos[ix.food_qpos_adr:ix.food_qpos_adr+3] = candidate
                mujoco.mj_forward(scene.model, scene.data)
                legal = True
                for contact in scene.data.contact[:scene.data.ncon]:
                    groups = {registry[int(contact.geom1)]['group'], registry[int(contact.geom2)]['group']}
                    if contact.dist < -.001 or (groups & {'arm', 'spoon'} and groups & {'table', 'plate', 'mouth'}):
                        legal = False
                        break
                if legal:
                    selected = attempt
                    break
            check(selected is not None and type(source.get('food_placement_attempt')) is int
                  and source['food_placement_attempt'] == selected, 'food_placement_attempt')
            if selected is not None:
                check(np.allclose(pre[ix.food_qpos_adr:ix.food_qpos_adr+3], candidates[selected], atol=1e-12, rtol=0), 'food_placement')
            other[ix.food_qpos_adr:ix.food_qpos_adr+3] = False
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


CONFIG_PATHS = ('configs/m1_scene.json', 'configs/m2_controller.json', 'configs/m2_acceptance.json')


def required_nodes():
    nodes = json.loads((repo_root()/'tests/m2_full_nodeids.json').read_text())
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError('invalid frozen test list')
    return nodes


def check_execution(execution, nodes, run_id):
    reasons = []
    if execution.get('run_id') != run_id or execution.get('exitstatus') != 0:
        reasons.append({'code': 'regression_execution'})
    collected = execution.get('collected', [])
    if Counter(collected) != Counter(nodes):
        reasons.append({'code': 'regression_nodes'})
    expected = Counter((node, when) for node in nodes for when in ('setup', 'call', 'teardown'))
    reports = execution.get('reports', [])
    if Counter((r.get('nodeid'), r.get('when')) for r in reports) != expected:
        reasons.append({'code': 'regression_reports'})
    if any(r.get('outcome') != 'passed' or r.get('wasxfail') for r in reports):
        reasons.append({'code': 'regression_outcome'})
    return reasons


def check_config_snapshots(run, inputs):
    reasons = []
    hashes = {r['path']: r['sha256'] for r in inputs}
    for source in CONFIG_PATHS:
        path = run/source.rsplit('/', 1)[-1]
        # Snapshot names are deliberately the source filenames: copy the exact bytes.
        if not path.is_file() or digest(path) != hashes.get(source):
            reasons.append({'code': 'config_snapshot', 'path': source})
    if reasons:
        return reasons
    cfg = read(run/'m2_controller.json')
    if cfg['hybrid_normal_force']['enabled'] is not False:
        reasons.append({'code': 'controller_profile'})
    from feedingrobot.validation.m2.spec import SCHEMA, cases, CALIBRATION_SEEDS, FORMAL_SEEDS
    acceptance = read(run/'m2_acceptance.json')
    expected = {'version': SCHEMA, 'cases': [f'T{i:02}' for i in range(1, 18)],
                'calibration_seeds': list(CALIBRATION_SEEDS), 'formal_seeds': list(FORMAL_SEEDS),
                'matrix': {'spec': 'feedingrobot.validation.m2.spec', 'case_families': len(cases()),
                           'calibration_trajectories': len(cases()) * len(CALIBRATION_SEEDS) * 3,
                           'formal_trajectories': len(cases()) * len(FORMAL_SEEDS) * 3}}
    if any(acceptance.get(key) != value for key, value in expected.items()):
        reasons.append({'code': 'acceptance_spec'})
    return reasons



def collect_inputs(root: Path | None = None) -> dict:
    """Independent input set. Does not read a report's own file list."""
    root = repo_root() if root is None else Path(root)
    from feedingrobot.scripts.validate_m1 import _tracked_files
    files = list(_tracked_files(root, root / "configs" / "m1_scene.json"))
    for folder in (root / "src" / "feedingrobot" / "controllers", root / "src" / "feedingrobot" / "validation"):
        found = [path.resolve() for path in folder.rglob("*.py") if path.is_file()]
        if not found:
            raise RuntimeError(f"missing required input directory {folder}")
        files.extend(found)
    for rel in ("configs/m2_controller.json", "configs/m2_acceptance.json", "assets/tests/m2_spring_surface.xml"):
        path = (root / rel).resolve()
        if not path.is_file():
            raise RuntimeError(f"missing required input {rel}")
        files.append(path)
    missing = sorted({str(path.relative_to(root)) for path in files if not path.is_file()})
    if missing:
        raise RuntimeError("missing required input " + missing[0])
    rows = [{"path": str(path.resolve().relative_to(root)), "sha256": digest(path.resolve())} for path in sorted(set(files))]
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update(row["path"].encode())
        aggregate.update(b"\0")
        aggregate.update(row["sha256"].encode())
        aggregate.update(b"\n")
    return {"files": rows, "aggregate_sha256": aggregate.hexdigest()}



def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def inputs():
    rows=collect_inputs()['files']
    root=repo_root()
    extras=['assets/tests/m2_full_surface.xml','assets/tests/m2_full_wall.xml','M2Plan.md','SimModelPlan.md','docs/m2_interface.md','docs/m2_acceptance.md','docs/m2_recovery.md','M2AcceptanceRefactorPlan.md']
    return sorted(rows+[{'path':p,'sha256':digest(root/p)} for p in extras],key=lambda r:r['path'])


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
