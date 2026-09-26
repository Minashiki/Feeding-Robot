"""Frozen test selection and configuration snapshots for full M2 runs."""
import json
from collections import Counter

from feedingrobot.sim.model import repo_root

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
    from feedingrobot.controllers.full_acceptance import digest, read
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
    return reasons
