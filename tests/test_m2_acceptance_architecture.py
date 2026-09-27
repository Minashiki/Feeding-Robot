"""Dependency boundaries, raw schema and immutable verification contracts."""
import ast
from pathlib import Path

from feedingrobot.sim.model import repo_root
from feedingrobot.validation.m2.provenance import collect_inputs


def test_validation_sources_are_hashed():
    root = repo_root()
    paths = {row['path'] for row in collect_inputs()['files']}
    expected = {str(p.relative_to(root)) for p in (root/'src/feedingrobot/validation').rglob('*.py')}
    assert expected <= paths
    assert 'tests/m2_full_nodeids.json' in paths


def test_dependency_boundaries():
    root = repo_root()/'src/feedingrobot'
    for name in ['cartesian_impedance', 'reference', 'guard', 'wrench', 'contracts']:
        tree = ast.parse((root/'controllers'/f'{name}.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('feedingrobot.validation')
    for path in (root/'validation/m2').glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and path.stem != 'legacy':
                assert not any((node.module or '').endswith('.' + name) for name in ['v3spec', 'v4fields', 'v6checks', 'acceptance'])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and path.stem not in {'runner', 'negatives'}:
                assert node.func.attr != 'mj_step'


def test_verify_legacy_never_runs_counterexamples(tmp_path, monkeypatch):
    from feedingrobot.validation.m2 import legacy
    from feedingrobot.controllers import v3spec
    def forbid(*args, **kwargs):
        raise AssertionError('read-only verifier attempted counterexample execution')
    monkeypatch.setattr(v3spec, 'run_adversarial', forbid)
    verdict = legacy.verify_finished_report(tmp_path, 'fixes-v6', False)
    assert not verdict['scope_passed'] and not verdict['m3_ready']
    assert not list(tmp_path.iterdir())
