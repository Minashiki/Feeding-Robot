"""A real physics trace cannot pass by reporting an unmoving reference."""
import numpy as np
import pytest

from feedingrobot.controllers.reference import ReferenceShaper
from feedingrobot.validation.m2.spec import Case
from feedingrobot.validation.m2.runner import prepare, run_variant
from feedingrobot.validation.m2.checks import score
from tests.m2_support import load_m2_config


@pytest.mark.parametrize('family', ['circle', 'sine', 'loaded_circle'])
def test_dropped_reference_motion_is_rejected(family, monkeypatch):
    case = Case(family, load=1. if family == 'loaded_circle' else 0.)
    config = load_m2_config()
    scene, effective, initial = prepare(case, 0, config)
    original = ReferenceShaper.shape
    def discard(self, twist, *args, **kwargs):
        return original(self, np.zeros(6), *args, **kwargs)
    monkeypatch.setattr(ReferenceShaper, 'shape', discard)
    data, meta = run_variant(case, 0, 'A', scene, effective, initial)
    errors, _ = score(case, 0, 'A', data, meta)
    codes = {row['code'] for row in errors}
    assert 'motion:' + family + '_target' in codes
    assert 'reference:candidate_twist' in codes
    assert not any(code.startswith(('identity:', 'shape:')) for code in codes)


@pytest.mark.parametrize('injection', ['disabled_block', 'replayed_command'])
def test_wall_faults_are_rejected(injection, monkeypatch):
    case = Case('wall')
    scene, effective, initial = prepare(case, 0, load_m2_config())
    if injection == 'disabled_block':
        original = ReferenceShaper._suppress
        def bypass(self, *args, **kwargs):
            saved = self.block_force
            self.block_force = float('inf')
            try:
                return original(self, *args, **kwargs)
            finally:
                self.block_force = saved
        monkeypatch.setattr(ReferenceShaper, '_suppress', bypass)
    else:
        original = ReferenceShaper.shape
        elapsed = 0.
        old_command = np.zeros(6)
        def replay(self, twist, pos, rot, force, dt, *args, **kwargs):
            nonlocal elapsed, old_command
            if np.linalg.norm(twist) > 1e-6:
                old_command = twist.copy()
            if elapsed >= 3.-1e-9:
                twist = old_command
            elapsed += dt
            return original(self, twist, pos, rot, force, dt, *args, **kwargs)
        monkeypatch.setattr(ReferenceShaper, 'shape', replay)
    data, meta = run_variant(case, 0, 'A', scene, effective, initial)
    codes = {row['code'] for row in score(case, 0, 'A', data, meta)[0]}
    expected = 'wall_progress_blocked' if injection == 'disabled_block' else 'release_distance'
    assert expected in codes, codes
    assert not any(code.startswith(('identity:', 'shape:')) for code in codes)
