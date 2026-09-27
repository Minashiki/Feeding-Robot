"""A real physics trace cannot pass by reporting an unmoving reference."""
import numpy as np
import pytest

from feedingrobot.controllers.reference import ReferenceShaper
from feedingrobot.validation.m2.spec import Case
from feedingrobot.validation.m2.runner import prepare, run_variant
from feedingrobot.validation.m2.checks import score
from tests.m2_support import load_m2_config


@pytest.mark.parametrize('family', ['circle', 'sine'])
def test_dropped_reference_motion_is_rejected(family, monkeypatch):
    case = Case(family)
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
