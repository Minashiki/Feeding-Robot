"""M2 IK must not invalidate the food placement accepted before the arm moved."""
import numpy as np
import pytest
from feedingrobot.validation.m2.runner import prepare
from feedingrobot.validation.m2.spec import Case
from feedingrobot.sim.model import load_config


@pytest.mark.parametrize('seed', [0, 1, 2, 4])
def test_plate_food_is_validated_after_ik(seed):
    scene, _, _ = prepare(Case('plate'), seed, load_config('configs/m2_controller.json'))
    source = scene.m2_initial_source
    attempt = source['food_placement_attempt']
    assert isinstance(attempt, int) and 0 <= attempt < 8
    candidates = np.random.default_rng(seed).uniform(-.025, .025, (8, 2))
    expected = np.r_[np.array([.45, -.18]) + candidates[attempt], .03]
    adr = scene.index.food_qpos_adr
    np.testing.assert_allclose(source['pre_settle_qpos'][adr:adr+3], expected, atol=1e-12, rtol=0)
    assert not scene._illegal()
    if seed == 4:
        assert attempt > 0
