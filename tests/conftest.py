import pytest

from feedingrobot.sim.model import load_config


def pytest_addoption(parser):
    parser.addoption("--m1-config", default="configs/m1_scene.json")


@pytest.fixture
def m1_config(request):
    return load_config(request.config.getoption("--m1-config"))
