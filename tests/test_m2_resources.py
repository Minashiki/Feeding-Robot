"""Fail closed if the desktop resource envelope is missing or exceeded."""
import pytest
from feedingrobot.scripts.m2_limited import MEMORY_MAX, THREAD_VARS, verify_limits


@pytest.fixture
def limits(monkeypatch):
    for name in THREAD_VARS:monkeypatch.setenv(name,'1')
    return {'cpu.max':'100000 100000','memory.max':str(MEMORY_MAX),'memory.swap.max':'0'}


def test_resource_envelope(limits):
    verify_limits(limits)


@pytest.mark.parametrize('key,value',[('cpu.max','max 100000'),('cpu.max','200000 100000'),('memory.max','max'),('memory.max',str(MEMORY_MAX+1)),('memory.swap.max','max')])
def test_refuse_unbounded_or_large_job(limits,key,value):
    limits[key]=value
    with pytest.raises(RuntimeError):verify_limits(limits)


def test_refuse_nested_threads(limits,monkeypatch):
    monkeypatch.setenv('OPENBLAS_NUM_THREADS','8')
    with pytest.raises(RuntimeError):verify_limits(limits)
