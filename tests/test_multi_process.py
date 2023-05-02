from res_pymex import ParallelPyMex
import pytest
import numpy as np


@pytest.fixture
def pymex():
    config_file = 'example/config.yaml'
    return ParallelPyMex(config_file, pool_size=2)


def test_pool_run(pymex):
    c1 = [1] * 58
    c2 = [0.5] * 58
    controls = np.array([c1, c2])
    npv = pymex.pool_run(controls)
    assert len(npv) == 2