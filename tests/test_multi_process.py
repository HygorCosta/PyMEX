import numpy as np
import pytest

from res_pymex.multi_process import ParallelPyMex


@pytest.fixture
def pymex():
    config_file = 'model\OLYMPUS\config_olympus.yaml'
    return ParallelPyMex(config_file, pool_size=4)


def test_pool_run(pymex):
    c1 = [1.00] * 36
    c2 = [0.50] * 36
    c3 = [0.25] * 36
    # c4 = [0.10] * 36
    realizations = [2, 25, 42]
    controls = np.array([c1, c2, c3])
    npv = pymex.pool_run(controls, realizations)
    assert len(npv) == 3


def test_net_present_value(pymex):
    c1 = [1.00] * 36
    c2 = [0.50] * 36
    c3 = [0.25] * 36
    # c4 = [0.10] * 36
    realizations = [1, 10, 27]
    # controls = np.array([c1, c2, c4, c4])
    controls = np.array([c1])
    npv = pymex.net_present_value(c1, 10)
    assert len(npv) == 4
