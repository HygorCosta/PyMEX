""" Test Mani Param - PyMEX Class"""
import os
from unittest import mock

import pandas as pd
import pytest

from res_pymex import PyMEX


@pytest.fixture
def pymex():
    """Create instance of Pymex to test.
    """
    controls = [1] * 58
    return PyMEX(controls, 'example/config.yaml')


def test_run_results_report(pymex):
    """Teste if production is not none.

    Args:
        pymex (instance of PyMEX)
    """
    procedure = mock.MagicMock()
    procedure.returncode = 0
    pymex.run_report_results(procedure)
    assert pymex.prod is not None


def test_cash_flow(pymex):
    """Test if clashflow was created and if is a pandas Series.
    """
    pymex.prod = pd.read_csv('producao.csv')
    cash_flow = pymex.cash_flow()
    assert len(cash_flow) == 361
    assert isinstance(cash_flow, pd.Series)

def test_npv(pymex):
    """Test if vpl is positive by default.
    """
    pymex.prod = pd.read_csv('producao.csv')
    pymex.net_present_value()
    assert isinstance(pymex.npv, float)
    assert pymex.npv > 0


def test_clean_up(pymex):
    """Test if clean up temporary works is working.
    """
    pymex.clean_up()
    assert os.path.exists(pymex.basename.dat) is False
    assert os.path.exists(pymex.basename.sr3) is False
    assert os.path.exists(pymex.basename.out) is False
    assert os.path.exists(pymex.basename.rwo) is False
    assert os.path.exists(pymex.basename.rwd) is False
    assert os.path.exists(pymex.basename.log) is False
