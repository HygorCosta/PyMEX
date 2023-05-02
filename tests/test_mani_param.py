import pytest
import os
import pandas as pd
from res_pymex import PyMEX
import unittest.mock as mock

@pytest.fixture
def pymex():
    controls = [1] * 58
    return PyMEX(controls)

def test_run_imex(pymex):
    pymex()

def test_run_results_report(pymex):
    procedure = mock.MagicMock()
    procedure.returncode = 0
    pymex.run_report_results(procedure)
    assert pymex.prod is not None


def test_cash_flow(pymex):
    pymex.prod = pd.read_csv('producao.csv')
    cash_flow = pymex.cash_flow()
    assert len(cash_flow) == 361
    assert isinstance(cash_flow, pd.Series)

def test_npv(pymex):
    pymex.prod = pd.read_csv('producao.csv')
    pymex.net_present_value()
    assert isinstance(pymex.npv, float)
    assert pymex.npv > 0


def test_clean_up(pymex):
    pymex.clean_up()
    assert os.path.exists(pymex.basename.dat) == False
    assert os.path.exists(pymex.basename.sr3) == False
    assert os.path.exists(pymex.basename.out) == False
    assert os.path.exists(pymex.basename.rwo) == False
    assert os.path.exists(pymex.basename.rwd) == False
    assert os.path.exists(pymex.basename.log) == False