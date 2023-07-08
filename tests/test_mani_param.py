""" Test Mani Param - PyMEX Class"""
import os
from unittest import mock
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from res_pymex import PyMEX


@pytest.fixture(name='pymex_instance')
def pymex():
    """Create instance of Pymex to test.
    """
    controls = [1] * 58
    return PyMEX('reservoir_tpl/Pituba/Pituba.dat', controls)

def test_imex_run_path_is_correct_find(pymex_instance):
    """Verifica se o arquivo executável do IMEX foi 
    corretamente encontrado.
    """
    esperado = Path(r'C:\Program Files\CMG\IMEX\2022.10\Win_x64\EXE\mx202210.exe')
    obtido = pymex_instance.get_imex_run_path()
    assert esperado.samefile(obtido)

def test_imex_return_code_is_zero(pymex_instance):
    """Check if the imex actually run and return code was zero."""
    pymex_instance.run_imex()
    assert pymex_instance.procedure.returncode == 0

def test_run_results_report_return_pandas_production(pymex_instance):
    """Teste if production is not none.
    """
    procedure = mock.MagicMock()
    procedure.returncode = 0
    pymex_instance.run_report_results(procedure)
    assert pymex.prod is not None

def test_cash_flow_was_created(pymex_instance):
    """Test if clashflow was created and if is a pandas Series.
    """
    pymex_instance.prod = pd.read_csv('producao.csv')
    cash_flow = pymex_instance.cash_flow()
    assert len(cash_flow) == 361
    assert isinstance(cash_flow, pd.Series)

def test_npv_is_greater_than_zero(pymex_instance):
    """Test if vpl is positive by default.
    """
    pymex_instance.prod = pd.read_csv('producao.csv')
    pymex_instance.net_present_value()
    assert isinstance(pymex_instance.npv, float)
    assert pymex_instance.npv > 0


def test_clean_up(pymex_instance):
    """Test if clean up temporary works is working.
    """
    pymex_instance.clean_up()
    assert os.path.exists(pymex_instance.basename.dat) is False
    assert os.path.exists(pymex_instance.basename.sr3) is False
    assert os.path.exists(pymex_instance.basename.out) is False
    assert os.path.exists(pymex_instance.basename.rwo) is False
    assert os.path.exists(pymex_instance.basename.rwd) is False
    assert os.path.exists(pymex_instance.basename.log) is False

def test_sub_include_path_file(pymex_instance):
    """Test file include path manipulation."""
    texto = """
        GRID CORNER 163 98 7

    INCLUDE 'INCLUDE\corners.inc'
    INCLUDE 'INCLUDE\ntg.inc'
    INCLUDE 'INCLUDE\por.inc'
    INCLUDE 'INCLUDE\permi.inc'
    PERMJ  EQUALSI
    INCLUDE 'INCLUDE\permk.inc'
    INCLUDE 'INCLUDE\null.inc'
    INCLUDE 'INCLUDE\falhas.inc'
    INCLUDE 'INCLUDE\pinchout.inc'


    CPOR 130E-6
    PRPOR 272.95"""
    obtido = pymex_instance._sub_include_path_file(texto)
    assert isinstance(obtido, str)


def test_create_well_operation(pymex_instance):
    price = np.array([70, -2, -2, -2])
    tma = 0.1
    obtido = pymex_instance.npv(price, tma)
    assert obtido > 0
    assert isinstance(obtido, float)