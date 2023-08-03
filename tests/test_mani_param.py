""" Test Mani Param - PyMEX Class"""
import os
from unittest import mock
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from res_pymex.mani_param import PyMEX


@pytest.fixture(name='olymp')
def pymex_olymp():
    """Create instance of Pymex to test.
    """
    controls = [1] * 36
    return PyMEX('model/OLYMPUS/config_olympus.yaml', np.array(controls), 44)

def test_inicializar_classe(olymp: PyMEX):
    assert olymp.realization == None
    assert olymp.production == None
    assert isinstance(olymp.model.tpl, Path)

def test_imex_run_path_is_correct_find(olymp: PyMEX):
    """Verifica se o arquivo executável do IMEX foi
    corretamente encontrado.
    """
    esperado = Path(r'C:\Program Files\CMG\IMEX\2022.10\Win_x64\EXE\mx202210.exe')
    assert esperado.samefile(olymp.cmginfo.sim_exe)

def test_imex_return_code_is_zero(olymp: PyMEX):
    """Check if the imex actually run and return code was zero."""
    olymp.base_run()
    assert olymp.procedure.returncode == 0

def test_run_results_report_return_pandas_production(olymp: PyMEX):
    """Teste if production is not none.
    """
    procedure = mock.MagicMock()
    procedure.returncode = 0
    olymp.run_report_results(procedure)
    assert pymex.prod is not None

def test_cash_flow_was_created(olymp: PyMEX):
    """Test if clashflow was created and if is a pandas Series.
    """
    olymp.prod = pd.read_csv('producao.csv')
    cash_flow = olymp.cash_flow()
    assert len(cash_flow) == 361
    assert isinstance(cash_flow, pd.Series)

def test_npv_is_greater_than_zero(olymp: PyMEX):
    """Test if vpl is positive by default.
    """
    olymp.prod = pd.read_csv('producao.csv')
    olymp.net_present_value()
    assert isinstance(olymp.npv, float)
    assert olymp.npv > 0


def test_clean_up(olymp: PyMEX):
    """Test if clean up temporary works is working.
    """
    olymp.clean_up()
    assert os.path.exists(olymp.basename.dat) is False
    assert os.path.exists(olymp.basename.sr3) is False
    assert os.path.exists(olymp.basename.out) is False
    assert os.path.exists(olymp.basename.rwo) is False
    assert os.path.exists(olymp.basename.rwd) is False
    assert os.path.exists(olymp.basename.log) is False

def test_sub_include_path_file(olymp: PyMEX):
    """Test file include path manipulation."""
    texto = r"""
        GRID CORNER 163 98 7

    INCLUDE 'INCLUDE/corners.inc'
    INCLUDE 'INCLUDE/ntg.inc'
    INCLUDE 'INCLUDE/por.inc'
    INCLUDE 'INCLUDE/permi.inc'
    PERMJ  EQUALSI
    INCLUDE 'INCLUDE/permk.inc'
    INCLUDE 'INCLUDE/null.inc'
    INCLUDE 'INCLUDE/falhas.inc'
    INCLUDE 'INCLUDE/pinchout.inc'


    CPOR 130E-6
    PRPOR 272.95"""
    obtido = olymp._sub_include_path_file(texto)
    assert isinstance(obtido, str)


def test_create_well_operation(olymp: PyMEX):
    price = np.array([70, -2, -2, -2])
    tma = 0.1
    obtido = olymp.npv(price, tma)
    assert obtido > 0
    assert isinstance(obtido, float)


def test_scaled_controls(olymp: PyMEX):
    assert len(olymp.scaled_controls[0]) == 18
    assert len(olymp.scaled_controls) == 2

def test_run(olymp: PyMEX):
    obtido = olymp.npv()
    assert obtido != 0

def test_run_olympus_write_dat_file(olymp: PyMEX):
    obtido = olymp.write_dat_file()
    assert obtido != 0

def test_run_olympus_copy_to(olymp: PyMEX):
    olymp.write_dat_file()
    olymp.copy_to()
    assert 1
