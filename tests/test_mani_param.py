""" Test Mani Param - PyMEX Class"""
import os
import asyncio
from pathlib import Path

import pytest
import numpy as np

from res_pymex.mani_param import PyMEX


@pytest.fixture(name='olymp')
def pymex_olymp():
    """Create instance of Pymex to test.
    """
    controls_1 = [0] * 11 + [1] * 7
    controls_2 = [0] * 11 + [1] * 7
    controls_3 = [0] * 11 + [1] * 7
    controls = controls_1 + controls_2 + controls_3
    model = PyMEX('./model/olympus_config.yaml')
    model.controls = np.array(controls)
    model.realization = 44
    return model

def test_inicializar_classe(olymp: PyMEX):
    """Testar a inicialização da classe"""
    assert olymp.model is not None
    assert olymp.wells is not None
    assert olymp.opt is not None
    assert hasattr(olymp.cmginfo, 'home_path')
    assert hasattr(olymp.cmginfo, 'sim_exe')
    assert hasattr(olymp.cmginfo, 'report_exe')
    assert len(olymp.controls) == 3
    assert olymp.controls[0].shape == (18,)
    assert olymp.realization == 44
    assert olymp.production is None

# def test_write_sheduling(olymp):
#     """Test write scheduling string """
#     scheduling = olymp._write_scheduling()
#     assert isinstance(scheduling, str)

def test_imex_run_path_is_correct_find(olymp: PyMEX):
    """Verifica se o arquivo executável do IMEX foi
    corretamente encontrado.
    """
    esperado = Path(r'C:\Program Files\CMG\IMEX\2022.10\Win_x64\EXE\mx202210.exe')
    assert esperado.samefile(olymp.cmginfo.sim_exe)

def test_write_dat_file(olymp):
    """Test if the modified datafile is created."""
    olymp.write_dat_file()
    assert os.path.isfile(olymp.model.basename.dat)

def test_write_dat_file_with_new_basename(olymp):
    """Test if the modified datafile is created
    with a new basename."""
    i = 3
    olymp.basename = f'Opt_{i:03d}'
    olymp.write_dat_file()
    assert os.path.isfile(olymp.model.basename.dat)

def test_write_dat_file_with_new_basename_and_run_imex(olymp):
    """Test if the modified datafile is created
    with a new basename."""
    i = 3
    olymp.basename = f'Opt_{i:03d}'
    olymp.write_dat_file()
    assert olymp.run_imex()

def test_write_rwd_file(olymp):
    """Test if the modified datafile is created
    with a new basename."""
    i = 2
    olymp.basename = f'Opt_{i:03d}'
    olymp.rwd_file()
    assert os.path.isfile(olymp.model.basename.rwd)

def test_run_report(olymp):
    """Test if the modified datafile is created
    with a new basename."""
    i = 3
    olymp.basename = f'Opt_{i:03d}'
    olymp.rwd_file()
    olymp.run_report_results()
    assert os.path.isfile(olymp.model.basename.rwo)

def test_read_rwo(olymp):
    """Test if the modified datafile is created
    with a new basename."""
    i = 2
    olymp.basename = f'Opt_{i:03d}'
    olymp.read_rwo_file()
    assert olymp.production is not None

def test_npv_greather_then_zero(olymp):
    """Test if the modified datafile is created
    with a new basename."""
    i = 3
    olymp.basename = f'Opt_{i:03d}'
    olymp.read_rwo_file()
    npv = olymp.npv()
    assert npv > 0

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

def test_run(olymp: PyMEX):
    obtido = olymp.npv()
    assert obtido != 0

def test_run_olympus_copy_to(olymp: PyMEX):
    olymp.write_dat_file()
    olymp.copy_to()
    assert 1

def test_write_multiple_async_files(olymp):
    controls = np.ones((20, 54))
    realizations = 10*[5, 44]
    names = [f'Opt_{i:03d}' for i in range(1, 21)]
    asyncio.run(
        olymp.write_multiple_dat_and_rwd_files(
            names, controls, realizations
        )
    )
    assert os.path.isfile('model/temp_run/Opt_001.dat')
    assert os.path.isfile('model/temp_run/Opt_001.rwd')
    assert os.path.isfile('model/temp_run/Opt_002.dat')
    assert os.path.isfile('model/temp_run/Opt_002.rwd')
    assert os.path.isfile('model/temp_run/Opt_003.dat')
    assert os.path.isfile('model/temp_run/Opt_004.dat')
    assert os.path.isfile('model/temp_run/Opt_005.dat')
    assert os.path.isfile('model/temp_run/Opt_006.dat')
    assert os.path.isfile('model/temp_run/Opt_007.dat')
    assert os.path.isfile('model/temp_run/Opt_008.dat')
    assert os.path.isfile('model/temp_run/Opt_009.dat')
    assert os.path.isfile('model/temp_run/Opt_010.dat')
    assert os.path.isfile('model/temp_run/Opt_010.rwd')
    assert os.path.isfile('model/temp_run/Opt_020.dat')
    assert os.path.isfile('model/temp_run/Opt_020.rwd')

def test_results_report(olymp):
    """Async results report execution"""
    base = olymp.model.temp_run / 'ResOpt'
    basenames = [f'{base}_{i:04d}' for i in range(1, 21)]
    results = asyncio.run(
        olymp.assync_results_report(basenames)
    )
    # olymp._results_report(basenames)
    assert results[0].stdout is not None

def test_assync_results_report(olymp):
    """Async results report execution"""
    base = olymp.model.temp_run / 'ResOpt'
    basenames = [f'{base}_{i:04d}' for i in range(1, 51)]
    olymp.parallel_results_report(basenames)
    # olymp._results_report(basenames)
    assert 1
