"""Test read dat class."""
import os

import numpy as np
import pandas as pd
import pytest

from res_pymex.read_dat import ReadDatFile


@pytest.fixture(name='pituba')
def read_dat_instance():
    """Create a instance of ReadDatFile"""
    return ReadDatFile('reservoir_tpl/Pituba/Pituba_test.dat')


def test_if_reservoir_tpl_folder_is_find(pituba):
    """Test if the dat path is a file"""
    assert os.path.isfile(pituba.reservoir_tpl)


def test_read_max_group_rate_returns_a_list(pituba):
    """Test if the functions returns a list with the maximum 
    water injection and liquid production."""
    obtido = pituba.read_max_group_rate()
    assert isinstance(obtido, list)
    assert len(obtido) == 2


@pytest.mark.parametrize('input_x, input_y, output',
                         [('prod', 'STL', 15898.73),
                          ('prod', 'STO', 15898.73),
                          ('prod', 'stw', 9539.238),
                          ('inj', 'STW', 18000)])
def test_read_gcomp(pituba, input_x, input_y, output):
    """Test if read group controls works for injection
    production."""
    obtido = pituba.read_group_controls(input_x, input_y)
    assert obtido == output


def test_read_producers(pituba):
    """Test if the read works for producers wells."""
    obtido = pituba.read_producers('INCLUDE/Produtores.inc')
    assert isinstance(obtido, pd.DataFrame)
    assert len(obtido.index) == 30


def test_get_constraint(pituba):
    """Test if constraint returns a numpy array with
    15 elements"""
    obtido = pituba.get_constraint('prod', 'primary')
    assert len(obtido) == 15
    assert np.all(obtido == 2000)


def test_hub_info_wells(pituba):
    """Test if the hub of wells information is correct read."""
    obtido = pituba.info_wells
    assert isinstance(obtido, pd.DataFrame)
    assert obtido['well'].nunique() == 29


def test_num_producers(pituba):
    """Test if producers equals 15."""
    obtido = pituba.num_producers
    assert obtido == 15


def test_num_injectors(pituba):
    """Test if injectors equals 14."""
    obtido = pituba.num_injectors
    assert obtido == 14

def test_num_wells(pituba):
    """Test if the total number of wells is equal to 29."""
    obtido = pituba.num_wells
    assert obtido == 29


def test_get_well_aliases(pituba):
    """Test if the aliases of all wells are received."""
    obtido = pituba.get_well_aliases()
    assert len(obtido) == 29
    assert obtido[0] == 'P1'
    assert obtido[-1] == 'I14'
