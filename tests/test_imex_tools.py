import pytest

from res_pymex import ImexTools


@pytest.fixture(name='ito')
def create_obj():
    """Create the instance of ImexTools"""
    return ImexTools('model/OLYMPUS/config_olympus.yaml')


def test_parse_include_files(ito):
    """Find all files paths for include files."""
    inc = ito._parse_include_files(ito.reservoir_tpl)
    assert len(inc) > 1