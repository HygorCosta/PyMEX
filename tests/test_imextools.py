"""Test ImexTools class."""
import pytest
from res_pymex import ImexTools


@pytest.fixture(name='imext')
def imex_instance():
    """Create the ImexTools instance."""
    return ImexTools('reservoir_tpl/Pituba/Pituba.dat')
    

def test_cmgfile_basename(imext):
    """Check basename files."""
    folder = imext.run_path
    assert 1