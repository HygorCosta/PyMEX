"""Read IMEX .dat File"""
import re
from pathlib import Path


class ReadDatFile:
    """Class to read the data file and extract reservoir
    information.

    These two keywords are mandatory in dat file.
        INCLUDE 'INCLUDE\\Produtores.inc'
        INCLUDE 'INCLUDE\\Injetores.inc'
    """

    def __init__(self, reservoir_dat) -> None:
        self.reservoir_tpl = Path(reservoir_dat)

    @staticmethod
    def _check_well_type(well_type: str):
        """Convert well type input"""
        group = None
        if well_type.lower()=='inj':
            group = 'GCONI'
        elif well_type.lower()=='prod':
            group = 'GCONP'
        else:
            raise ValueError('Well type inj or prod')
        return group

    def read_group_controls(self, well_type: str, const_type: str):
        """Read group controls to platform production or injection constraints.

        Args:
            well_type (str): 'inj' for group of injector and 'prod' for group of producers
            const_type (str): constraint type (STW and STL)

        Returns:
            float: maximum platform operation (injection or production) or None if is not specified

        Examples:
            >>> self.read_group_controls('inj', 'STW')
            18000.0
            >>> self.read_group_controls('prod', 'STL')
            15830.73

        """
        with open(self.reservoir_tpl, 'r+', encoding='UTF-8') as file:
            dat = file.read()
            group = self._check_well_type(well_type)
            pattern = fr'{group}[\s\S]+?MAX\s+{const_type}\s+(\d+\.?\d+)'
            max_plat = re.findall(pattern, dat, flags=re.M)
            if len(max_plat):
                max_plat = float(max_plat[0])
            else:
                max_plat = None
        return max_plat
