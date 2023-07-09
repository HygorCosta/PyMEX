"""Read IMEX .dat File"""
import re
from pathlib import Path

import numpy as np
import pandas as pd


class ReadDatFile:
    """Class to read the data file and extract reservoir
    information.

    These two keywords are mandatory in dat file.
        INCLUDE 'INCLUDE\\Produtores.inc'
        INCLUDE 'INCLUDE\\Injetores.inc'
    """

    def __init__(self, reservoir_dat) -> None:
        self.reservoir_tpl = Path(reservoir_dat)
        self.max_plat_prod = self.read_group_controls('prod', 'STL')
        self.max_plat_inj = self.read_group_controls('inj', 'STW')
        self.info_wells = self.get_wells_info()

    @staticmethod
    def _check_well_type(well_type: str):
        """Convert well type input"""
        group = None
        if well_type.lower() == 'inj':
            group = 'GCONI'
        elif well_type.lower() == 'prod':
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
            >>> self.read_group_controls('prod', 'stl')
            15830.73

        """
        with open(self.reservoir_tpl, 'r+', encoding='UTF-8') as file:
            dat = file.read()
            group = self._check_well_type(well_type)
            pattern = fr'^{group}[\s\S]+?MAX\s+{const_type.upper()}\s+(\d+(?:\.\d+)?)'
            max_plat = re.findall(pattern, dat, flags=re.M)
            if len(max_plat):
                max_plat = float(max_plat[0])
            else:
                max_plat = None
        return max_plat

    def read_wells_include_file(self, rel_path: str):
        """Read include wells files.

        Args:
            rel_path (str): relative path to injectors ou producers include wells.

        Returns:
            pd.Dataframa: nome do poço, ordem de restrição, propriedade de operação, tipo de restrição (min|max) e valor
        """
        producers_inc = self.reservoir_tpl.parent / rel_path
        with open(producers_inc, 'r+', encoding='UTF-8') as file:
            producers = file.read()
            well_pattern = r'^WELL\s+\'(\w+)\''
            operate_pattern = r'^OPERATE?\s+(MAX|MIN)\s+(STL|BHP|STO|STW).+?(\d+(?:\.\d+)?)\s+CONT$'
            wells_alias = re.findall(well_pattern, producers, flags=re.M)
            constraints = re.findall(operate_pattern, producers, flags=re.M)
            df = pd.DataFrame(constraints, columns=[
                              'const_type', 'operate', 'value'])
            df['const_order'] = np.where(df.index % 2, 'secondary', 'primary')
            df['well'] = np.repeat(wells_alias, 2)
        return df[['well', 'const_order', 'operate', 'const_type', 'value']]

    def get_constraint(self, well_type: str, group: str):
        """Get constraint from producers.

        Args:
            well_type (str): 'prod' for producers and 'inj' for injectores
            group (str): select the constraint group from 'primary' or 'secondary'

        Returns:
            nd.array: with the same length of wells
        """
        well_group = self.info_wells.groupby('well_type').get_group(well_type)
        const_grouped = well_group.groupby('const_order')
        return const_grouped.get_group(group)['value'].astype(float).to_numpy()

    def get_wells_info(self):
        """Dataframe with wells information."""
        prod_info = self.read_wells_include_file(
            rel_path='INCLUDE/Produtores.inc')
        prod_info['well_type'] = 'prod'
        inj_info = self.read_wells_include_file(
            rel_path='INCLUDE/Injetores.inc')
        inj_info['well_type'] = 'inj'
        return pd.concat([prod_info, inj_info], ignore_index=True)

    @property
    def num_producers(self):
        """Number of well producers"""
        producers = self.info_wells.groupby('well_type').get_group('prod')
        return producers['well'].nunique()

    @property
    def num_injectors(self):
        """Number of well injectors"""
        injectors =  self.info_wells.groupby('well_type').get_group('inj')
        return injectors['well'].nunique()

    @property
    def num_wells(self):
        """Total number of wells"""
        return self.info_wells['well'].nunique()

    def get_well_aliases(self):
        """Sequence alias order for well producers and injetors."""
        return self.info_wells['well'].unique()
