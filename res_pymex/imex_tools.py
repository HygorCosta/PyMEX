"""Imex Tools.
# *- coding: utf8 -*-
# Copyright (c) 2019 Hygor Costa
#
# This file is part of Py_IMEX.
#
#
# You should have received a copy of the GNU General Public License
# along with HUM.  If not, see <http://www.gnu.org/licenses/>.
#
# Created: Jul 2019
# Author: Hygor Costa
"""

import multiprocessing as mp
import re
import shutil
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class ImexTools:
    """Create tools to simulate the value of the well_controls in
    IMEX.

    Input:
    - reservoir_tpl(str): path to reservoir tpl file
    - well_controls(np.array): unitary controls of each variable
        np.array([0, 1, 1, 1, 0.5, 0.5, 0.6])  - 7 variables
    - num_wells(np.array): number of wells
        np.array([num_producers, num_injectors]) - np.array([4, 3])
    - max_well_rate(np.array): maximum well rate of each well
    """

    def __init__(self, run_config):
        """
        Parameters
        ----------
        well_controls: array
            Values of the wells rate.
        res_param: dictionary
            Reservoir parameters.
        """
        self.cfg = self.read_config_file(run_config)
        self.reservoir_tpl = Path(self.cfg['tpl'])
        self.temp_run = self.reservoir_tpl.parent / 'temp_run'
        self.basename = self.cmgfile()
        self.schedule = self.inc_run_path / self.cfg['schedule']
        self.max_plat_prod = self.cfg['max_plat_prod']
        self.max_plat_inj = self.cfg['max_plat_inj']
        self.info_wells = self.get_wells_info()
        self.run_path = None

    @staticmethod
    def read_config_file(run_config):
        """Read yaml file and convert to a dictionary."""
        with open(run_config, "r", encoding='UTF-8') as yamlfile:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return data

    def well_rate_max(self):
        """Get the maximum rate."""
        # prod_max = self.get_constraint('prod', 'primary')
        # inj_max = self.get_constraint('inj', 'primary')
        prod_max = self.cfg['max_rate_prod']
        inj_max = self.cfg['max_rate_inj']
        return np.array(prod_max + inj_max)

    def cmgfile(self):
        """
        A simple wrapper for retrieving CMG file extensions
        given the basename.
        :param basename:
        :return:
        """
        basename = f'{self.reservoir_tpl.stem}_{mp.current_process().pid}'
        self.run_path = self.temp_run / basename
        self.run_path.mkdir(parents=True, exist_ok=False)
        self.inc_run_path = self.run_path / self.cfg['inc_folder']
        shutil.copytree(self.reservoir_tpl.parent / self.cfg['inc_folder'], 
                        self.inc_run_path, dirs_exist_ok=True)
        basename = self.run_path / basename
        Extension = namedtuple(
            "Extension",
            "dat out irf mrf rwd rwo log sr3",
        )
        basename = Extension(
            basename.with_suffix(".dat"),
            basename.with_suffix(".out"),
            basename.with_suffix(".irf"),
            basename.with_suffix(".mrf"),
            basename.with_suffix(".rwd"),
            basename.with_suffix(".rwo"),
            basename.with_suffix(".log"),
            basename.with_suffix(".sr3"),
        )
        return basename

    def scale_variables(self, controls):
        """Transform the design variables."""
        num_cycles = len(controls) / self.num_wells
        controls_per_cycle = np.split(controls, num_cycles)
        return [control * self.well_rate_max() for control in controls_per_cycle]

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
            pd.Dataframa: nome do poço, ordem de restrição, propriedade
            de operação, tipo de restrição (min|max) e valor
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
        # producers = self.info_wells.groupby('well_type').get_group('prod')
        # return producers['well'].nunique()
        return self.cfg['nb_prod']

    @property
    def num_injectors(self):
        """Number of well injectors"""
        injectors = self.info_wells.groupby('well_type').get_group('inj')
        return injectors['well'].nunique()

    @property
    def num_wells(self):
        """Total number of wells"""
        # return self.info_wells['well'].nunique()
        return self.cfg['nb_prod'] + self.cfg['nb_inj']

    def get_well_aliases(self):
        """Sequence alias order for well producers and injetors."""
        # return self.info_wells['well'].unique()
        return self.cfg['prod_names'] + self.cfg['inj_names']

    def _parse_include_files(self, datafile):
        """Parse simulation file for *INCLUDE files and return a list."""
        with open(datafile, "r", encoding='UTF-8') as file:
            lines = file.read()

        pattern = r'\n\s*[*]?\s*include\s+[\'|"]([\.\w-])[\'|"]'
        return re.findall(pattern, lines, flags=re.IGNORECASE)

    # def copy_to(self):
    #     """Copy simulation files to destination directory."""
    #     src_files = [self.reservoir_tpl.parent / f for f in self._parse_include_files(self.reservoir_tpl)]
    #     dst_files = [self.run_path / f for f in self._parse_include_files(self.reservoir_tpl)]
    #     for src, dst in zip(src_files, dst_files):
    #         dst.mkdir(parents=True, exist_ok=True)
    #         shutil.copy(src, dst)
