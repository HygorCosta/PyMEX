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

from pathlib import Path

import numpy as np
import yaml


class ImexTools:
    """Create tools for simulate the value of the well_controls in
    IMEX."""

    def __init__(self, well_controls, config_file):
        """
        Parameters
        ----------
        well_controls: array
            Values of the wells rate.
        res_param: dictionary
            Reservoir parameters.
        """
        self.controls = np.array(well_controls)
        self.res_param = self._read_config_file(config_file)
        self.run_path = self._set_paths("run_folder")
        self.tpl = self._set_paths("tpl")
        if well_controls:
            self.modif_controls = self.multiply_variables()
        if self.res_param["change_cronograma_file"]:
            self.cronograma = self._set_paths("cronograma")

    def _read_config_file(self, config_file):
        with open(config_file, "r", encoding='UTF-8') as file:
            res_param = yaml.load(file, Loader=yaml.FullLoader)
        return res_param

    def _set_paths(self, folder_name: str):
        path = Path(self.res_param["path"])
        return path / self.res_param[folder_name]

    def well_rate_max(self):
        """Get the maximum rate."""
        produtor = self._well_bounds_rate(well='prod')
        injetor = self._well_bounds_rate(well='inj')
        return np.concatenate((produtor, injetor))

    def _well_bounds_rate(self, well:str):
        """Get wells bounds rate

        Args:
            well (str): 'prod' for producers and 'inj' for injectors

        Returns:
            np.array: shape(nwells, 1)
        """
        well_rate = None
        if well == 'prod':
            bound = self.res_param['max_rate_prod']
            nwells = self.res_param['nb_prod']
        elif well == 'inj':
            bound = self.res_param['max_rate_inj']
            nwells = self.res_param['nb_inj']            
        if isinstance(bound, list) and len(bound) == nwells:
            well_rate = np.array(bound)
        elif isinstance(bound, (int, float)):
            well_rate = np.repeat(bound, nwells)
        return well_rate

 
    def multiply_variables(self):
        """Transform the design variables."""
        controls_per_cycle = np.split(self.controls, self.res_param["nb_cycles"])
        return [control * self.well_rate_max() for control in controls_per_cycle]

    def time_steps(self):
        """Manipulate the time variable to write in template"""
        # Start cycle - Time ZERO:
        time_type = self.res_param["type_time"]
        nb_cycles = self.res_param["nb_cycles"]
        time_conc = self.res_param["time_concession"]
        time_steps = np.insert(self.modif_controls[-1], 0, 0)

        if time_type == 1:  # time include as design variable
            time_steps = np.cumsum(time_steps)
        else:
            time_steps = np.linspace(start=0, stop=time_conc, num=nb_cycles + 1)
        return time_steps

    def full_capacity(self):
        """Determine the well ratio of the last well (prod or inj)"""
        max_plat_prod = self.res_param["max_plat_prod"]
        max_plat_inj = self.res_param["max_plat_inj"]
        nb_prod = self.res_param["nb_prod"]
        nb_inj = self.res_param["nb_inj"]
        for index, control in enumerate(self.modif_controls):
            last_prod_well = max_plat_prod - sum(control[:nb_prod])
            last_inj_well = max_plat_inj - sum(control[nb_inj])
            self.modif_controls[index].insert(nb_prod, last_prod_well)
            self.modif_controls[index].append(last_inj_well)
