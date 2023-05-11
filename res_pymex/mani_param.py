"""
# -*- coding: utf8 -*-
# Copyright (c) 2019 Hygor Costa
#
# This file is part of Py_IMEX.
#
# You should have received a copy of the GNU General Public License
# along with HUM.  If not, see <http://www.gnu.org/licenses/>.
#
# Created: Jul 2019
# Author: Hygor Costa
"""
import os
import re
import sys
import subprocess
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

from .imex_tools import ImexTools


class PyMEX(ImexTools):
    """
    Manipulate the files give in.
    """

    def __init__(self, reservoir_dat: str, controls: np.ndarray, m3_to_bbl=True, gas_to_boe=True, restore_file=None):
        super().__init__(reservoir_dat)
        if isinstance(controls, list):
            controls = np.array(controls)
        self.scaled_controls = self.scale_variables(controls)
        self.m3_to_bbl = m3_to_bbl
        self.gas_to_boe = gas_to_boe
        self.prod = None
        self.procedure = None
        self.restore_file = restore_file

    @staticmethod
    def _wells_rate(well_prod):
        """Return the string of the wells rate."""
        prod_values = map(str, np.round(well_prod, 4))
        return " ".join(prod_values)

    def _control_alter_strings(self, control):
        div = "**" + "-" * 30
        wells_name = [f'\'{w}\'' for w in self.get_well_aliases()]
        alter = f'*ALTER {" ".join(wells_name)}'
        controls = self._wells_rate(control)
        lines = [div, alter, controls, div]
        return "\n".join(lines)

    def _wells_alter_strings(self):
        return [self._control_alter_strings(control) for control in self.scaled_controls]

    def _modify_cronograma_file(self):
        """Replace control tag <PyMEX>alter_wells</PyMEX>
        in cronograma_file.
        """
        with open(self.cronograma, "r+", encoding='UTF-8') as file:
            content = file.read()
            pattern = re.compile(r"<PyMEX>alter_wells</PyMEX>")
            controls = self._wells_alter_strings()
            for control in controls:
                content = pattern.sub(control, content, count=1)
        return content

    @staticmethod
    def _sub_include_path_file(text: str):
        return re.sub(r'INCLUDE\s+\'(.*)\'', r"INCLUDE '..\\\1'", text, flags=re.M)

    def create_well_operation(self):
        """Create a include file (.inc) to be incorporated to the .dat.
        Wrap the design variables in separated control cycles
        if the problem is time variable,
        so the last list corresponds to these.
        new_controls = list(chunks(design_controls, npp + npi))
        """
        with open(self.reservoir_tpl, "r", encoding='UTF-8') as tpl,\
                open(self.basename.dat, "w", encoding='UTF-8') as dat:
            template_content = tpl.read()
            template_content = self._sub_include_path_file(template_content)
            operation_content = self._modify_cronograma_file()
            pattern = r'^INCLUDE.*Cronograma.*\.inc\''
            template_content = re.sub(pattern,
                                      operation_content, template_content, flags=re.M)
            dat.write(template_content)

    def rwd_file(self):
        """create *.rwd (output conditions) from report.tmpl."""
        tpl_report = Path('res_pymex/TemplateReport.tpl')
        with open(tpl_report, "r", encoding='UTF-8') as tmpl, \
                open(self.basename.rwd, "w", encoding='UTF-8') as rwd:
            tpl = Template(tmpl.read())
            content = tpl.substitute(SR3FILE=self.basename.sr3.name)
            rwd.write(content)

    def get_cmg_run_path(self, pattern: str):
        """Search the IMEX executable.

        Returns:
            Path: Path istance
        """
        try:
            cmg_home = os.environ['CMG_HOME']
            paths = list(Path(cmg_home).rglob(pattern))
            return paths[-1]
        except KeyError as error:
            raise KeyError('Verifique se a variável de ambiente CMG_HOME existe!') from error

    def run_imex(self):
        """call IMEX + Results Report."""
        self.create_well_operation()
        self.rwd_file()
        try:
            with open(self.basename.log, 'w', encoding='UTF-8') as log:
                imex_path = self.get_cmg_run_path('mx20*.exe')
                path = [imex_path, '-f', self.basename.dat.name]
                self.procedure = subprocess.run(
                    path, stdout=log, cwd=self.run_path, check=True, shell=True)
        except subprocess.CalledProcessError as error:
            sys.exit(
                f"Error - Não foi possível executar o IMEX, verificar: {error}")

    def read_rwo_file(self):
        """Read output file (rwo) and build the production dataframe."""
        with open(self.basename.rwo, 'r+', encoding='UTF-8') as rwo:
            self.prod = pd.read_csv(rwo, sep="\t", index_col=False,
                                    usecols=np.r_[:6], skiprows=np.r_[0, 1, 3:6])
            self.prod = self.prod.rename(
                columns={
                    'TIME': 'time',
                    'Period Oil Production - Monthly SC': "oil_prod",
                    'Period Gas Production - Monthly SC': "gas_prod",
                    'Period Water Production - Monthly SC': "water_prod",
                    'Period Water Production - Monthly SC.1': "water_inj",
                    'Liquid Rate SC': "liq_prod"
                }
            )
            if self.m3_to_bbl:
                self.prod.loc[:, self.prod.columns != 'time'] *= 6.29
            if self.gas_to_boe:
                self.prod.gas_prod /= 1017.045686

    def run_report_results(self):
        """Get production for results."""
        try:
            path_results = [self.get_cmg_run_path('Report.exe'),
                            '-f',
                            self.basename.rwd.name,
                            '-o',
                            self.basename.rwo.name]
            self.procedure = subprocess.run(
                path_results,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                cwd=self.run_path,
                check=True,
                shell=True
            )
        except subprocess.CalledProcessError as error:
            print(f"Report não pode ser executador, verificar: {error}")

    def restore_run(self):
        """Restart the IMEX run."""
        with open(self.basename.rwo, 'r+', encoding='UTF-8') as rwo:
            self.prod = pd.read_csv(rwo, sep="\t", index_col=False, header=6)
            self.prod = self.prod.dropna(axis=1, how="all")
            self.prod = self.prod.rename(
                columns={
                    'TIME': 'time',
                    'Period Oil Production - Monthly SC': "oil_prod",
                    'Period Gas Production - Monthly SC': "gas_prod",
                    'Period Water Production - Monthly SC': "water_prod",
                    'Period Water Production - Monthly SC.1': "water_inj",
                    'Liquid Rate SC': "liq_prod"
                }
            )

    def cash_flow(self, prices: np.ndarray):
        """Return the cash flow from production."""
        production = self.prod.loc[:, ['oil_prod',
                                       'gas_prod',
                                       'water_prod',
                                       'water_inj']]
        return production.mul(prices).sum(axis=1)

    def npv(self, prices: np.ndarray, tma_aa:float):
        """ Calculate the net present value of the \
            reservoir production"""
        self.base_run()
        periodic_rate = ((1 + tma_aa) ** (1 / 365)) - 1
        cash_flows = self.cash_flow(prices).to_numpy()
        time = self.prod["time"].to_numpy()
        tax = 1 / np.power((1 + periodic_rate), time)
        return np.sum(cash_flows * tax) * 1e-9

    def base_run(self):
        """
        Run Imex.
        """
        if not self.restore_file:
            # Verify if the Run_Path exist
            self.run_path.mkdir(parents=True, exist_ok=True)
            self.run_imex()
            self.run_report_results()
            self.read_rwo_file()
        else:
            self.restore_run()

    def clean_up(self):
        """Delet imex auxiliar files."""
        for _, filename in self.basename._asdict().items():
            try:
                os.remove(filename)
            except OSError:
                print(
                    f"File {filename} could not be removed,\
                      check if it's yet open."
                )
