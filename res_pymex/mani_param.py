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
from collections import namedtuple

import numpy as np
import pandas as pd

from .imex_tools import ImexTools


class PyMEX(ImexTools):
    """
    Manipulate the files give in.
    """
    __cmginfo = namedtuple("cmginfo", "home_path sim_exe report_exe")

    def __init__(self, config_reservoir: str, controls: np.ndarray):
        super().__init__(config_reservoir)
        if isinstance(controls, list):
            controls = np.array(controls)
        self.scaled_controls = self.scale_variables(controls)
        self.prod = None
        self.procedure = None

    @staticmethod
    def _wells_rate(well_prod):
        """Return the string of the wells rate."""
        prod_values = map(str, np.round(well_prod, 4))
        return " ".join(prod_values)

    def _control_alter_strings(self, control):
        div = "**" + "-" * 30
        wells_name = [f'\'{well}\'' for well in self.get_well_aliases()]
        prod = "**" + "-" * 9 + " PRODUCERS " + "-" * 10
        alter_prod = f'*ALTER {" ".join(wells_name[:self.num_producers])}'
        inj = "**" + "-" * 9 + " INJECTORS " + "-" * 10
        alter_inj = f'*ALTER {" ".join(wells_name[self.num_producers:])}'
        prod_controls = self._wells_rate(control[:self.num_producers])
        inj_controls = self._wells_rate(control[self.num_producers:])
        lines = [div, prod, alter_prod, prod_controls,
                 div, inj, alter_inj, inj_controls, div]
        return "\n".join(lines)

    def _wells_alter_strings(self):
        return [self._control_alter_strings(control) for control in self.scaled_controls]

    def _modify_cronograma_file(self):
        """Replace control tag <PyMEX>alter_wells</PyMEX>
        in cronograma_file.
        """
        with open(self.schedule, "r+", encoding='UTF-8') as file:
            content = file.read()
            pattern = re.compile(r"<PyMEX>ALTER_WELLS</PyMEX>")
            controls = self._wells_alter_strings()
            for control in controls:
                content = pattern.sub(control, content, count=1)

        with open(self.schedule, "w", encoding='UTF-8') as file:
            file.write(content)

    @staticmethod
    def _sub_include_path_file(text: str):
        return re.sub(r'INCLUDE\s+\'(.*)\'', r"INCLUDE '..\\\1'", text, flags=re.M)

    # def create_well_operation(self):
    #     """Create a include file (.inc) to be incorporated to the .dat.
    #     Wrap the design variables in separated control cycles
    #     if the problem is time variable,
    #     so the last list corresponds to these.
    #     new_controls = list(chunks(design_controls, npp + npi))
    #     """
    #     with open(self.reservoir_tpl, "r", encoding='UTF-8') as tpl,\
    #             open(self.basename.dat, "w", encoding='UTF-8') as dat:
    #         template_content = tpl.read()
    #         # template_content = self._sub_include_path_file(template_content)
    #         self._modify_cronograma_file()

    def write_dat_file(self):
        """Copy dat file to run path."""
        with open(self.reservoir_tpl, "r", encoding='UTF-8') as tpl:
            content = tpl.read()
        with open(self.basename.dat, "w", encoding='UTF-8') as dat:
            dat.write(content)

    def rwd_file(self):
        """create *.rwd (output conditions) from report.tmpl."""
        tpl_report = Path('res_pymex/TemplateReport.tpl')
        with open(tpl_report, "r", encoding='UTF-8') as tmpl, \
                open(self.basename.rwd, "w", encoding='UTF-8') as rwd:
            tpl = Template(tmpl.read())
            content = tpl.substitute(SR3FILE=self.basename.sr3.absolute())
            rwd.write(content)

    @classmethod
    def get_cmginfo(cls):
        """Get CMG Simulatior Information and Executables."""
        try:
            cmg_home = os.environ['CMG_HOME']
            simulator = list(Path(cmg_home).rglob('mx*.exe'))
            sim_exe = sorted(simulator)[-1]
            report = list(Path(cmg_home).rglob('report*.exe'))
            report_exe = sorted(report)[-1]
            return cls.__cmginfo(cmg_home, sim_exe, report_exe)
        except KeyError as error:
            raise KeyError(
                'Verifique se a variável de ambiente CMG_HOME existe!') from error       

    def run_imex(self):
        """call IMEX + Results Report."""
        # self.create_well_operation()
        if not hasattr(self, "cmginfo"):
            self.cmginfo = self.get_cmginfo()
        self._modify_cronograma_file()
        self.rwd_file()
        self.write_dat_file()
        try:
            with open(self.basename.log, 'w', encoding='UTF-8') as log:
                sim_command = [self.cmginfo.sim_exe, "-f",
                                self.basename.dat.absolute(), "-wait", "-dd"]
                if self.cfg['num_cores_parasol'] > 1:
                    sim_command += ["-parasol", str(self.cfg['num_cores_parasol']), "-doms"]
                self.procedure = subprocess.Popen(sim_command, stdout=log)
                # self.procedure = subprocess.run(
                #     path, stdout=log, cwd=self.temp_run, check=True, shell=True)
        except subprocess.CalledProcessError as error:
            sys.exit(
                f"Error - Não foi possível executar o IMEX, verificar: {error}")

    def read_rwo_file(self):
        """Read output file (rwo) and build the production dataframe."""
        with open(self.basename.rwo, 'r+', encoding='UTF-8') as rwo:
            self.prod = pd.read_csv(rwo, sep="\t", index_col=False,
                                    usecols=np.r_[:5], skiprows=np.r_[0, 1, 3:6])
            self.prod = self.prod.rename(
                columns={
                    'TIME': 'time',
                    'Period Oil Production - Monthly SC': "oil_prod",
                    'Period Gas Production - Monthly SC': "gas_prod",
                    'Period Water Production - Monthly SC': "water_prod",
                    'Period Water Production - Monthly SC.1': "water_inj"
                }
            )
            # if self.m3_to_bbl:
            #     self.prod.loc[:, self.prod.columns != 'time'] *= 6.29
            # if self.gas_to_boe:
            #     self.prod.gas_prod /= 1017.045686

    def run_report_results(self):
        """Get production for results."""
        try:
            report_command = [self.cmginfo.report_exe,
                            '-f',
                            self.basename.rwd.name,
                            '-o',
                            self.basename.rwo.name]
            self.procedure = subprocess.run(
                report_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                cwd=self.basename.dat.parent,
                shell=True,
                check=True
            )
        except subprocess.CalledProcessError as error:
            print(f"Report não pode ser executador, verificar: {error}")

    # def restore_run(self):
    #     """Restart the IMEX run."""
    #     with open(self.basename.rwo, 'r+', encoding='UTF-8') as rwo:
    #         self.prod = pd.read_csv(rwo, sep="\t", index_col=False, header=6)
    #         self.prod = self.prod.dropna(axis=1, how="all")
    #         self.prod = self.prod.rename(
    #             columns={
    #                 'TIME': 'time',
    #                 'Period Oil Production - Monthly SC': "oil_prod",
    #                 'Period Gas Production - Monthly SC': "gas_prod",
    #                 'Period Water Production - Monthly SC': "water_prod",
    #                 'Period Water Production - Monthly SC.1': "water_inj",
    #                 'Liquid Rate SC': "liq_prod"
    #             }
    #         )

    def cash_flow(self, prices: np.ndarray):
        """Return the cash flow from production."""
        production = self.prod.loc[:, ['oil_prod',
                                       'gas_prod',
                                       'water_prod',
                                       'water_inj']]
        return production.mul(prices).sum(axis=1)

    def npv(self):
        """ Calculate the net present value of the \
            reservoir production"""
        self.base_run()
        periodic_rate = ((1 + self.cfg['tma']) ** (1 / 365)) - 1
        cash_flows = self.cash_flow(self.cfg['prices']).to_numpy()
        time = self.prod["time"].to_numpy()
        tax = 1 / np.power((1 + periodic_rate), time)
        return np.sum(cash_flows * tax) * 1e-9

    def base_run(self):
        """
        Run Imex.
        """
        # if not self.restore_file:
        # Verify if the Run_Path exist
        self.temp_run.mkdir(parents=True, exist_ok=True)
        self.run_imex()
        self.run_report_results()
        self.read_rwo_file()
        # else:
        #     self.restore_run()

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

    # def _control_time(self):
    #     """Define control time."""
    #     time_conc = self.res_param["time_concession"]
    #     times = np.linspace(0, time_conc, int(time_conc / 30) + 1, dtype=int)
    #     control_time = np.round(self.time_steps()[:-1])
    #     id_sort = np.searchsorted(times, control_time)
    #     times = np.unique(np.insert(times, id_sort, control_time))
    #     return control_time, times