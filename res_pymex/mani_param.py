"""
# -*- coding: utf8 -*-
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
import re
import pandas as pd
import multiprocessing as mp
from string import Template
import os
from sys import platform
import subprocess
from pathlib import Path
import numpy as np
from .imex_tools import ImexTools


def my_pid():
    """Returns the relative pid of a pool
    process."""
    cur_proc = mp.current_process()
    if cur_proc._identity:
        return cur_proc._identity[0]
    return 0


def create_name():
    """Create dat name for parallel
    run."""
    return f"rank{my_pid()}"


class PyMEX(ImexTools):
    """
    Manipulate the files give in.
    """

    def __init__(self, controls, config_file, restore_file=None):
        super().__init__(controls, config_file)
        self.prod = None
        self.restore_file = restore_file
        self.basename = self.cmgfile(create_name())
        self.npv = None

    def _control_time(self):
        """Define control time."""
        time_conc = self.res_param["time_concession"]
        times = np.linspace(0, time_conc, int(time_conc / 30) + 1, dtype=int)
        control_time = np.round(self.time_steps()[:-1])
        id_sort = np.searchsorted(times, control_time)
        times = np.unique(np.insert(times, id_sort, control_time))
        return control_time, times

    @staticmethod
    def _alter_wells(well_type, number):
        """Create the wells names."""
        wells_name = [f"'{well_type}{i + 1}'" for i in range(number)]
        return " ".join(["*ALTER"] + wells_name)

    @staticmethod
    def _wells_rate(well_prod):
        """Return the string of the wells rate."""
        prod_values = map(str, np.round(well_prod, 4))
        return " ".join(prod_values)

    def _control_alter_strings(self, control):
        div = "**" + "-" * 30
        nb_prod = self.res_param["nb_prod"]
        nb_inj = self.res_param["nb_inj"]
        prod_name = self._alter_wells("P", nb_prod)
        prod_rate = self._wells_rate(control[:nb_prod])
        inj_name = self._alter_wells("I", nb_inj)
        inj_rate = self._wells_rate(control[nb_prod:])
        lines = [
            div,
            prod_name,
            prod_rate,
            div,
            inj_name,
            inj_rate,
            div,
            "\n",
        ]
        return "\n".join(lines)

    def _wells_alter_strings(self):
        controls = []
        for control in self.modif_controls:
            controls.append(self._control_alter_strings(control))
        return controls

    def _modify_cronograma_file(self):
        """Replace control tag <PyMEX>alter_wells</PyMEX>
        in cronograma_file.
        """
        with open(self.cronograma, "r+") as file:
            content = file.read()
            pattern = re.compile(r"<PyMEX>alter_wells</PyMEX>")
            controls = self._wells_alter_strings()
            for control in controls:
                content = pattern.sub(control, content, count=1)
        return content

    def _regular_spaced_time(self):
        content_text = []

        def write_alter_time(day, control):
            """Write the ALTER element."""
            div = "**" + "-" * 30
            if day == 0:
                time = "**TIME " + str(day)
            else:
                time = "*TIME " + str(day)

            wells = self._control_alter_strings(control)
            lines = [
                div,
                time,
                div,
                wells,
                "\n",
            ]
            return "\n".join(lines)

        control_time, times = self._control_time()
        count = 0
        ctime = control_time[count]

        for time_step in times[:-1]:
            if count < len(control_time) and ctime == time_step:
                control = self.modif_controls[count]
                content_text += write_alter_time(ctime, control)
                count += 1
                if count < len(control_time):
                    ctime = control_time[count]
            else:
                time = "*TIME " + str(time_step) + "\n"
                content_text += time
        return content_text

    def include_operation(self):
        """Print operation of the wells."""
        if self.res_param["change_cronograma_file"]:
            return self._modify_cronograma_file()
        else:
            return self._regular_spaced_time()

    def create_well_operation(self):
        """Create a include file (.inc) to be incorporated to the .dat.

        Wrap the design variables in separated control cycles
        if the problem is time variable,
        so the last list corresponds to these.
        new_controls = list(chunks(design_controls, npp + npi))
        """
        type_opera = self.res_param["type_opera"]
        if type_opera == 0:  # full_capacity
            self.full_capacity()

        with open(self.tpl, "r") as tpl, open(self.basename.dat, "w") as dat:
            template_content = tpl.read()
            if self.controls is not None:
                operation_content = self.include_operation()
                pattern = re.compile(r"<PyMEX>cronograma</PyMEX>")
                template_content = pattern.sub(operation_content, template_content)
            dat.write(template_content)

    def rwd_file(self):
        """create *.rwd (output conditions) from report.tmpl."""
        tpl_report = Path(self.res_param["path"]) / self.res_param["tpl_report"]
        with open(tpl_report, "r") as tmpl, open(self.basename.rwd, "w") as rwd:
            tpl = Template(tmpl.read())
            content = tpl.substitute(SR3FILE=self.basename.sr3.name)
            rwd.write(content)

    def run_imex(self):
        """call IMEX + Results Report."""
        # environ['CMG_HOME'] = '/cmg'

        with open(self.basename.log, "w") as log:
            if platform == 'linux':
                path = ["/cmg/RunSim.sh",
                        "imex",
                        "2018.10",
                        self.basename.dat]
            elif platform == 'win32':
                path = [self.res_param['cmg_exe_path'], '-f', self.basename.dat.name]
            procedure = subprocess.run(path, stdout=log, cwd=self.run_path, check=True, shell=True)
            self.run_report_results(procedure)

    def run_report_results(self, procedure):
        """Get production for results."""
        if procedure.returncode == 0:
            path_results = [self.res_param['cmg_results_path'],
                            '-f',
                            self.basename.rwd.name,
                            '-o',
                            self.basename.rwo.name]
            subprocess.run(
                path_results,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                cwd=self.run_path,
                check=True,
                shell=True
            )
            try:
                with open(self.basename.rwo, 'r+') as rwo:
                    self.prod = pd.read_csv(rwo, sep="\t", index_col=False, usecols=np.r_[:6], skiprows=np.r_[0,1,3:6])
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
                    if self.res_param['vol_m3_to_bbl']:
                        self.prod.loc[:, self.prod.columns != 'time'] *= 6.29
                    if self.res_param['gas_to_oil_equiv']:
                        self.prod.gas_prod /= 1017.045686
            except StopIteration as err:
                print("StopIteration error: Failed in Imex run.")
                print(f"Verify {self.basename.log}")
                raise err
        else:
            # IMEX has failed, receive None
            self.prod = None

    def restore_run(self):
        """Restart the IMEX run."""
        with open(self.basename["rwo"]) as rwo:
            self.prod = pd.read_csv(rwo, sep="\t", index_col=False, header=6)
        self.prod = self.prod.dropna(axis=1, how="all")
        self._colum_names()

    def cash_flow(self):
        """Return the cash flow from production."""
        production = self.prod.loc[:, ['oil_prod',
                                       'gas_prod',
                                       'water_prod',
                                       'water_inj']]
        return production.mul(self.res_param['prices']).sum(axis=1)

    def net_present_value(self):
        """ Calculate the net present value of the \
            reservoir production"""

        # Convert to periodic rate
        periodic_rate = ((1 + self.res_param["tma"]) ** (1 / 365)) - 1

        # Create the cash flow (x 10^6) (Format of the numpy.npv())
        cash_flows = self.cash_flow().to_numpy()

        # Discount tax
        time = self.prod["time"].to_numpy()
        tax = 1 / np.power((1 + periodic_rate), time)
        # Billions
        return np.sum(cash_flows * tax) * 1e-9

    def __call__(self):
        """
        Run Imex.
        """
        if not self.restore_file:
            # Verify if the Run_Path exist
            self.run_path.mkdir(parents=True, exist_ok=True)

            # Write the well controls in data file
            self.create_well_operation()

            # Create .rwd file
            self.rwd_file()

            # Run Imex + Results Report
            self.run_imex()

            # Evaluate the net present value
            if self.res_param["evaluate_npv"]:
                self.npv = self.net_present_value()

            # Remove all files create in Run Imex
            if self.res_param['clean_up_results']:
                self.clean_up()
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
