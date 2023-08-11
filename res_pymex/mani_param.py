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
import logging
import os
import re
import sys
import subprocess
from pathlib import Path
from string import Template
from collections import namedtuple
from .settings import Model, Wells, Optimization

import numpy as np
import pandas as pd
import shutil
import yaml

from .imex_tools import cmgfile


logging.basicConfig(format='%(process)d-%(message)s')


class PyMEX:
    """
    Manipulate the files give in.
    """
    __cmginfo = namedtuple("cmginfo", "home_path sim_exe report_exe")

    def __init__(self, reservoir_config:yaml, controls:np.ndarray, model_number:int = None):
        self.import_settings(reservoir_config)
        self.realization = model_number
        self.scaled_controls = self._get_scaled_controls(controls)
        self.cmginfo = self.get_cmginfo()
        self.production = None
        self.procedure = None

    def import_settings(self, config_yaml):
        with open(config_yaml, "r", encoding='UTF-8') as yamlfile:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
            self.model = self._import_model(data)
            self.wells = self._import_wells_operate(data)
            self.opt = self._import_opt_settings(data)

    @staticmethod
    def _import_model(data_yaml):
        new_model = Model()
        data_yaml = data_yaml['model']
        new_model.tpl = Path(data_yaml['dat_tpl'])
        new_model.temp_run = new_model.tpl.parent / 'temp_run'
        new_model.basename = cmgfile(new_model.temp_run / new_model.tpl.name)
        new_model.tpl_report = Path(data_yaml['report_tpl'])
        new_model.tpl_schedule = Path(data_yaml['schedule_tpl'])
        return new_model

    @staticmethod
    def _import_wells_operate(data_yaml):
        new_well = Wells()
        new_well.control_type = data_yaml['wells']['control_type']
        #Producers
        prod_yaml = data_yaml['wells']['producers']
        new_well.prod = prod_yaml['names']
        new_well.prod_operate = np.array([prod_yaml['max_operation'], prod_yaml['min_operation']], np.float64)
        #Injectors
        inj_yaml = data_yaml['wells']['injectors']
        new_well.inj = inj_yaml['inj_names']
        new_well.inj_operate = np.array([inj_yaml['max_operation'], inj_yaml['min_operation']], np.float64)
        #Constraints
        new_well.max_plat = np.array(data_yaml['constraint']['platform']['prod_max_slt'])
        return new_well

    @staticmethod
    def _import_opt_settings(data_yaml):
        new_opt = Optimization()
        data_yaml = data_yaml['optimization']
        new_opt.numb_cic = int(data_yaml['nb_cycles'])
        new_opt.tma = data_yaml['tma']
        new_opt.prices = data_yaml['prices']
        new_opt.parasol = data_yaml['num_cores_parasol']
        new_opt.clean_files = data_yaml['clean_up_results']
        return new_opt

    @staticmethod
    def _wells_rate(well_prod):
        """Return the string of the wells rate."""
        prod_values = map(str, np.round(well_prod, 5))
        return '\t' + " ".join(prod_values)

    def _control_alter_strings(self, control):
        div = "**" + "-" * 30
        wells = self.wells.prod + self.wells.inj
        wells_name = [f'\'{well}\'' for well in wells]
        prod = "**" + "-" * 9 + " PRODUCERS " + "-" * 10
        alter_prod = f'*ALTER {" ".join(wells_name[:self.numb_prod])}'
        inj = "**" + "-" * 9 + " INJECTORS " + "-" * 10
        alter_inj = f'*ALTER {" ".join(wells_name[self.numb_prod:])}'
        prod_controls = self._wells_rate(control[:self.numb_prod])
        inj_controls = self._wells_rate(control[self.numb_prod:])
        lines = [div, prod, alter_prod, prod_controls,
                 div, inj, alter_inj, inj_controls, div]
        return "\n".join(lines)

    @property
    def _wells_operate_bounds(self):
        return np.hstack((self.wells.prod_operate, self.wells.inj_operate))

    def _delta_well_operate(self):
        return self._wells_operate_bounds[0, :] - self._wells_operate_bounds[1,:]

    def _get_scaled_controls(self, controls):
        controls_per_cycle = np.split(controls, self.opt.numb_cic)
        delta_control = self._delta_well_operate()
        min_control = self._wells_operate_bounds[1,:]
        return [min_control + delta_control * control
                        for control in controls_per_cycle]

    def _wells_alter_strings(self):
        return [self._control_alter_strings(control) for control in self.scaled_controls]

    def _modify_cronograma_file(self):
        """Replace control tag <PyMEX>alter_wells</PyMEX>
        in cronograma_file.
        """
        with open(self.model.tpl_schedule, "r+", encoding='UTF-8') as file:
            content = file.read()
            pattern = re.compile(r"<PyMEX>ALTER_WELLS</PyMEX>")
            controls = self._wells_alter_strings()
            for control in controls:
                content = pattern.sub(control, content, count=1)

        with open(self.model.basename.schedule, "w", encoding='UTF-8') as file:
            file.write(content)

    def write_dat_file(self):
        """Copy dat file to run path."""
        with open(self.model.tpl, "r", encoding='UTF-8') as tpl:
            content = re.sub(r'\*?INCLUDE\s+\'', r"INCLUDE '..//", tpl.read(), flags=re.S)
            content = Template(content)
        with open(self.model.basename.dat, "w", encoding='UTF-8') as dat:
            if self.realization:
                content = content.substitute(N1=self.realization,
                                             SCHEDULE=f'\'{self.model.basename.schedule.name}\'')
            else:
                content = content.substitute(SCHEDULE=f'\'{self.model.basename.schedule.name}\'')
            dat.write(content)

    def rwd_file(self):
        """create *.rwd (output conditions) from report.tmpl."""
        with open(self.model.tpl_report, "r", encoding='UTF-8') as tmpl, \
                open(self.model.basename.rwd, "w", encoding='UTF-8') as rwd:
            tpl = Template(tmpl.read())
            content = tpl.substitute(SR3FILE=self.model.basename.sr3.name)
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

    def save_dat_file(self):
        """Only generates the dat file and schedule include."""
        self.model.temp_run.mkdir(parents=True, exist_ok=True)
        self.write_dat_file()
        self._modify_cronograma_file()

    def process_and_get_npv(self):
        """Takes the sr3 file and calculate the npv"""
        self.run_report_results()
        self.read_rwo_file()
        self.clean_up()
        return self.npv()

    def run_imex(self):
        """call IMEX + Results Report."""
        # self.create_well_operation()
        self.model.temp_run.mkdir(parents=True, exist_ok=True)
        self.write_dat_file()
        # self.copy_to()
        self._modify_cronograma_file()
        self.rwd_file()
        try:
            logging.debug('Run IMEX...')
            with open(self.model.basename.log, 'w', encoding='UTF-8') as log:
                sim_command = [self.cmginfo.sim_exe, "-f",
                                self.model.basename.dat.absolute(), "-wait", "-dd"]
                if self.opt.parasol > 1:
                    sim_command += ["-parasol", str(self.opt.parasol), "-doms"]
                self.procedure = subprocess.run(sim_command, stdout=log, check=True)
        except subprocess.CalledProcessError as error:
            sys.exit(
                f"Error - Não foi possível executar o IMEX, verificar: {error}")

    def read_rwo_file(self):
        """Read output file (rwo) and build the production dataframe."""
        with open(self.model.basename.rwo, 'r+', encoding='UTF-8') as rwo:
            self.production = pd.read_csv(rwo, sep="\t", index_col=False,
                                    usecols=np.r_[:5], skiprows=np.r_[0, 1, 3:6])
            self.production = self.production.rename(
                columns={
                    'TIME': 'time',
                    'Period Oil Production - Monthly SC': "oil_prod",
                    'Period Gas Production - Monthly SC': "gas_prod",
                    'Period Water Production - Monthly SC': "water_prod",
                    'Period Water Production - Monthly SC.1': "water_inj"
                }
            )

    def run_report_results(self):
        """Get production for results."""
        try:
            logging.debug('Run Results Report...')
            report_command = [self.cmginfo.report_exe,
                            '-f',
                            self.model.basename.rwd.name,
                            '-o',
                            self.model.basename.rwo.name]
            self.procedure = subprocess.run(
                report_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                cwd=self.model.basename.dat.parent,
                shell=True,
                check=True
            )
        except subprocess.CalledProcessError as error:
            print(f"Report não pode ser executador, verificar: {error}")

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
        periodic_rate = ((1 + self.cfg['tma']) ** (1 / 365.25)) - 1
        cash_flows = self.cash_flow(self.cfg['prices']).to_numpy()
        time = self.prod["time"].to_numpy()
        tax = 1 / np.power((1 + periodic_rate), time)
        return np.sum(cash_flows * tax) * 1e-9

    def base_run(self):
        """
        Run Imex.
        """
        logging.debug('## Inicialized PyMEX... ')
        self.model.temp_run.mkdir(parents=True, exist_ok=True)
        self.run_imex()
        self.run_report_results()
        self.read_rwo_file()
        self.clean_up()

    def clean_up(self):
        """ Clean files from run path."""
        logging.debug(f'Deleting {self.model.basename.dat.stem}')
        for file in self.model.basename:
            file.unlink(missing_ok=True)

    def get_realization(self, content, model_number):
        """Replace $N1 for model number in TPL dat."""
        tpl = Template(content)
        content = tpl.substitute(N1=model_number)
        return content

    def _parse_include_files(self, datafile):
        """Parse simulation file for *INCLUDE files and return a list."""
        with open(datafile, "r", encoding='UTF-8') as file:
            lines = file.read()

        pattern = r'\n\s*\*?include\s*[\'|"](.*)[\'|"]'
        return re.findall(pattern, lines, flags=re.IGNORECASE)

    def copy_to(self):
        """Copy simulation files to destination directory."""
        logging.debug('Copying include files...')
        if self.realization is None: #deterministic
            shutil.copytree(self.reservoir_tpl.parent / self.cfg['inc_folder'],
                        self.inc_run_path, dirs_exist_ok=True)
        else: #robust
            src_files = [self.reservoir_tpl.parent / f for f in self._parse_include_files(self.basename.dat)]
            dst_files = [self.basename.dat.parent / f for f in self._parse_include_files(self.basename.dat)]
            for src, dst in zip(src_files, dst_files):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)

    @property
    def numb_prod(self):
        return len(self.wells.prod)

    @property
    def numb_inj(self):
        return len(self.wells.inj)

    @property
    def num_wells(self):
        return self.numb_prod + self.numb_inj
