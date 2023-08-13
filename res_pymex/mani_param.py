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
import math
from pathlib import Path
import subprocess
from string import Template
from collections import namedtuple
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import yaml
from .settings import Model, Wells, Optimization
from .imex_tools import cmgfile


logging.basicConfig(format='%(process)d-%(message)s')


class PyMEX:
    """
    Manipulate the files give in.
    """
    __cmginfo = namedtuple("cmginfo", "home_path sim_exe report_exe")

    def __init__(self, reservoir_config:yaml):
        self.import_settings(reservoir_config)
        self.cmginfo = self.get_cmginfo()
        self._controls = None
        self._realization = None
        self._production = None

    def import_settings(self, config_yaml):
        """Import reservoir and optimization settings.
        Example in config folder.

        Args:
            config_yaml (str): yaml file
        """
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
        return new_model

    @staticmethod
    def _import_wells_operate(data_yaml):
        new_well = Wells()
        new_well.control_type = data_yaml['wells']['control_type']
        #Producers
        prod_yaml = data_yaml['wells']['producers']
        new_well.prod = prod_yaml['names']
        new_well.prod_operate = np.array([prod_yaml['max_operation'], prod_yaml['min_operation']],
                                          np.float64)
        #Injectors
        inj_yaml = data_yaml['wells']['injectors']
        new_well.inj = inj_yaml['names']
        new_well.inj_operate = np.array([inj_yaml['max_operation'], inj_yaml['min_operation']],
                                         np.float64)
        #Constraints
        new_well.max_plat = np.array(data_yaml['constraint']['platform']['prod_max_slt'])
        return new_well

    @staticmethod
    def _import_opt_settings(data_yaml):
        new_opt = Optimization()
        data_yaml = data_yaml['optimization']
        new_opt.dates = data_yaml['dates']
        new_opt.numb_cic = math.ceil(12*new_opt.dates['conc']/new_opt.dates['step_control'])
        new_opt.tma = data_yaml['tma']
        new_opt.prices = data_yaml['prices']
        new_opt.parasol = data_yaml['num_cores_parasol']
        new_opt.clean_files = data_yaml['clean_up_results']
        new_opt.sao = data_yaml['sao']
        return new_opt

    @staticmethod
    def _wells_rate(well_prod):
        """Return the string of the wells rate."""
        prod_values = map(str, np.round(well_prod, 5))
        return '\t' + " ".join(prod_values)

    def _control_alter_strings(self, controls):
        div = "**" + "-" * 30
        wells = self.wells.prod + self.wells.inj
        target = ['** Inserted by PyMEX', div]
        for well, control in zip(wells, controls):
           bhp =  f'*TARGET {self.wells.control_type}\n'
           name_well = f'\t\'{well}\'\n'
           value = f'\t{round(control,5)}\n'
           target.append(bhp + name_well + value)
        target.append(div)
        return "\n".join(target)

    @property
    def _wells_operate_bounds(self):
        return np.hstack((self.wells.prod_operate, self.wells.inj_operate))

    @property
    def basename(self):
        """Get basename file."""
        return self.model.basename.dat.name

    @basename.setter
    def basename(self, new_name:str):
        self.model.basename = cmgfile(self.model.temp_run / new_name)

    @property
    def controls(self):
        """Get controls."""
        return self._controls

    @controls.setter
    def controls(self, controls:np.ndarray):
        """Get scaled controls from nominal ones."""
        if np.any(controls > 1):
            raise AttributeError('Controls out of range.')
        self._controls = self._get_scaled_controls(controls)

    @property
    def realization(self):
        """Get realization value"""
        return self._realization

    @realization.setter
    def realization(self, realization:int):
        self._realization = realization

    @property
    def production(self):
        """Get production pandas dataset"""
        return self._production

    def _get_scaled_controls(self, controls):
        controls_per_cycle = np.split(controls, self.opt.numb_cic)
        delta_control = self._wells_operate_bounds[0, :] - self._wells_operate_bounds[1,:]
        min_control = self._wells_operate_bounds[1,:]
        return [min_control + delta_control * control
                        for control in controls_per_cycle]

    # def _modify_cronograma_file(self):
    #     """Replace control tag <PyMEX>alter_wells</PyMEX>
    #     in cronograma_file.
    #     """
    #     with open(self.model.tpl_schedule, "r+", encoding='UTF-8') as file:
    #         content = file.read()
    #         pattern = re.compile(r"<PyMEX>ALTER_WELLS</PyMEX>")
    #         controls = [self._control_alter_strings(control) for control in self._controls]
    #         for control in controls:
    #             content = pattern.sub(control, content, count=1)

    #     with open(self.model.basename.schedule, "w", encoding='UTF-8') as file:
    #         file.write(content)

    def _write_scheduling(self) -> str:
        start = self.opt.dates['start']
        end = start + relativedelta(years=self.opt.dates['conc'])
        date_range = range(self.opt.dates['write_freq'],
                            12*self.opt.dates['conc'],
                            self.opt.dates['write_freq'])
        date_list = [start + relativedelta(months=x) for x in date_range]
        if end not in date_list:
            date_list.append(end)
        control_range = range(0,
                              12*self.opt.dates['conc'],
                              self.opt.dates['step_control'])
        control_list = [start + relativedelta(months=x) for x in control_range]
        controls = iter([self._control_alter_strings(control) for control in self._controls])
        schedule = []
        bool_control = np.isin(date_list, control_list)
        schedule.append(next(controls))
        schedule.append(f'*DATE {start.strftime("%Y %m %d")}.1\n')
        for date, is_control in zip(date_list, bool_control):
            schedule.append(f'*DATE {date.strftime("%Y %m %d")}')
            if is_control:
                schedule.append(next(controls))
                schedule.append(f'*DATE {date.strftime("%Y %m %d")}.1\n')
        return '\n'.join(schedule)

    def write_dat_file(self):
        """Copy dat file to run path."""
        self.model.temp_run.mkdir(parents=True, exist_ok=True)
        with open(self.model.tpl, "r", encoding='UTF-8') as tpl:
            content = re.sub(r'\*?INCLUDE\s+\'', r"INCLUDE '../", tpl.read(), flags=re.S)
            content = Template(content)
        with open(self.model.basename.dat, "w", encoding='UTF-8') as dat:
            if self._realization:
                content = content.substitute(N1=self._realization,
                                             SCHEDULE=self._write_scheduling())
            else:
                content = content.substitute(SCHEDULE=self._write_scheduling())
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

    def process_and_get_npv(self):
        """Takes the sr3 file and calculate the npv"""
        self.run_report_results()
        self.read_rwo_file()
        self.clean_up()
        return self.npv()

    def run_imex(self) -> int:
        """call IMEX + Results Report."""
        # self.create_well_operation()
        try:
            logging.info('Run IMEX...')
            with open(self.model.basename.log, 'w', encoding='UTF-8') as log:
                sim_command = [self.cmginfo.sim_exe, "-f",
                                self.model.basename.dat.absolute(), "-wait", "-dd"]
                if self.opt.parasol > 1:
                    sim_command += ["-parasol", str(self.opt.parasol), "-doms"]
                return subprocess.run(sim_command, stdout=log, check=True)
        except subprocess.CalledProcessError as error:
            sys.exit(
                f"Error - Não foi possível executar o IMEX, verificar: {error}")

    def read_rwo_file(self):
        """Read output file (rwo) and build the production dataframe."""
        with open(self.model.basename.rwo, 'r+', encoding='UTF-8') as rwo:
            self._production = pd.read_csv(rwo, sep="\t", index_col=False,
                                    usecols=np.r_[:5], skiprows=np.r_[0, 1, 3:6])
            self._production = self._production.rename(
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
            logging.info('Run Results Report...')
            report_command = [self.cmginfo.report_exe,
                            '-f',
                            self.model.basename.rwd.name,
                            '-o',
                            self.model.basename.rwo.name]
            return subprocess.run(
                report_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                cwd=self.model.basename.dat.parent,
                shell=True,
                check=True
            )
        except subprocess.CalledProcessError as error:
            print(f"Report não pode ser executador, verificar: {error}")
            return 1

    def npv(self):
        """ Calculate the net present value of the \
            reservoir production"""
        periodic_rate = ((1 + self.opt.tma) ** (1 / 365.25)) - 1
        cash_flows = self._production.loc[:, ['oil_prod',
                                       'gas_prod',
                                       'water_prod',
                                       'water_inj']
                                    ].mul(self.opt.prices).sum(axis=1).to_numpy()
        time = self._production["time"].to_numpy()
        tax = 1 / np.power((1 + periodic_rate), time)
        return np.sum(cash_flows * tax) * 1e-9

    def base_run(self):
        """
        Run Imex.
        """
        logging.info('## Inicialized PyMEX... ')
        self.model.temp_run.mkdir(parents=True, exist_ok=True)
        self.run_imex()
        self.run_report_results()
        self.read_rwo_file()
        self.clean_up()

    def clean_up(self):
        """ Clean files from run path."""
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

    @property
    def numb_prod(self):
        """Number of producer wells.

        Returns:
            int
        """
        return len(self.wells.prod)

    @property
    def numb_inj(self):
        """Number of injector wells.

        Returns:
            int
        """
        return len(self.wells.inj)

    @property
    def num_wells(self):
        """Total number of wells.

        Returns:
            int
        """
        return self.numb_prod + self.numb_inj
