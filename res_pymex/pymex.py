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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
from typing import List
from string import Template
import asyncio
import aiofiles
from dateutil.relativedelta import relativedelta
import numpy as np
import polars as pl
import yaml
from .settings import Settings
from .util import cmgfile, delete_files, get_cmginfo


class PyMEX(Settings):
    """
    Manipulate the files give in.
    """

    def __init__(self, reservoir_config:yaml):
        super().__init__(reservoir_config)
        self.cmginfo = get_cmginfo()
        self._tpl_model = self._read_tpl_model()
        self._tpl_rwd = self._read_rwd_tpl()
        self.model.temp_run.mkdir(parents=True, exist_ok=True)
        self._controls = None
        self._realization = None
        self._production = None

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

    def _read_tpl_model(self):
        """Read template model"""
        with open(self.model.tpl, "r", encoding='UTF-8') as tpl:
            content = re.sub(r'\*?INCLUDE\s+\'', r"INCLUDE '../", tpl.read(), flags=re.S)
            return Template(content)

    async def _write_dat_async(self, filename, content):
        async with aiofiles.open(filename, 'w+', encoding='utf-8') as file:
            await file.write(content)

    async def write_multiple_dat_and_rwd_files(self,
                                                basenames:List[str],
                                                controls:np.ndarray,
                                                realizations:List[int]=None):
        """Write one dat file for each control in controls list."""
        tasks = []
        if realizations is not None:
            for basename, control, realization in zip(basenames, controls, realizations):
                self.basename = basename
                self.controls = control
                self.realization = realization
                dat_content = self._tpl_model.substitute(N1=self._realization,
                                                SCHEDULE=self._write_scheduling())
                rwd_content = self._tpl_rwd.substitute(SR3FILE=self.model.basename.sr3.name)
                tasks.append(
                    asyncio.ensure_future(
                    self._write_dat_async(self.model.basename.dat, dat_content)
                    )
                )
                tasks.append(
                    asyncio.ensure_future(
                    self._write_dat_async(self.model.basename.rwd, rwd_content)
                    )
                )
        else:
            for basename, control in zip(basenames, controls):
                self.basename = basename
                self.controls = control
                self.realization = realization
                dat_content = self._tpl_model.substitute(SCHEDULE=self._write_scheduling())
                rwd_content = self._tpl_rwd.substitute(SR3FILE=self.model.basename.sr3.name)
                tasks.append(
                    asyncio.ensure_future(
                    self._write_dat_async(self.model.basename.dat, dat_content)
                    )
                )
                tasks.append(
                    asyncio.ensure_future(
                    self._write_dat_async(self.model.basename.rwd, rwd_content)
                    )
                )
        await asyncio.gather(*tasks)

    async def write_multiple_dat_files(self,
                                        basenames:List[str],
                                        controls:np.ndarray,
                                        realizations:List[int]):
        """Write one dat file for each control in controls list."""
        tasks = []
        for basename, control, realization in zip(basenames, controls, realizations):
            self.basename = basename
            self.controls = control
            self.realization = realization
            dat_content = self._tpl_model.substitute(N1=self._realization,
                                            SCHEDULE=self._write_scheduling())
            tasks.append(
                asyncio.ensure_future(
                self._write_dat_async(self.model.basename.dat, dat_content)
                )
            )
        await asyncio.gather(*tasks)

    async def write_multiple_rwd_files(self, basenames:List[str]):
        """Write one rwd file for each control in list."""
        tasks = []
        for basename in zip(basenames):
            self.basename = basename
            rwd_content = self._tpl_rwd.substitute(SR3FILE=self.model.basename.sr3.name)
            tasks.append(
                asyncio.ensure_future(
                self._write_dat_async(self.model.basename.rwd, rwd_content)
                )
            )
        await asyncio.gather(*tasks)

    def _read_rwd_tpl(self):
        with open(self.model.tpl_report, 'r', encoding='utf-8') as tmpl:
            return Template(tmpl.read())


    def run_local_imex(self) -> int:
        """call IMEX + Results Report."""
        # self.create_well_operation()
        try:
            with open(self.model.basename.log, 'w', encoding='UTF-8') as log:
                sim_command = [self.cmginfo.sim_exe, "-f",
                                self.model.basename.dat.absolute(), "-wait", "-dd"]
                if self.opt.parasol > 1:
                    sim_command += ["-parasol", str(self.opt.parasol), "-doms"]
                return subprocess.run(sim_command, stdout=log, check=True)
        except subprocess.CalledProcessError as error:
            sys.exit(
                f"Error - Não foi possível executar o IMEX, verificar: {error}")

    def run_local_report_results(self):
        """Get production for results."""
        try:
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

    def get_mult_npvs(self, rwo_files:List[str]):
        """Retorna uma lista de valores de npv, um para cada rwo file."""
        rwo_files = [rwo for rwo in rwo_files if rwo.endswith('.rwo')]
        periodic_rate = ((1 + self.opt.tma) ** (1 / 365.25)) - 1
        col_name = ['time', 'Qo', 'Qgp', 'Qwp', 'Qwi', 'Empt_col']
        rwo_names = []
        queries = []
        for rwo in rwo_files:
            rwo_names.append(Path(rwo).name)
            dframe = pl.scan_csv(rwo,
                            separator='\t',
                            skip_rows=6,
                            truncate_ragged_lines=True,
                            has_header=False,
                            new_columns=col_name
                            )
            queries.append(
                dframe
                .with_columns(
                    [
                    (pl.col('Qo') * self.opt.prices[0]
                    + pl.col('Qgp') * self.opt.prices[1]
                    + pl.col('Qwp') * self.opt.prices[2]
                    + pl.col('Qwi') * self.opt.prices[3]).alias('cf'),
                    (1 / np.power((1 + periodic_rate),
                                    pl.col('time'))).alias('tax'),
                    ]
                )
                .with_columns(pv = pl.col('cf') * pl.col('tax') * 1e-9)
                .select('pv').sum()
            )
        npvs = pl.concat(queries).collect()
        npvs = npvs.with_columns(pl.Series(name='rwo', values=rwo_names))
        return npvs.select(['rwo', 'pv'])

    def clean_up_temp_run(self, ignore_files=None):
        """ Clean files from run path."""
        if ignore_files is None:
            files = [self.model.temp_run / file for file in os.listdir(self.model.temp_run)]
        else:
            files = [self.model.temp_run / file for file
                      in os.listdir(self.model.temp_run)
                      if Path(file).stem not in ignore_files]
        nworkers = 6
        chunksize = round(len(files) / nworkers)
        with ThreadPoolExecutor(nworkers) as exe:
            for i in range(0, len(files), chunksize):
                filenames = files[i:(i + chunksize)]
                _ = exe.submit(delete_files, filenames)
        print('Temp folders is clean.')

    def _parse_include_files(self, datafile):
        """Parse simulation file for *INCLUDE files and return a list."""
        with open(datafile, "r", encoding='UTF-8') as file:
            lines = file.read()
        pattern = r'\n\s*\*?include\s*[\'|"](.*)[\'|"]'
        return re.findall(pattern, lines, flags=re.IGNORECASE)