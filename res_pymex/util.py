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
# """
import os
from pathlib import Path
from collections import namedtuple

def cmgfile(basename):
    """
    A simple wrapper for retrieving CMG file extensions
    given the basename.
    :param basename:
    :return:
    """
    # if sufix:
    #     basename = file.parent / f'{file.stem}_{sufix:03d}'
    # else:
    #     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     basename = file.parent / f'{file.stem}_{current_time}'
    Extension = namedtuple(
        "Extension",
        "dat out rwd rwo log sr3 schedule",
    )
    basename = Extension(
        basename.with_suffix(".dat"),
        basename.with_suffix(".out"),
        basename.with_suffix(".rwd"),
        basename.with_suffix(".rwo"),
        basename.with_suffix(".log"),
        basename.with_suffix(".sr3"),
        basename.parent / f'{basename.stem}_SCHEDULE.inc',
    )
    return basename


def delete_files(filepaths):
    """Delete files inside filepath folder"""
    for filepath in filepaths:
        os.remove(filepath)
    print('Temp_run folder is clean')


def get_cmginfo():
    """Get CMG Simulatior Information and Executables."""
    try:
        cmginfo = namedtuple("cmginfo", "home_path sim_exe report_exe")
        cmg_home = os.environ['CMG_HOME']
        simulator = list(Path(cmg_home).rglob('mx*.exe'))
        sim_exe = sorted(simulator)[-1]
        report = list(Path(cmg_home).rglob('report*.exe'))
        report_exe = sorted(report)[-1]
        return cmginfo(cmg_home, sim_exe, report_exe)
    except KeyError as error:
        raise KeyError(
            'Verifique se a variável de ambiente CMG_HOME existe!') from error
