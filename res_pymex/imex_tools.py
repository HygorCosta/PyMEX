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
from collections import namedtuple
from datetime import datetime


def cmgfile(file, sufix:int=None):
    """
    A simple wrapper for retrieving CMG file extensions
    given the basename.
    :param basename:
    :return:
    """
    if sufix:
        basename = file.parent / f'{file.stem}_{sufix:03d}'
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = file.parent / f'{file.stem}_{current_time}'
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
