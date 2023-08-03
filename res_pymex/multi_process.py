""" Parallel computation. """
import multiprocessing as mp
from itertools import repeat

import numpy as np

from .mani_param import PyMEX


class ParallelPyMex:

    """Run imex in parallel for reservoir optimization."""

    def __init__(self, reservoir_dat, pool_size:int = 4):
        """TODO: to be defined.

        Parameters
        ----------
        controls: array
            Solution candidates for design variables.
        res_param: dictionary
            Reservoir parameters
        pool_size: int
            Pool size for multiprocessing, if pool_size is None
            so it's sequential.

        """
        self.reservoir_dat = reservoir_dat
        self.pool_size = pool_size

    def net_present_value(self, controls, realization:int = None):
        """Evaluate the npv foe each single control"""
        model = PyMEX(self.reservoir_dat, controls, realization)
        return model.npv()

    def pool_run(self, controls, realization:int = None):
        """Run imex in parallel.
        Parameters
        ----------
        pool_size: int
            Number of process.

        """
        if self.pool_size and controls.ndim > 1:
            if realization is not None:
                with mp.Pool(self.pool_size) as proc:
                    npv = proc.starmap(self.net_present_value, zip(controls,
                                                                realization))
            else:
                with mp.Pool(self.pool_size) as proc:
                    npv = proc.starmap(self.net_present_value, controls)
        else:
            npv = self.net_present_value(controls)
        return np.array(npv)
