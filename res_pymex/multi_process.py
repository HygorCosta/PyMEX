""" Parallel computation. """
import multiprocessing as mp
from itertools import repeat

import numpy as np

from .mani_param import PyMEX


class ParallelPyMex:

    """Run imex in parallel for reservoir optimization."""

    def __init__(self, reservoir_dat, pool_size=0.5*mp.cpu_count()):
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

    def net_present_value(self, controls, prices, tma):
        """Evaluate the npv foe each single control"""
        model = PyMEX(self.reservoir_dat, controls)
        return model.npv(prices, tma)

    def pool_run(self, controls, prices, tma_aa):
        """Run imex in parallel.
        Parameters
        ----------
        pool_size: int
            Number of process.

        """
        if self.pool_size and controls.ndim > 1:
            with mp.Pool(self.pool_size) as proc:
                npv = proc.starmap(self.net_present_value, zip(
                    controls, repeat(prices), repeat(tma_aa)))
        else:
            npv = self.net_present_value(controls, prices, tma_aa)
        return np.array(npv)
