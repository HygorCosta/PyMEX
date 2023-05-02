""" Parallel computation. """
import multiprocessing as mp
import numpy as np
from .mani_param import PyMEX


class ParallelPyMex:

    """Run imex in parallel for reservoir optimization."""

    def __init__(self, config_file, pool_size=None):
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
        self.config_file = config_file
        self.pool_size = pool_size

    def run(self, controls):
        """Run PyMEX in a sequential way."""
        model = PyMEX(controls, self.config_file)
        model()
        return model.npv

    def pool_run(self, controls):
        """Run imex in parallel.

        Parameters
        ----------
        pool_size: int
            Number of process.

        """
        if self.pool_size and controls.ndim > 1:
            with mp.Pool(self.pool_size) as proc:
                npv = proc.map(self.run, controls.tolist())
        else:
            npv = self.run(controls)
        return np.array(npv)
