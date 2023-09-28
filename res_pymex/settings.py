"""Python package to settings reservoir information."""
import math
from pathlib import Path
import numpy as np
import yaml
from .util import cmgfile

class Model():
    """Model information."""
    def __init__(self):
        self.tpl = ""
        self.basename = ""
        self.temp_run = ""
        self.results = ""
        self.tpl_report = ""

class Wells():
    """Wells information."""
    def __init__(self) -> None:
        self.control_type = ''
        self.prod = []
        self.prod_operate = []
        self.inj = []
        self.inj_operate = []
        self.max_plat = []

class Optimization():
    """Optimization information."""
    def __init__(self) -> None:
        self.numb_cic = None
        self.realizations = None
        self.tma = None
        self.prices = []
        self.dates = []
        self.max_sim = None
        self.sao = []


class Settings:
    """Settings reservoir information to run."""

    def __init__(self, reservoir_config:str) -> None:
        with open(reservoir_config, "r", encoding='UTF-8') as yamlfile:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
            self.model = self.import_model(data)
            self.wells = self.import_wells_operate(data)
            self.opt = self.import_opt_settings(data)

    @staticmethod
    def import_model(data_yaml):
        """Set model information."""
        new_model = Model()
        data_yaml = data_yaml['model']
        new_model.tpl = Path(data_yaml['dat_tpl'])
        new_model.temp_run = Path(data_yaml['temp_run'])
        new_model.temp_run.mkdir(exist_ok=True)
        new_model.results = Path(data_yaml['results'])
        new_model.results.mkdir(exist_ok=True)
        new_model.basename = cmgfile(new_model.temp_run / new_model.tpl.name)
        new_model.tpl_report = Path(data_yaml['report_tpl'])
        return new_model

    @staticmethod
    def import_wells_operate(data_yaml):
        """Set well information."""
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
    def import_opt_settings(data_yaml):
        """Set opt information."""
        new_opt = Optimization()
        data_yaml = data_yaml['optimization']
        new_opt.dates = data_yaml['dates']
        new_opt.realizations = data_yaml['realizations']
        new_opt.numb_cic = math.ceil(12*new_opt.dates['conc']/new_opt.dates['step_control'])
        new_opt.tma = data_yaml['tma']
        new_opt.prices = data_yaml['prices']
        new_opt.max_sim = data_yaml['max_sim']
        new_opt.sao = data_yaml['sao']
        return new_opt
