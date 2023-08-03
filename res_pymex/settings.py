
class Model():
    def __init__(self):
        self.tpl = ""
        self.temp_run = ""
        self.basename = ""
        self.schedule = ""
        self.tpl_report = ""

class Wells():
    def __init__(self) -> None:
        self.control_type = ''
        self.prod = []
        self.prod_operate = []
        self.inj = []
        self.inj_operate = []
        self.max_plat = []

class Optimization():
    def __init__(self) -> None:
        self.numb_cic = None
        self.tma = None
        self.prices = []
        self.parasol = None
        self.clean_files = None
