from res_pymex import ImexTools

def create_instance():
    controls = [1] * 58
    return ImexTools(controls, 'example/config.yaml')

def test_res_param():
    tools = create_instance()