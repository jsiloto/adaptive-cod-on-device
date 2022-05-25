import os

from literature_models.base.base_wrapper import BaseWrapper
from literature_models.base.dummy_wrapper import Dummy
from literature_models.lee2021.wrapper import Lee2021
from literature_models.matsubara2022.wrapper import Matsubara2022

wrapper_dict = {
    # "dummy": Dummy,
    "lee2021": Lee2021,
    # "matsubara2022": Matsubara2022,
}

def get_all_options():
    all_options = []
    for k, v in wrapper_dict.items():
        for mode in v.get_mode_options():
            all_options.append((k, v, mode))
    return all_options


def eval_single_model(model_class, mode, out_dir='output'):
    wrapper: BaseWrapper = model_class(mode=mode)
    os.makedirs(out_dir, exist_ok=True)
    model_name = wrapper.get_printname()
    model_file = wrapper.generate_torchscript(out_dir)
    metrics = wrapper.generate_metrics()
    print("Done Evaluating {}".format(model_name))
    return model_name, model_file, metrics