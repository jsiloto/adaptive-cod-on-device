import os

from literature_models.base.base_wrapper import BaseWrapper
from literature_models.lee2021.wrapper import Lee2021
from literature_models.matsubara2022.wrapper import Matsubara2022

wrapper_dict = {
    "lee2021": Lee2021,
    "matsubara2022": Matsubara2022,
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
    model_file = wrapper.generate_torchscript(out_dir)
    if not os.path.exists(model_file):
        raise RuntimeError("Missing {}".format(model_file))
    out_file = wrapper.generate_metrics(out_dir)
    if not os.path.exists(out_file):
        raise RuntimeError("Missing {}".format(out_file))
    print("Done Evaluating {}".format(wrapper.get_printname()))
    return model_file