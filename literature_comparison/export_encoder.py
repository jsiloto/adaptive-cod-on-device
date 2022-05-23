import argparse
import os

from lee2021.wrapper import Lee2021
from base_wrapper import BaseWrapper
from matsubara2022.wrapper import Matsubara2022

wrapper_dict = {
    "lee2021": Lee2021,
    "matsubara2022": Matsubara2022,
}


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Runner')
    argparser.add_argument('--model', required=True, help='model name')
    argparser.add_argument('--config', help='String representing Computation/Bandwidth configuration')
    return argparser


def print_available_options():
    print("Model Not Found - Avilable Model Options: [Configs]")
    for k, v in wrapper_dict.items():
        print("{}: {}".format(k, v.get_config_options()))


def eval_single_model(model, config):
    model_class = wrapper_dict[model]
    wrapper: BaseWrapper = model_class(config=config)
    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)
    out_file = wrapper.generate_torchscript(out_dir)
    if not os.path.exists(out_file):
        raise RuntimeError("Missing {}".format(out_file))
    out_file = wrapper.generate_metrics(out_dir)
    if not os.path.exists(out_file):
        raise RuntimeError("Missing {}".format(out_file))
    print("Done Evaluating {}".format(wrapper.get_printname()))

def eval_all_models(args):
    for k, v in wrapper_dict.items():
        for config in v.get_config_options():
            eval_single_model(k, config)


if __name__ == '__main__':
    args = get_argparser().parse_args()
    if args.model == "all":
        eval_all_models(args)
    elif args.model in wrapper_dict:
        eval_single_model(args.model, args.config)
    else:
        print_available_options()
