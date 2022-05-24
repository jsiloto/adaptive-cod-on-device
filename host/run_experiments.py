import argparse
import json
import random
import shutil
import time
import traceback
from copy import copy

from experiment_manager.run_single import experiment
from literature_models.model_wrapper import get_all_options, eval_single_model


def get_argparser():
    argparser = argparse.ArgumentParser(description='On Device Experiments')
    argparser.add_argument('--out_dir', type=str, default='out', help='Output Directory')
    argparser.add_argument('-random', action='store_true', help='Run Experiments in a random order')
    argparser.add_argument('-clean', action='store_true', help='Remove all previous experiments')
    argparser.add_argument('--repeats', type=int, default=0, help='Repeat N times all experiments')
    argparser.add_argument('--model', help='(Optional) Run specific model')
    argparser.add_argument('--mode', help='(Optional) Run specific mode')
    argparser.add_argument('--seconds', type=int, default=60)
    return argparser


if __name__ == "__main__":
    args = get_argparser().parse_args()
    all_options = get_all_options()

    option_copy = copy(all_options)
    for i in range(args.repeats):
        all_options += option_copy

    if args.random:
        random.shuffle(all_options)

    if args.clean:
        shutil.rmtree(args.out_dir)

    for idx, option in enumerate(all_options):
        print()
        print("############################################################")
        print("Running Experiment {}/{}".format(idx+1, len(all_options)))
        model_name, model_class, mode = option
        model_name, model_file, metrics = eval_single_model(model_class=model_class, mode=mode, out_dir=args.out_dir)
        try:
            experiment_data = experiment(seconds=args.seconds, model_name=model_name,
                                         model_file=model_file, mode=mode,
                                         url="", norepeat=False, out_dir=args.out_dir, save=False)

            experiment_data.update(metrics)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            experiment_name = model_name + "-" + timestr
            with open(args.out_dir + "/" + experiment_name + ".json", 'w') as f:
                json.dump(experiment_data, f)

        except Exception as e:
            print("Failed experiment: {}".format(model_name))
            print(e)
            print(traceback.format_exc())

