import json
import argparse
import subprocess
from types import SimpleNamespace
import itertools

def get_argparser():
    argparser = argparse.ArgumentParser(description='Experiment Runner')
    argparser.add_argument('--json', required=True, type=str)
    argparser.add_argument('-no-repeat', action='store_true')
    return argparser


# Using Subprocess instead of single thread to isolate bluetooth errors
def experiment(seconds, model, alpha, url, norepeat, retry=True):
    while True:
        run_string = "python3 run_single.py --seconds {}".format(seconds)
        if len(model)>0:
            run_string += " --model {} --alpha {}".format(model, alpha)
        if len(url) > 0:
            run_string += " --url {}".format(url)
        if norepeat:
            run_string += " -no-repeat"

        print("Experiment: {}".format(run_string))
        proc = subprocess.run(run_string, shell=True, check=True)
        if proc.returncode == 0:
                return proc

        if not retry:
            continue
        else:
            print("Experiment Failed Trying again")

if __name__ == "__main__":
    args = get_argparser().parse_args()
    with open(args.json, "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    # Run base Experiment
    experiment(seconds=config.seconds, model="", alpha=1.0, url="", norepeat=args.no_repeat)

    all_configs = list(itertools.product([config.seconds],
                                         config.model,
                                         config.alpha,
                                         ["", config.url],
                                         [args.no_repeat]))
    for config in all_configs:
        experiment(*config)

    print("##################################################")
    print("########     ALL EXPERIMENTS FINISHED     ########")
    print("##################################################")





