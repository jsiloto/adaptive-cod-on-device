import json
import argparse
import subprocess
from types import SimpleNamespace
import itertools

def get_argparser():
    argparser = argparse.ArgumentParser(description='Experiment Runner')
    argparser.add_argument('--json', required=True, type=str)
    return argparser

if __name__ == "__main__":
    args = get_argparser().parse_args()
    with open(args.json, "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    # Using Subprocess instead of single thread to isolate bluetooth errors
    def experiment(seconds, model, alpha, url, retry=True):
        while True:
            run_string = "python3 run_single.py --seconds {}".format(seconds)
            if len(model)>0:
                run_string += " --model {} --alpha {}".format(model, alpha)
            if len(url) > 0:
                run_string += " --url {}".format(url)

            print("Experiment: {}".format(run_string))
            proc = subprocess.run(run_string, shell=True, check=True)
            if proc.returncode == 0:
                    return proc

            if not retry:
                continue
            else:
                print("Experiment Failed Trying again")

    # Run base Experiment
    experiment(seconds=config.seconds, model="", alpha=1.0, url="")

    all_configs = list(itertools.product([config.seconds], config.model, config.alpha, ["", config.url]))
    for config in all_configs:
        experiment(*config)

    print("##################################################")
    print("########     ALL EXPERIMENTS FINISHED     ########")
    print("##################################################")





