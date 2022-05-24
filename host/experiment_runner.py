import argparse

from experiment_manager.run_single import experiment
from literature_models.model_wrapper import get_all_options, eval_single_model

def get_argparser():
    argparser = argparse.ArgumentParser(description='On Device Experiments')
    argparser.add_argument('--out_dir', type=str, default='out', help='Output Directory')
    argparser.add_argument('-random', help='Run Experiments in a random order')
    argparser.add_argument('--model', help='(Optional) Run specific model')
    argparser.add_argument('--mode', help='(Optional) Run specific mode')
    argparser.add_argument('--seconds', type=int, default=60)
    return argparser

if __name__ == "__main__":
    args = get_argparser().parse_args()
    all_options = get_all_options()
    for o in all_options:
        print(o)

    for option in all_options:
        model_name, model_class, mode = option
        model_name, model_file, metrics = eval_single_model(model_class=model_class, mode=mode, out_dir=args.out_dir)
        try:
            experiment(seconds=args.seconds, model_name=model_name,
                   model_file=model_file, mode=mode,
                   url="", norepeat=False, out_dir=args.out_dir, save=False)
        except Exception as e:
            print("Failed experiment: {}".format(model_name))
            print(e)





# List all available models

# For each model
    # Build model
    # Build CSV
    # Run Experiment

# Compile all Results