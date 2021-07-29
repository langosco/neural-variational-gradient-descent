import os
import json_tricks as json
import argparse
from nvgd.src import utils
from nvgd.experiments import config as cfg

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action='store_true')
parser.add_argument("--results_path", type=str,
                            default=cfg.results_path + "nvgd-sweep/")
parser.add_argument("--all_results_filename", type=str,
                            default="all_sweep_results.json")
args = parser.parse_args()

logpath = args.results_path + "log/"
all_results_file = args.results_path + args.all_results_filename
if args.debug:
    all_results_file += '--debug'
print(f"Parsing results from {logpath}.")

# read all dictionaries from results
results = []
for sweep_result in os.listdir(logpath):
    results.append(json.load(os.path.join(logpath, sweep_result)))


print(f"Saving all results to {all_results_file}.")
json.dump(utils.dict_concatenate(results), 
          all_results_file,
          allow_nan=True,
          indent=4)

