import time
import argparse
import numpy as onp
import json_tricks as json
from jax import random
from pathlib import Path

from nvgd.src import utils
from nvgd.experiments import nvgd_bnn
from nvgd.experiments import config as cfg


parser = argparse.ArgumentParser()
parser.add_argument("--debug", action='store_true')
parser.add_argument("--results_path", type=str,
                            default=cfg.results_path + "nvgd-sweep/")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--opt", type=str, default="adam")
parser.add_argument("--use_hypernetwork", action='store_true')
parser.add_argument("--search_method", type=str, default="random")
parser.add_argument("--num_sweep_iter", type=int, default=100)
args = parser.parse_args()


OVERWRITE_FILE = True
NUM_STEPS = args.steps if not args.debug else 2
EVALUATE_EVERY = -1

key = random.PRNGKey(0)
key, subkey = random.split(key)

Path(args.results_path + "log/").mkdir(parents=True, exist_ok=True)

if args.search_method == "grid":
    sweep_dict = {
        "meta_lr": onp.logspace(start=-4, stop=-2, num=4),
        "particle_stepsize": onp.logspace(start=-4, stop=-2, num=6),
        "patience": [5, 20, 50], # default 5
        "max_train_steps_per_iter": [50, 100],
        "particle_steps_per_iter": [1],
        "use_hypernetwork": [args.use_hypernetwork],
        "early_stopping": [True],
        "optimizer": [args.opt],
    }
    sweep_kwargs = utils.dict_cartesian_product(**sweep_dict)
    print("SETTING SWEEP CONFIG:")
    print(json.dumps(sweep_dict, indent=4))
    print()
elif args.search_method == "random":
    def sample_kwargs():
        return {
            "meta_lr": 10**onp.random.uniform(-8, -1), # (-5, 0)
            "particle_stepsize": 10**onp.random.uniform(-5, -2),
            "patience": onp.random.choice(50)+1, # default 5
            "max_train_steps_per_iter": onp.random.choice(range(10, 100)),
            "particle_steps_per_iter": onp.random.choice(10)+1,
            "use_hypernetwork": args.use_hypernetwork,
            "early_stopping": True,
            "dropout": onp.random.choice([True, False]),
            "optimizer": args.opt,
        }

    def generate_sweep_kwargs(num):
        for i in range(num):
            yield sample_kwargs()

    sweep_kwargs = generate_sweep_kwargs(num=args.num_sweep_iter)
else:
    raise ValueError


if args.search_method == "random":
    print(f'Running sweep for {args.num_sweep_iter} steps.')
    print()

outcomes = []
key, subkey = random.split(key)
for param_setting in sweep_kwargs:
    print()
    print("Training with setting:")
    print(json.dumps(param_setting, allow_nan=True, indent=4))

    final_acc, _ = nvgd_bnn.train(key=subkey,
                                  n_iter=NUM_STEPS,
                                  evaluate_every=EVALUATE_EVERY,
                                  overwrite_file=OVERWRITE_FILE,
                                  results_file="/dev/null",
                                  **param_setting)
    param_setting.update({"accuracy": final_acc})
    outcomes.append(param_setting)
    filename = args.results_path + f"log/acc_{final_acc}_at_{time.time()}.json"

    print("...done. Accuracy:", final_acc)
    print(f"Saving results to {filename}")
    json.dump(param_setting, filename, allow_nan=True, indent=4)
    print("---------------------------------------")

argmax_idx = onp.array([par['accuracy'] for par in outcomes]).argmax()
max_accuracy = outcomes[argmax_idx]['accuracy']

print("---------------------------------------")
print()
print("SWEEP DONE")
print()
print(f"Max accuracy of {max_accuracy} achieved using setting")
print(json.dumps(outcomes[argmax_idx], indent=4))