import os
import argparse
import config as cfg
import numpy as onp
import nvgd_bnn
import svgd_bnn
import sgld_bnn
from jax import random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default='all',
                    help="Which method to sweep. Can be 'nvgd', 'sgld',"
                         "'svgd', or 'all'.")
parser.add_argument("--debug", action='store_true')
parser.add_argument("--results_folder", type=str,
                    default="bnn-sweep")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--opt", type=str, default="sgd")
parser.add_argument("--hidden_sizes", nargs="*", type=int, default=[256]*3)
parser.add_argument("--use_hypernetwork", action='store_true')
parser.add_argument("--results_path", type=str, default=cfg.results_path)
args = parser.parse_args()


DEBUG = args.debug
if DEBUG:
    print("Running in debug mode")
    NUM_STEPS = 2
    n_lrs = 1
else:
    NUM_STEPS = args.steps
    n_lrs = 15

OVERWRITE_FILE = True
EVALUATE_EVERY = -1  # never

key = random.PRNGKey(0)
key, subkey = random.split(key)
results_path = args.results_path + args.results_folder + "/"
sweep_results_file = results_path + "best-stepsizes.csv"  # best LR / acc goes here
dumpfile = "/dev/null"
final_accs = []

if args.opt == "sgd":
    vgd_stepsizes = onp.logspace(start=-8, stop=-3, num=n_lrs)
else:
    vgd_stepsizes = onp.logspace(start=-5, stop=1, num=n_lrs)


sgld_stepsizes = onp.logspace(start=-9, stop=-5, num=n_lrs)


Path(results_path).mkdir(parents=True, exist_ok=True)
if not os.path.isfile(sweep_results_file) or OVERWRITE_FILE:
    with open(sweep_results_file, "w") as f:
        f.write("name,optimal_stepsize,max_val_accuracy\n")


def save_single_run(name, accuracy, step_size):
    file = results_path + name + "_runs.csv"
    if not os.path.isfile(file):
        with open(file, "w") as f:
            f.write("stepsize,accuracy\n")

    with open(file, "a") as f:
        f.write(f"{step_size},{accuracy}\n")


def save_best_run(name, accuracy_list):
    """
    return best step size and highest accuracy
    args:
        name: name of sweep to save in csv file
        accuracy_list: list with entries (accuracy, step_size)
    """
    accuracy_list = onp.array(accuracy_list)
    argmax_idx = accuracy_list[:, 0].argmax()
    max_accuracy, max_stepsize = accuracy_list[argmax_idx].tolist()

    print(f"Max accuracy {max_accuracy} achieved using step size {max_stepsize}.")
    print()

    with open(sweep_results_file, "a") as f:
        f.write(f"{name},{max_stepsize},{max_accuracy}\n")

    return max_accuracy, max_stepsize


def sweep_nvgd():
    print("Sweeping NVGD...")
    final_accs = []
    for particle_stepsize in vgd_stepsizes:
        final_acc, nvgd_rundata = nvgd_bnn.train(
            key=subkey,
            particle_stepsize=particle_stepsize,
            n_iter=NUM_STEPS,
            evaluate_every=EVALUATE_EVERY,
            overwrite_file=OVERWRITE_FILE,
            results_file=dumpfile,
            optimizer=args.opt,
            hidden_sizes=args.hidden_sizes,
            use_hypernetwork=args.use_hypernetwork
            )

        save_single_run("nvgd", final_acc, particle_stepsize)
        final_accs.append((final_acc, particle_stepsize))

    max_accuracy, nvgd_max_stepsize = save_best_run("nvgd", final_accs)


def sweep_sgld():
    print("Sweeping Langevin...")
    if args.opt != "sgd":
        print(f"Using vanilla SGLD for langevin, even though "
              "you requested adaptive optimizer {args.opt}")
    final_accs = []
    for particle_stepsize in sgld_stepsizes:
        final_acc = sgld_bnn.train(key=subkey,
                                   particle_stepsize=particle_stepsize,
                                   n_iter=NUM_STEPS,
                                   evaluate_every=EVALUATE_EVERY,
                                   results_file=dumpfile)
        final_accs.append((final_acc, particle_stepsize))
        save_single_run("sgld", final_acc, particle_stepsize)

    max_accuracy, sgld_max_stepsize = save_best_run("sgld", final_accs)


def sweep_svgd():
    print("Sweeping SVGD...")
    final_accs = []
    for particle_stepsize in vgd_stepsizes:
        final_acc = svgd_bnn.train(key=subkey,
                                   particle_stepsize=particle_stepsize,
                                   n_iter=NUM_STEPS,
                                   evaluate_every=EVALUATE_EVERY,
                                   results_file=dumpfile,
                                   optimizer=args.opt)
        final_accs.append((final_acc, particle_stepsize))
        save_single_run("svgd", final_acc, particle_stepsize)

    max_accuracy, svgd_max_stepsize = save_best_run("svgd", final_accs)


if args.run == "nvgd":
    sweep_nvgd()
elif args.run == "sgld":
    sweep_sgld()
elif args.run == "svgd":
    sweep_svgd()
elif args.run == "all":
    sweep_nvgd()
    sweep_sgld()
    sweep_svgd()
else:
    raise ValueError("cli argument 'run' must be one of 'nvgd', 'sgld',"
                     "'svgd', or 'all'.")
