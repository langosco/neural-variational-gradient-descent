# Train a Bayesian neural network to classify data using
# (parallel) Langevin dynamics
import os
import argparse
import jax.flatten_util
import pandas as pd
import optax
from jax import jit, value_and_grad, vmap, random
from tqdm import tqdm
from nvgd.src import utils
import config as cfg
import dataloader
import bnn

data = dataloader.data
NUM_CLASSES = 10

# Config
key = random.PRNGKey(0)
LEARNING_RATE = 1e-7
DISABLE_PROGRESS_BAR = True


def train(key,
          particle_stepsize: float = 1e-7,
          evaluate_every: int = 10,
          n_iter: int = 400,
          n_samples: int = cfg.n_samples,
          results_file: str = cfg.results_path + 'sgld-bnn.csv',
          overwrite_file: bool = False):
    """Train langevin BNN"""
    opt = utils.sgld(particle_stepsize)

    print("Initializing parameters...")
    key, subkey = random.split(key)
    param_set = vmap(bnn.model.init, (0, None))(
        random.split(subkey, n_samples), data.train_images[:5])
    opt_state = opt.init(param_set)

    # save accuracy to file
    if not os.path.isfile(results_file) or overwrite_file:
        with open(results_file, "w") as file:
            file.write("step,accuracy\n")

    @jit
    def step(param_set, opt_state, images, labels):
        """Update param_set elements in parallel using Langevin dynamics."""
        step_losses, g = vmap(value_and_grad(bnn.loss), (0, None, None))(
            param_set, images, labels)
        g, opt_state = opt.update(g, opt_state, param_set)
        return optax.apply_updates(param_set, g), opt_state, step_losses

    print("Training...")
    # training loop
    losses = []
    accuracies = []
    for step_counter in tqdm(range(n_iter), disable=DISABLE_PROGRESS_BAR):
        images, labels = next(data.train_batches)
        param_set, opt_state, step_losses = step(param_set, opt_state, images, labels)
        losses.append(step_losses)

        if step_counter % (n_iter // evaluate_every) == 0:
            acc = bnn.compute_acc(param_set)
            accuracies.append(acc)
            print(f"Step {step_counter}, Accuracy:", acc)
            print(f"Particle mean: {jax.flatten_util.ravel_pytree(param_set)[0].mean()}")
            with open(results_file, "a") as file:
                file.write(f"{step_counter},{acc}\n")

    return bnn.compute_acc(param_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100, help="Number of parallel chains")
    parser.add_argument("--n_iter", type=int, default=200)
    args = parser.parse_args()

    # 1) read stepsize from sweep csv
    # 2) run one epoch and keep track of accuracy
    # 3) save to csv
    print("Loading optimal step size")
    results_file = cfg.results_path + "sgld-bnn.csv"
    stepsize_csv = cfg.results_path + "bnn-sweep/best-stepsizes.csv"
    try:
        sweep_results = pd.read_csv(stepsize_csv, index_col=0)
        stepsize = sweep_results['optimal_stepsize']['sgld']
    except (FileNotFoundError, TypeError):
        print('CSV sweep results not found; using default')
        stepsize = 1e-3

    rngkey = random.PRNGKey(0)
    train(key=rngkey,
          n_samples=args.n_samples,
          n_iter=args.n_iter)
