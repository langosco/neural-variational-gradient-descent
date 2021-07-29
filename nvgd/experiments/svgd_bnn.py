import os
import argparse
from jax import vmap, random
from tqdm import tqdm
import optax
from nvgd.src import models, metrics
import bnn
import dataloader
import config as cfg
import pandas as pd

data = dataloader.data

# Config
# date = datetime.today().strftime('%a-%H:%M-%f')
DISABLE_PROGRESS_BAR = True
LAMBDA_REG = 10**2


def train(key,
          particle_stepsize: float = 1e-3,
          evaluate_every: int = 10,
          n_iter: int = 400,
          n_samples: int = 100,
          results_file: str = cfg.results_path + 'svgd-bnn.csv',
          overwrite_file: bool = False,
          optimizer="sgd"):
    """
    Initialize model; warmup; training; evaluation.
    Returns a dictionary of metrics.
    Args:
        particle_stepsize: learning rate of BNN
        evaluate_every: compute metrics every `evaluate_every` steps
        n_iter: number of train-update iterations
        write_results_to_file: whether to save accuracy in csv file
    """
    csv_string = f"{particle_stepsize}"

    # initialize particles and the dynamics model
    key, subkey = random.split(key)
    init_particles = vmap(bnn.init_flat_params)(random.split(subkey, n_samples))

    if optimizer == "sgd":
        opt = optax.sgd(particle_stepsize)
    elif optimizer == "adam":
        opt = optax.adam(particle_stepsize)
    else:
        raise ValueError("must be adam or sgd")

    key, subkey1, subkey2 = random.split(key, 3)
    svgd_grad = models.KernelGradient(get_target_logp=bnn.get_minibatch_logp,
                                      scaled=False,
                                      lambda_reg=LAMBDA_REG)

    particles = models.Particles(key=subkey2,
                                 gradient=svgd_grad.gradient,
                                 init_samples=init_particles,
                                 custom_optimizer=opt)

    def evaluate(step_counter, ps):
        stepdata = {
            "accuracy": bnn.compute_acc_from_flat(ps),
            "step_counter": step_counter,
        }
        with open(results_file, "a") as file:
            file.write(csv_string + f"{step_counter},{stepdata['accuracy']}\n")
        return stepdata

    if not os.path.isfile(results_file) or overwrite_file:
        with open(results_file, "w") as file:
            file.write("meta_lr,particle_stepsize,patience,"
                       "max_train_steps,step,accuracy\n")

    print("Training...")
    for step_counter in tqdm(range(n_iter), disable=DISABLE_PROGRESS_BAR):
        train_batch = next(data.train_batches)
        particles.step(train_batch)

        if (step_counter+1) % evaluate_every == 0:
            metrics.append_to_log(particles.rundata,
                                  evaluate(step_counter, particles.particles))

        if step_counter % data.steps_per_epoch == 0:
            print(f"Starting epoch {step_counter // data.steps_per_epoch + 1}")

    # final eval
    final_eval = evaluate(-1, particles.particles)
    particles.done()

    return final_eval['accuracy']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100, help="Number of parallel chains")
    parser.add_argument("--n_iter", type=int, default=200)
    args = parser.parse_args()

    # 1) read stepsize from sweep csv
    # 2) run one epoch and keep track of accuracy
    # 3) save to csv
    print("Loading optimal step size")
    results_file = cfg.results_path + "svgd-bnn.csv"
    stepsize_csv = cfg.results_path + "bnn-sweep/best-stepsizes.csv"
    try:
        sweep_results = pd.read_csv(stepsize_csv, index_col=0)
        stepsize = sweep_results['optimal_stepsize']['svgd']
    except (FileNotFoundError, TypeError):
        print('CSV sweep results not found; using default')
        stepsize = 1e-3

    rngkey = random.PRNGKey(0)
    train(key=rngkey,
          n_samples=args.n_samples,
          n_iter=args.n_iter)
