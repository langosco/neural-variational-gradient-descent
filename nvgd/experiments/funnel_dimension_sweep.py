import os
import argparse
import time
from pathlib import Path
import json_tricks as json
from tqdm import tqdm
import jax.numpy as jnp
from jax import jit, random
import numpy as onp

from nvgd.src import distributions, flows, kernels, metrics
import config as cfg

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

key = random.PRNGKey(args.seed)
on_cluster = not os.getenv("HOME") == "/home/lauro"

# Config
NUM_STEPS = 500  # 500
PARTICLE_STEP_SIZE = 1e-2  # for particle update
LEARNING_RATE = 1e-4  # for neural network
NUM_PARTICLES = 200  # 200
MAX_DIM = 40  # sweep from 2 to MAX_DIM
PATIENCE = 15

if args.debug:
    NUM_STEPS = 30
    NUM_PARTICLES = 5
    MAX_DIM = 5

mmd_kernel = kernels.get_rbf_kernel(1.)
mmd = jit(metrics.get_mmd(mmd_kernel))

def get_mmds(particle_list, ys):
    mmds = []
    for xs in [p.particles for p in particle_list]:
        mmds.append(mmd(xs, ys))
    return mmds


def sample(d, key, n_particles):
    target = distributions.Funnel(d)
    proposal = distributions.Gaussian(jnp.zeros(d), jnp.ones(d))
    funnel_setup = distributions.Setup(target, proposal)

    key, subkey = random.split(key)
    neural_learner, neural_particles, err1 = flows.neural_svgd_flow(subkey, funnel_setup, n_particles=n_particles, n_steps=NUM_STEPS, particle_lr=PARTICLE_STEP_SIZE, learning_rate=LEARNING_RATE, patience=PATIENCE)
    svgd_gradient, svgd_particles, err2    = flows.svgd_flow(       subkey, funnel_setup, n_particles=n_particles, n_steps=NUM_STEPS, particle_lr=PARTICLE_STEP_SIZE, scaled=True,  bandwidth=None)
    sgld_gradient, sgld_particles, err3    = flows.sgld_flow(       subkey, funnel_setup, n_particles=n_particles, n_steps=NUM_STEPS, particle_lr=PARTICLE_STEP_SIZE)
    return (neural_particles, svgd_particles, sgld_particles), (neural_learner, svgd_gradient, sgld_gradient)


print("SWEEPING DIMENSIONS...")
mmd_sweep = []
for d in tqdm(range(2, MAX_DIM), disable=on_cluster):
    key, subkey = random.split(key)
    particles, gradients = sample(d, subkey, NUM_PARTICLES)

    target = distributions.Funnel(d)
    key, subkey = random.split(key)
    ys = target.sample(NUM_PARTICLES, subkey)
    mmds = get_mmds(particles, ys)
    mmd_sweep.append(mmds)

mmd_sweep = onp.array(mmd_sweep)

results = {
    "NVGD": mmd_sweep[:, 0].tolist(),
    "SVGD":  mmd_sweep[:, 1].tolist(),
    "SGLD":  mmd_sweep[:, 2].tolist(),
}

##################
# save json results
print("SAVING RESULTS...")
if args.debug:
    results_path = Path(cfg.results_path) / "debug" / "funnel-dimension-sweep" / "runs"
else:
    results_path = Path(cfg.results_path) / "funnel-dimension-sweep" / "runs"

results_path.mkdir(parents=True, exist_ok=True)
filename = f"seed_{args.seed}.json"
if os.path.isfile(results_path / filename):
    filename = f"seed_{args.seed}_{time.time()}.json"

with open(results_path / filename, "w") as f:
    json.dump(results, f, indent=4, sort_keys=True, allow_nan=True)

print("DONE.")
