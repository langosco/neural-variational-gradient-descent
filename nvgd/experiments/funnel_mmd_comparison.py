import time
import os
from jax._src.random import PRNGKey
import json_tricks as json
import jax.numpy as jnp
from jax import random
from nvgd.src import distributions, flows, metrics
from pathlib import Path
import config as cfg
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

key = random.PRNGKey(args.seed)

# config
if args.debug:
    NUM_STEPS = 50
    NUM_PARTICLES = 20  # 200
else:
    NUM_STEPS = 5000  # 5000
    NUM_PARTICLES = 200  # 200

PARTICLE_STEP_SIZE = 1e-2
LEARNING_RATE = 1e-4
d = 2
PATIENCE = 15  # (-1 vs. 0 makes a noticeable diff, but not large)

target = distributions.Funnel(d)
proposal = distributions.Gaussian(jnp.zeros(d), jnp.ones(d))
funnel_setup = distributions.Setup(target, proposal)
target_samples = target.sample(NUM_PARTICLES, random.PRNGKey(-1)) # use fixed key


key, subkey = random.split(key)
neural_learner, neural_particles, err1 = flows.neural_svgd_flow(
    key=subkey,
    setup=funnel_setup,
    n_particles=NUM_PARTICLES,
    n_steps=NUM_STEPS,
    particle_lr=PARTICLE_STEP_SIZE,
    compute_metrics=metrics.get_funnel_tracer(target_samples),
    catch_exceptions=False,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE,
    aux=False,
)

sgld_gradient, sgld_particles, err2 = flows.sgld_flow(
    subkey,
    funnel_setup,
    n_particles=NUM_PARTICLES,
    n_steps=NUM_STEPS,
    particle_lr=PARTICLE_STEP_SIZE,
    compute_metrics=metrics.get_funnel_tracer(target_samples),
    catch_exceptions=False,
)

sgld_gradient2, sgld_particles2, err3 = flows.sgld_flow(
    subkey,
    funnel_setup,
    n_particles=NUM_PARTICLES,
    n_steps=NUM_STEPS,
    particle_lr=PARTICLE_STEP_SIZE/5,
    compute_metrics=metrics.get_funnel_tracer(target_samples),
    catch_exceptions=False
)

svgd_gradient, svgd_particles, err4 = flows.svgd_flow(
    subkey,
    funnel_setup,
    n_particles=NUM_PARTICLES,
    n_steps=NUM_STEPS,
    particle_lr=PARTICLE_STEP_SIZE*5,
    scaled=True,
    bandwidth=None,
    compute_metrics=metrics.get_funnel_tracer(target_samples),
    catch_exceptions=False,
)

# Note: I scaled the svgd step-size (by hand) so that it is maximial while
# still converging to a low MMD.

particle_containers = (neural_particles, sgld_particles,
                       sgld_particles2, svgd_particles)
names = ("Neural", "SGLD", "SGLD2", "SVGD")
results = {name: p.rundata["rbf_mmd"].tolist()  # CHANGED from 'funnel_mmd'
           for name, p in zip(names, particle_containers)}


##################
# save json results
if args.debug:
    results_path = Path(cfg.results_path) / "debug" / "funnel-mmd-comparison" / "runs"
else:
    results_path = Path(cfg.results_path) / "funnel-mmd-comparison" / "runs"

results_path.mkdir(parents=True, exist_ok=True)
filename = f"seed_{args.seed}.json"
if os.path.isfile(results_path / filename):
    filename = f"seed_{args.seed}_{time.time()}.json"

with open(results_path / filename, "w") as f:
    json.dump(results, f, indent=4, sort_keys=True)
