# Maximize the stein discrepancy, keeping distributions p and q fixed. Compare
# neural stein discrepancy, kernelized SD, and theoretical optimum.
from functools import partial
from jax import grad, jit, random, vmap
from tqdm import tqdm
import jax.numpy as jnp
import numpy as onp
import json_tricks as json

import utils
import stein
import kernels
import distributions
import models
import config as cfg

key = random.PRNGKey(0)

# Poorly conditioned Gaussian
d = 50
variances = jnp.logspace(-5, 0, num=d)
target = distributions.Gaussian(jnp.zeros(d), variances)
proposal = distributions.Gaussian(jnp.zeros(d), jnp.ones(d))


@partial(jit, static_argnums=1)
def get_sd(samples, fun):
    """Compute SD(samples, p) given witness function fun"""
    return stein.stein_discrepancy(samples, target.logpdf, fun)


def kl_gradient(x):
    """Optimal witness function."""
    return grad(lambda x: target.logpdf(x) - proposal.logpdf(x))(x)


print("Computing theoretically optimal Stein discrepancy...")
sds = []
for _ in range(100):
    samples = proposal.sample(100)
    sds.append(get_sd(samples, kl_gradient))


# Neural SD
print("Computing neural Stein discrepancy...")
key, subkey = random.split(key)
learner = models.SteinNetwork(target_dim=d,
                           key=subkey,
                           learning_rate=1e-2,
                           patience=-1)
BATCH_SIZE = 1000


def sample(key):
    return proposal.sample(BATCH_SIZE*2, key).split(2)


key, subkey = random.split(key)
split_particles = sample(subkey)
split_dlogp = [vmap(grad(target.logpdf))(x) for x in split_particles]
for _ in tqdm(range(900)):
    key, subkey = random.split(key)
    learner.step(*split_particles,
                 *split_dlogp)

learner.done()


# Kernelized SD (using Gaussian kernel w/ median heuristic)
@jit
def compute_scaled_ksd(samples):
    kernel = kernels.get_rbf_kernel(kernels.median_heuristic(samples))
    phi = stein.get_phistar(kernel, target.logpdf, samples)
    ksd = stein.stein_discrepancy(samples, target.logpdf, phi)
    return ksd**2 / utils.l2_norm_squared(samples, phi)


print("Computing kernelized Stein discrepancy...")
ksds = []
for _ in tqdm(range(100)):
    samples = proposal.sample(400)
    ksds.append(compute_scaled_ksd(samples))


print("Saving results...")
# save json results
results = {
    "KSD": onp.mean(ksds).tolist(),
    "Optimal_SD": onp.mean(sds).tolist(),
    "Neural_SD": learner.rundata["sd"].tolist(),
}

with open(cfg.results_path + "sd_maxing.json", "w") as f:
    json.dump(results, f, indent=4, sort_keys=True)
