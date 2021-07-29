from jax import random, jit, vmap, grad
import os
import warnings
from tqdm import tqdm
from nvgd.src import metrics, kernels, models, utils

default_num_particles = 50
default_num_steps = 100
# default_particle_lr = 1e-1
# default_learner_lr = 1e-2
NUM_WARMUP_STEPS = 500
on_cluster = not os.getenv("HOME") == "/home/lauro"
disable_tqdm = on_cluster

#raise NotImplementedError("Need to update call to SteinNetwork to match "
#                          "the updated argument signature in flows.py")

def neural_svgd_flow(key,
                     setup,
                     n_particles=default_num_particles,
                     n_steps=default_num_steps,
                     particle_lr=1e-2,
                     compute_metrics=None,
                     n_learner_steps=50,
                     catch_exceptions: bool = True,
                     **learner_kwargs):
    """
    args:
        learning_rate: meta-learning rate of gradient learner model
    """
    key, keya, keyb, keyc = random.split(key, 4)
    target, proposal = setup.get()
    learner = models.SteinNetwork(key=keya,
                                  target_dim=target.d,
                                  **learner_kwargs)

    if compute_metrics is None:
        compute_metrics = metrics.get_mmd_tracer(target.sample(500, keyc))
    particles = models.Particles(key=keyb,
                                 gradient=learner.gradient,
                                 init_samples=proposal.sample,
                                 n_particles=n_particles,
                                 learning_rate=particle_lr,
                                 optimizer="sgd",
                                 compute_metrics=compute_metrics)

    vdlogp = jit(vmap(grad(target.logpdf)))
    for _ in tqdm(range(n_steps), disable=disable_tqdm):
        try:
            key, subkey = random.split(key)
            split_particles = particles.next_batch(
                subkey,
                n_train_particles=2*n_particles//3
            )
            split_dlogp = [vdlogp(x) for x in split_particles]
            learner.train(split_particles=split_particles,
                          split_dlogp=split_dlogp,
                          n_steps=n_learner_steps)
            particles.step(learner.get_params())
        except Exception as err:
            if catch_exceptions:
                warnings.warn("Caught Exception")
                return learner, particles, err
            else:
                raise err
    particles.done()
    return learner, particles, None


def svgd_flow(key,
              setup,
              n_particles=default_num_particles,
              n_steps=default_num_steps,
              particle_lr=1e-1,
              lambda_reg=1/2,
              particle_optimizer="sgd",
              scaled=True,
              bandwidth=1.,
              compute_metrics=None,
              catch_exceptions: bool = True,
              ):
    key, keyb, keyc = random.split(key, 3)
    target, proposal = setup.get()

    kernel_gradient = models.KernelGradient(target_logp=target.logpdf,
                                            kernel=kernels.get_rbf_kernel,
                                            bandwidth=bandwidth,
                                            lambda_reg=lambda_reg,
                                            scaled=scaled)

    if compute_metrics is None:
        compute_metrics = metrics.get_mmd_tracer(target.sample(500, keyc))
    svgd_particles = models.Particles(key=keyb,
                                      gradient=kernel_gradient.gradient,
                                      init_samples=proposal.sample,
                                      n_particles=n_particles,
                                      learning_rate=particle_lr,
                                      optimizer=particle_optimizer,
                                      compute_metrics=compute_metrics)
    for _ in tqdm(range(n_steps), disable=disable_tqdm):
        try:
            svgd_particles.step(None)
        except Exception as err:
            if catch_exceptions:
                warnings.warn("caught error!")
                return kernel_gradient, svgd_particles, err
            else:
                raise err
    svgd_particles.done()
    return kernel_gradient, svgd_particles, None


def sgld_flow(key,
              setup,
              n_particles=default_num_particles,
              n_steps=default_num_steps,
              particle_lr=1e-2,
              lambda_reg=1/2,
              custom_optimizer=None,
              compute_metrics=None,
              catch_exceptions: bool = True):
    keya, keyb, keyc = random.split(key, 3)
    target, proposal = setup.get()
    energy_gradient = models.EnergyGradient(target.logpdf, lambda_reg=lambda_reg)
    if custom_optimizer is None:
        seed = keyc[1] # TODO this is an ugly hack
        custom_optimizer = utils.sgld(particle_lr, seed)

    if compute_metrics is None:
        compute_metrics = metrics.get_mmd_tracer(target.sample(500, keya))
    particles = models.Particles(key=keyb,
                                 gradient=energy_gradient.gradient,
                                 init_samples=proposal.sample,
                                 n_particles=n_particles,
                                 learning_rate=particle_lr,
                                 custom_optimizer=custom_optimizer,
                                 compute_metrics=compute_metrics)
    for _ in tqdm(range(n_steps), disable=disable_tqdm):
        try:
            particles.step(None)
        except Exception as err:
            if catch_exceptions:
                warnings.warn("Caught and returned exception")
                return energy_gradient, particles, err
            else:
                raise err
    particles.done()
    return energy_gradient, particles, None
