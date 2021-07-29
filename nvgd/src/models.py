import jax.numpy as np
import jax.numpy as jnp
from jax import jit, vmap, random, value_and_grad, grad
import haiku as hk
import optax

from tqdm import tqdm
from functools import partial
import warnings
from typing import Mapping
import os

from . import utils, metrics, stein, kernels, nets

on_cluster = not os.getenv("HOME") == "/home/lauro"
disable_tqdm = on_cluster

"""
This file implements methods that simulate different kinds of particle dynamics.
It is structured as follows:

The class `Particles` acts as container for the particle positions and associated data.
Any update rule can be 'plugged in' by supplying the `gradient` argument. The following
update rules are implemented here:
- `SteinNetwork`: the method developed in this project, which dynamically learns a trajectory using a neural network.
- `KernelGradient`: simulates SVGD dynamics
- `EnergyGradient`: simulates Langevin dynamics

Finally, the mixin classes `VectorFieldMixin` and `EBMMixin` define different constraints on the neural update rule.
"""


class Patience:
    """Criterion for early stopping"""
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.time_waiting = 0
        self.min_validation_loss = None
        self.disable = patience == -1

    def update(self, validation_loss):
        """Returns True when early stopping criterion (validation loss
        failed to decrease for `self.patience` steps) is met"""
        if self.min_validation_loss is None or self.min_validation_loss > validation_loss:
            self.min_validation_loss = validation_loss
            self.time_waiting = 0
        else:
            self.time_waiting += 1
        return

    def out_of_patience(self):
        return (self.time_waiting > self.patience) and not self.disable

    def reset(self, patience=None):
        self.time_waiting = 0
        self.min_validation_loss = None
        if patience:
            self.patience = patience


class Particles:
    """
    Container class for particles, particle optimizer,
    particle update step method, and particle metrics.
    """
    def __init__(self,
                 key,
                 gradient: callable,
                 init_samples,
                 learning_rate=1e-2,
                 optimizer="sgd",
                 custom_optimizer=None,
                 n_particles: int = 50,
                 compute_metrics=None):
        """
        Args:
            gradient: takes in args (params, key, particles) and returns
        an array of shape (n, d). Used to compute particle update x = x + eps * gradient(*args)
            init_samples: either a callable sample(num_samples, key), or an array
        of shape (n, d) containing initial samples.
            learning_rate: scalar step-size for particle updates
            compute_metrics: callable, takes in particles as array of shape (n, d) and
        outputs a dict shaped {'name': metric for name, metric in
        zip(names, metrics)}. Evaluated once every 50 steps.
        """
        self.gradient = gradient
        self.n_particles = n_particles
        self.threadkey, subkey = random.split(key)
        self.init_samples = init_samples
        self.particles = self.init_particles(subkey)

        # optimizer for particle updates
        if custom_optimizer:
            self.optimizer_str = "custom"
            self.learning_rate = None
            self.opt = custom_optimizer
        else:
            self.optimizer_str = optimizer
            self.learning_rate = learning_rate
            self.opt = utils.optimizer_mapping[optimizer](learning_rate)
        self.optimizer_state = self.opt.init(self.particles)
        self.step_counter = 0
        self.rundata = {}
        self.donedone = False
        self.compute_metrics = compute_metrics

    def init_particles(self, key=None):
        """Returns an jnp.ndarray of shape (n, d) containing particles."""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        if callable(self.init_samples):
            particles = self.init_samples(self.n_particles, key)
        else:
            particles = self.init_samples
            self.n_particles = len(particles)
        self.d = particles.shape[1]
        return particles

    def get_params(self):
        return self.particles

    def next_batch(self,
                   key,
                   n_train_particles: int = None,
                   n_val_particles: int = None):
        """
        Return next subsampled batch of training particles (split into training
        and validation) for the training of a gradient field approximator.
        """
        particles = self.get_params()
        shuffled_batch = random.permutation(key, particles)

        if n_train_particles is None:
            if n_val_particles is None:
                n_val_particles = self.n_particles // 4
            n_train_particles = jnp.clip(self.n_particles - n_val_particles, 0)
        elif n_val_particles is None:
            n_val_particles = jnp.clip(self.n_particles - n_train_particles, 0)

        assert n_train_particles + n_val_particles == self.n_particles
        return shuffled_batch[:n_train_particles], shuffled_batch[-n_val_particles:]

    @partial(jit, static_argnums=0)
    def _step(self, particles, optimizer_state, params):
        """
        Updates particles in the direction given by self.gradient

        Arguments:
            particles: jnp.ndarray of shape (n, d)
            params: can be anything. e.g. inducing particles in the case of SVGD,
        deep NN params for learned f, or None.

        Returns:
            particles (updated)
            optimizer_state (updated)
            grad_aux: dict containing auxdata
        """
        grads, grad_aux = self.gradient(params, particles, aux=True)
        updated_grads, optimizer_state = self.opt.update(grads, optimizer_state, particles)
        particles = optax.apply_updates(particles, updated_grads)
        grad_aux.update({
            "global_grad_norm": optax.global_norm(grads),
            "global_grad_norm_post_update": optax.global_norm(updated_grads),
        })
        grad_aux.update({})
        # grad_aux.update({"grads": updated_grads})
        return particles, optimizer_state, grad_aux

    def step(self, params):
        """Log rundata, take step. Mutates state"""
        updated_particles, self.optimizer_state, auxdata = self._step(
            self.particles, self.optimizer_state, params)
        self.log(auxdata)
        self.particles = updated_particles
        self.step_counter += 1
        return None

    def log(self, grad_aux=None):
        metrics.append_to_log(self.rundata, self._log(self.particles, self.step_counter))
        if self.step_counter % 10 == 0 and self.compute_metrics:
            aux_metrics = self.compute_metrics(self.particles)
            metrics.append_to_log(self.rundata,
                                  {k: (self.step_counter, v) for k, v in aux_metrics.items()})
        if grad_aux is not None:
            metrics.append_to_log(self.rundata, grad_aux)

    @partial(jit, static_argnums=0)
    def _log(self, particles, step):
        auxdata = {}
        if self.d < 400:
            auxdata.update({
                "step": step,
                "particles": particles,
                "mean": np.mean(particles, axis=0),
                "std": np.std(particles, axis=0),
            })
        return auxdata

    def done(self):
        """converts rundata into arrays"""
        if self.donedone:
            print("already done.")
            return
        skip = "particles accuracy".split()
        self.rundata = {
            k: v if k in skip else np.array(v)
            for k, v in self.rundata.items()
        }
        if "particles" in self.rundata:
            self.rundata["particles"] = np.array(self.rundata['particles'])
        self.donedone = True


class VectorFieldMixin:
    """Define the architecture of the witness function and initialize it."""
    def __init__(self,
                 target_dim: int,
                 key=random.PRNGKey(42),
                 sizes: list = None,
                 aux=False,
                 normalize_inputs=False,
                 extra_term: callable = lambda x: 0,
                 hypernet: bool = False,
                 particle_unravel: callable = None,
                 **kwargs):
        """
        args:
            aux: bool; whether to add mean and std as auxiliary input to MLP.
            normalize_inputs: whether to normalize particles
            hypernet: if true, use a hypernet architecture (for BNN inference.)
        """
        self.aux = aux
        self.d = target_dim
        self.sizes = sizes if sizes else [32, 32, self.d]
        self.auxdim = self.d*2
        if self.sizes[-1] != self.d:
            warnings.warn(f"Output dim should equal target dim; instead "
                          f"received output dim {sizes[-1]} and "
                          f"target dim {self.d}.")
        self.threadkey, subkey = random.split(key)
        self.normalize_inputs = normalize_inputs
        self.extra_term = extra_term
        self.hypernet = hypernet
        self.particle_unravel = particle_unravel

        # net and optimizer
        if hypernet:
            def field(x, aux, dropout: bool = False):
                h = nets.StaticHypernet(sizes=[64, 64])
                params = self.particle_unravel(x)
                return utils.ravel(h(params, dropout))
        else:
            def field(x, aux, dropout: bool = False):
                mlp = nets.MLP(self.sizes)
                scale = hk.get_parameter("scale", (), init=lambda *args: np.ones(*args))
                mlp_input = np.concatenate([x, aux]) if self.aux else x
                return scale * mlp(mlp_input, dropout)
        self.field = hk.transform(field)
        self.params = self.init_params()
        super().__init__(**kwargs)

    def compute_aux(self, particles):
        """Auxiliary data that will be concatenated onto MLP input.
        Output has shape (self.auxdim,).
        Can also be None."""
        if not self.aux:
            return None
        aux = np.concatenate([np.mean(particles, axis=0), np.std(particles, axis=0)])
        assert self.auxdim == len(aux)
        return aux

    def init_params(self, key=None, keep_params=False):
        """Initialize MLP parameter"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        x_dummy = np.ones(self.d)
        aux_dummy = np.ones(self.auxdim) if self.aux else None
        params = self.field.init(key, x_dummy, aux_dummy)
        return params

    def get_params(self):
        return self.params

    def get_field(self, init_particles=None, params=None, dropout=False):
        """Retuns function v. v is a vector field, can take either single
        particle of shape (d,) or batch shaped (..., d).
        """
        if params is None:
            params = self.get_params()
        if self.normalize_inputs:
            if init_particles is None:
                raise ValueError("init_particles must not be None when"
                                 "normalize_inputs is True.")
            norm = nets.get_norm(init_particles)
        else:
            norm = lambda x: x
        aux = self.compute_aux(init_particles)

        if dropout:
            def v(x, key):
                """x should have shape (n, d) or (d,)"""
                return self.field.apply(
                    params, key, norm(x), aux, dropout=dropout) + self.extra_term(x)
        else:
            def v(x):
                """x should have shape (n, d) or (d,)"""
                return self.field.apply(
                    params, None, norm(x), aux, dropout=dropout) + self.extra_term(x)
        return v


class EBMMixin():
    def __init__(self,
                 target_dim: int,
                 key=random.PRNGKey(42),
                 sizes: list = None,
                 **kwargs):
        self.d = target_dim
        self.sizes = sizes if sizes else [32, 32, 1]
        if self.sizes[-1] != 1:
            warnings.warn(f"Output dim should equal 1; instead "
                          f"received output dim {sizes[-1]}")
        self.threadkey, subkey = random.split(key)

        # net and optimizer
        self.ebm = hk.transform(
            lambda *args: nets.MLP(self.sizes)(*args))
        self.params = self.init_params()
        super().__init__(**kwargs)

    def init_params(self, key=None):
        """Initialize MLP parameter"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        x_dummy = np.ones(self.d)
        params = self.ebm.init(key, x_dummy)
        return params

    def get_params(self):
        return self.params

    def get_field(self, init_particles, params=None):
        del init_particles
        if params is None:
            params = self.get_params()

        def ebm(x):
            """x should have shape (d,)"""
            # norm = nets.get_norm(init_particles)
            # x = norm(x)
            return np.squeeze(self.ebm.apply(params, None, x))
        return grad(ebm)


class TrainingMixin:
    """
    Encapsulates methods for training the Stein network (which approximates
    the particle update). Agnostic re: architecture. Needs existence of
    a self.params at initialization.
    Methods to implement:
    * self.loss_fn
    * self._log
    """
    def __init__(self,
                 learning_rate: float = 1e-2,
                 patience: int = 10,
                 dropout: bool = False,
                 **kwargs):
        """
        args:
            dropout: whether to use dropout during training
        """
#        schedule_fn = optax.piecewise_constant_schedule(
#                -learning_rate, {50: 1/5, 100: 1/2})
#        self.opt = optax.chain(
#                optax.scale_by_adam(),
#                optax.scale_by_schedule(schedule_fn))
        self.opt = optax.adam(learning_rate)
        self.optimizer_state = self.opt.init(self.params)
        self.dropout = dropout

        # state and logging
        self.step_counter = 0
        self.rundata = {"train_steps": []}
        self.frozen_states = []
        self.patience = Patience(patience)
        super().__init__(**kwargs)

    @partial(jit, static_argnums=0)
    def _step(self,
              key,
              params,
              optimizer_state,
              dlogp,
              val_dlogp,
              particles,
              val_particles):
        """
        update parameters and compute validation loss
        args:
            dlogp: array of shape (n_train, d)
            val_dlogp: array of shape (n_validation, d)
        """
        [loss, loss_aux], grads = value_and_grad(self.loss_fn,
                                                 has_aux=True)(params,
                                                               dlogp,
                                                               key,
                                                               particles,
                                                               dropout=self.dropout)
        grads, optimizer_state = self.opt.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, grads)

        _, val_loss_aux = self.loss_fn(params,
                                       val_dlogp,
                                       key,
                                       val_particles,
                                       dropout=False)
        auxdata = {k: v for k, v in loss_aux.items()}
        auxdata.update({"val_" + k: v for k, v in val_loss_aux.items()})
        auxdata.update({"global_gradient_norm": optax.global_norm(grads),})
        return params, optimizer_state, auxdata

    def step(self,
             particles,
             validation_particles,
             dlogp,
             val_dlogp):
        """Step and mutate state"""
        self.threadkey, key = random.split(self.threadkey)
        self.params, self.optimizer_state, auxdata = self._step(
            key, self.params, self.optimizer_state, dlogp, val_dlogp,
            particles, validation_particles)
        self.write_to_log(auxdata)
        self.step_counter += 1
        return None

    def write_to_log(self, step_data: Mapping[str, np.ndarray]):
        metrics.append_to_log(self.rundata, step_data)

    def train(self,
              split_particles,
              split_dlogp,
              n_steps=5,
              early_stopping=True,
              progress_bar=False):
        """
        batch and next_batch cannot both be None.

        Arguments:
            split_particles: arrays (training, validation) of particles,
                shaped (n, d) resp (m, d)
            split_dlogp: arrays (training, validation) of loglikelihood
                gradients. Same shape as split_particles.
            key: random.PRGNKey
            n_steps: int, nr of steps to train
        """
        self.patience.reset()

        def step():
            self.step(*split_particles, *split_dlogp)
            val_loss = self.rundata["val_loss"][-1]
            self.patience.update(val_loss)
            return

        for i in tqdm(range(n_steps), disable=not progress_bar):
            step()
            # self.write_to_log({"model_params": self.get_params()})
            if self.patience.out_of_patience() and early_stopping:
                break
        self.write_to_log({"train_steps": i+1})
        return

#    def warmup(self,
#               key,
#               sample_split_particles: callable,
#               next_data: callable = lambda: None,
#               n_iter: int = 10,
#               n_inner_steps: int = 30,
#               progress_bar: bool = False,
#               early_stopping: bool = True):
#        """resample from particle initializer to stabilize the beginning
#        of the trajectory
#        args:
#            key: prngkey
#            sample_split_particles: produces next x_train, x_val sample
#            next_data: produces next batch of data
#            n_iter: number of iterations (50 training steps each)
#        """
#        for _ in tqdm(range(n_iter), disable=not progress_bar):
#            key, subkey = random.split(key)
#            self.train(sample_split_particles(subkey),
#                       n_steps=n_inner_steps,
#                       data=next_data(),
#                       early_stopping=early_stopping)

    def freeze_state(self):
        """Stores current state as tuple (step_counter, params, rundata)"""
        self.frozen_states.append((self.step_counter,
                                   self.get_params(),
                                   self.rundata))
        return

    def loss_fn(self, params, dlogp, key, particles, dropout):
        raise NotImplementedError()

    def gradient(self, params, particles, aux=False):
        raise NotImplementedError()


class SteinNetwork(VectorFieldMixin, TrainingMixin):
    """Parametrize vector field to maximize the stein discrepancy"""
    def __init__(self,
                 target_dim: int,
                 key: np.array = random.PRNGKey(42),
                 sizes: list = None,
                 learning_rate: float = 5e-3,
                 patience: int = 0,
                 aux=False,
                 lambda_reg=1/2,
                 use_hutchinson: bool = False,
                 dropout=False,
                 normalize_inputs=False,
                 extra_term: callable = lambda x: 0,
                 l1_weight: float = None,
                 hypernet: bool = False,
                 particle_unravel: callable = None):
        """
        args:
            aux: bool, whether to concatenate particle dist info onto
        mlp input
            use_hutchinson: when True, use Hutchinson's estimator to
        compute the stein discrepancy.
            normalize_inputs: normalize particles
        """
        super().__init__(target_dim, key=key, sizes=sizes,
                         learning_rate=learning_rate, patience=patience,
                         aux=aux, dropout=dropout, normalize_inputs=normalize_inputs, 
                         extra_term=extra_term, hypernet=hypernet,
                         particle_unravel=particle_unravel)
        self.lambda_reg = lambda_reg
        self.scale = 1.  # scaling of self.field
        self.use_hutchinson = use_hutchinson
        self.l1_weight = l1_weight

    def loss_fn(self,
                params,
                dlogp: np.ndarray,
                key: np.ndarray,
                particles: np.ndarray,
                dropout: bool = False):
        """
        Arguments:
            params: neural net paramers
            dlogp: gradient grad(log p)(x), shaped (n, d)
            key: random PRNGKey
            particles: array of shape (n, d)
            dropout: whether to use dropout in the gradient network
        """
        n, d = particles.shape
        v = self.get_field(particles, params, dropout=dropout)
        if dropout:
            f = utils.negative(v)
        else:
            def f(x, dummy_key):
                return -v(x)

        # stein discrepancy
        def h(x, dlogp_x, key):
            zkey, fkey = random.split(key)
            z = random.normal(zkey, (d,))
            zdf = grad(lambda _x: np.vdot(z, f(_x, fkey)))
            div_f = np.vdot(zdf(x), z)
            #div_f = np.trace(jacfwd(f)(x, fkey))
            sd = np.vdot(f(x, fkey), dlogp_x) + div_f
            l2 = np.vdot(f(x, fkey), f(x, fkey))
            aux = {
                "sd": sd,
                "l2": l2,
            }
            return -sd + l2 * self.lambda_reg, aux
        keys = random.split(key, n)
        loss, aux = vmap(h)(particles, dlogp, keys)
        loss = loss.mean()
        aux = {k: v.mean() for k, v in aux.items()}
        fnorm = optax.global_norm(jnp.mean(vmap(f)(particles, keys), axis=0))
        pnorm = optax.global_norm(jnp.mean(dlogp, axis=0))
        aux.update({"loss": loss,
                    "l1_diff": fnorm - pnorm,
                    "l1_ratio": fnorm / pnorm})
#        #  add L1 term
#        if self.l1_weight:
#            loss = loss + self.l1_weight * np.abs(jnp.mean(vmap(f)(particles) - dlogp))
        return loss, aux

    def gradient(self, params, particles, aux=False):
        """
        Plug-in particle update method. No dropout.
        Update particles via particles = particles - eps * v(particles)
        args:
            params: pytree of neural net parameters
            particles: array of shape (n, d)
            aux: bool
        """
        v = vmap(self.get_field(particles, params, dropout=False))
        if aux:
            return v(particles), {}
        else:
            return v(particles)

    def grads(self, particles):
        """Same as `self.gradient` but uses state"""
        return self.gradient(self.get_params(), particles)

    def done(self):
        """converts rundata into arrays"""
        self.rundata = {
            k: v if k in ["model_params", "gradient_norms"] else np.array(v)
            for k, v in self.rundata.items()
        }


class KernelGradient():
    """Computes the SVGD approximation to grad(KL), ie
    phi*(y) = E[grad(log p)(y) k(x, y) + div(k)(x, y)]"""
    def __init__(self,
                 target_logp: callable = None,  # TODO replace with dlogp supplied as array
                 get_target_logp: callable = None,
                 kernel=kernels.get_rbf_kernel,
                 bandwidth=None,
                 scaled=False,
                 lambda_reg=1/2,
                 use_hutchinson: bool = False):
        """get_target_log is a callable that takes in a batch of data
        (can be any pytree of jnp.ndarrays) and returns a callable logp
        that computes the target log prob (up to an additive constant).
        scaled: whether to rescale gradients st. they match
        (grad(logp) - grad(logp))/(2 * lambda_reg) in scale
        """
        if target_logp:
            assert not get_target_logp
            self.get_target_logp = lambda *args: target_logp
        elif get_target_logp:
            self.get_target_logp = get_target_logp
        else:
            return ValueError("One of target_logp and get_target_logp must"
                              "be given.")
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.rundata = {}
        self.scaled = scaled
        self.use_hutchinson = use_hutchinson

    def get_field(self, inducing_particles, batch=None):
        """return -phistar"""
        target_logp = self.get_target_logp(batch)
        bandwidth = self.bandwidth if self.bandwidth else kernels.median_heuristic(inducing_particles)
        kernel = self.kernel(bandwidth)
        phi = stein.get_phistar(kernel, target_logp, inducing_particles)
        return utils.negative(phi), bandwidth

    def gradient(self, batch, particles, aux=False):
        """Compute approximate KL gradient.
        args:
            batch: minibatch data used to estimate logp (can be None)
            particles: array of shape (n, d)
        """
        target_logp = self.get_target_logp(batch)
        v, h = self.get_field_scaled(particles, batch) if self.scaled \
            else self.get_field(particles, batch)
        if aux:
            return vmap(v)(particles), {"bandwidth": h,
                                        "logp": vmap(target_logp)(particles)}
        else:
            return vmap(v)(particles)

    def get_field_scaled(self, inducing_particles, batch=None):
        hardcoded_seed = random.PRNGKey(0)  # TODO seed should change across iters
        target_logp = self.get_target_logp(batch)
        bandwidth = self.bandwidth if self.bandwidth else kernels.median_heuristic(inducing_particles)
        kernel = self.kernel(bandwidth)
        phi = stein.get_phistar(kernel, target_logp, inducing_particles)
        l2_phi_squared = utils.l2_norm_squared(inducing_particles, phi)
        if self.use_hutchinson:
            ksd = stein.stein_discrepancy_hutchinson(hardcoded_seed, inducing_particles, target_logp, phi)
        else:
            ksd = stein.stein_discrepancy(inducing_particles, target_logp, phi)
        alpha = ksd / (2*self.lambda_reg*l2_phi_squared)
        return utils.mul(phi, -alpha), bandwidth


class EnergyGradient():
    """Compute pure SGLD gradient grad(log p)(x) (without noise)"""
    def __init__(self,
                 target_logp,
                 lambda_reg=1/2):
        self.target_logp = target_logp
        self.lambda_reg = lambda_reg
        self.rundata = {}

    def target_score(self, x):
        return grad(self.target_logp)(x) / (2*self.lambda_reg)

    def get_field(self, inducing_particles):
        """Return vector field used for updating, grad(log p)(x)$
        (without noise)."""
        return utils.negative(self.target_score)

    def gradient(self, _, particles, aux=False):
        """Compute gradient used for SGD particle update"""
        v = self.get_field(particles)
        if aux:
            return vmap(v)(particles), {}
        else:
            return vmap(v)(particles)
