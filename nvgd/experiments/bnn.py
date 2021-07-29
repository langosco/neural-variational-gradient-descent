from jax import numpy as jnp
from jax import jit, vmap, random, value_and_grad
import jax
import jax.numpy as jnp
import haiku as hk
from nvgd.experiments import config as cfg
from nvgd.experiments import dataloader
from nvgd.src import nets, utils

data = dataloader.data
model = nets.cnn


# Get cnn_unravel method to map tensor to dict
dummy_rngkey = random.PRNGKey(0)
base_params = model.init(dummy_rngkey, dataloader.data.train_images[:2])
_, cnn_unravel = jax.flatten_util.ravel_pytree(base_params)
del _



def init_flat_params(key):
    return utils.ravel(model.init(key, data.train_images[:2]))


# Accuracy
@jit
def accuracy(logits, labels):
    """
    Standard (single model) accuracy.
    Args:
        logits: shaped (batch, num_classes).
        labels: categorical labels shaped (batch,) int array (not one-hot).
    """
    preds = jnp.argmax(logits, axis=1)
    return jnp.mean(preds == labels)


@jit
def ensemble_accuracy(logits, labels):
    """use ensemble predictions to compute validation accuracy.
    args:
        logits: result from vmap(model.apply, (0, None))(param_set, images),
            shaped (num_models, batch, NUM_CLASSES)
        labels: batch of corresponding labels, shape (batch,)"""
    preds = jnp.mean(vmap(jax.nn.softmax)(logits), axis=0)  # mean prediction
    return jnp.mean(preds.argmax(axis=1) == labels)


@jit
def minibatch_accuracy(param_set, images, labels):
    """ensemble accuracy computed on a minibatch given by images, labels"""
    logits = vmap(model.apply, (0, None))(param_set, images)
    return ensemble_accuracy(logits, labels)


def compute_acc(param_set):
    accs = []
    for batch in data.val_batches:
        accs.append(minibatch_accuracy(param_set, *batch))
    return jnp.mean(jnp.array(accs))


def compute_acc_from_flat(param_set_flat):
    param_set = vmap(cnn_unravel)(param_set_flat)
    return compute_acc(param_set)


@jit
def test_accuracy(param_set):
    """
    args:
        param_set: pytree of neural network parameters
            such that every leaf has an added first axis
            storing n sets of nn parameters
    """
    return jax.lax.map( 
        lambda batch: minibatch_accuracy(param_set, *batch),
        data.test_batches_arr
        ).mean()


@jit
def val_accuracy(param_set):
    """
    args:
        param_set: pytree of neural network parameters
            such that every leaf has an added first axis
            storing n sets of nn parameters
    """
    return jax.lax.map(
        lambda batch: minibatch_accuracy(param_set, *batch),
        data.val_batches_arr
        ).mean()


@jit
def val_accuracy_single_net(params):
    """same as val_accuracy, but params is just a simple pytree
    where each leaf stores a parameter array"""
    def acc(batch):
        images, labels = batch
        logits = model.apply(params, images)
        return accuracy(logits, labels)
    return jax.lax.map(acc, data.val_batches_arr).mean()

# Loss
def crossentropy_loss(logits, labels, label_smoothing=0.):
    """Compute cross entropy for logits and labels w/ label smoothing
    Args:
        logits: [batch, num_classes] float array.
        labels: categorical labels [batch,] int array (not one-hot).
        label_smoothing: label smoothing constant, used to determine the 
        on and off values.
    """
    num_classes = logits.shape[-1]
    labels = jax.nn.one_hot(labels, num_classes)
    if label_smoothing > 0:
        labels = labels * (1 - label_smoothing) + label_smoothing / num_classes
    logp = jax.nn.log_softmax(logits)
    return -jnp.sum(logp * labels)  # summed loss over batch
                                    # equal to model_loglikelihood(data | params)


def log_prior(params):
    """Gaussian prior used to regularize weights (same as initialization).
    unscaled."""
    params_flat, _ = jax.flatten_util.ravel_pytree(params)
    return - jnp.sum(params_flat**2) / nets.INIT_STDDEV_CNN**2 / 2


def loss(params, images, labels):
    """Minibatch approximation of the (unnormalized) Bayesian
    negative log-posterior evaluated at `params`. That is,
    -log model_likelihood(data_batch | params) * batch_rescaling_constant - log prior(params))"""
    logits = model.apply(params, images)
    return data.train_data_size/cfg.batch_size * crossentropy_loss(logits, labels) - log_prior(params)


def get_minibatch_logp(batch):
    """
    Returns a callable that computes target posterior
    given flattened param vector.

    args:
        batch: tuple (images, labels)
    """
    @jit
    def minibatch_logp(params_flat):
        return -loss(cnn_unravel(params_flat), *batch)
    return minibatch_logp


def minibatch_logp(params_flat, batch):
    return -loss(cnn_unravel(params_flat), *batch)


@jit
def squared_error_loss(particles, fnet, ftrue):
    """
    MC estimate of the true loss (without using the div(f) trick.
    Up to rescaling + constant, this is equal to the squared
    error E[(f - f_true)**2].
    args:
        particles: array shaped (n, d)
        fnet: callable, computes witness fn
        ftrue: callable, computes grad(log p) - grad(log q)
    """
    return jnp.mean(vmap(
        lambda x: jnp.inner(fnet(x), fnet(x)) / 2 - jnp.inner(fnet(x), ftrue(x)))(
            particles))


@jit
def split_vdlogp(split_particles, batch):
    """
    Compute value and grad of logp.
    args:
        split_particles: tuple (train_particles, val_particles)
        batch: tuple (images, labels)

    returns:
        two tuples: logp = (train_logp, val_logp) and similarly for grad(logp).
    """
    vg = vmap(value_and_grad(minibatch_logp), (0, None))
    train_out, val_out = [vg(x, batch) for x in split_particles]
    return tuple(zip(train_out, val_out))
