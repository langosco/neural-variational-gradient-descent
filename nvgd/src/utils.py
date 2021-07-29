from contextlib import contextmanager
import sys
import os
import jax.numpy as np
import jax
from jax import jit, vmap, random, grad, jacfwd
from jax.ops import index_update, index
from jax import lax
import time
from functools import wraps
import itertools

from collections.abc import Iterable
import collections
import warnings
import numpy as onp
from typing import NamedTuple


def isiterable(obj):
    return isinstance(obj, Iterable)


##############################
### KL divergence utilities
def smooth_and_normalize(vec, normalize=True):
    """
    Parameters:
    * vec : np.array of shape (n,)
    * normalize : bool

    Returns:
    out : np.array of shape (n,).
    If vec_i = 0, then out_i = epsilon. If vec_i !=0, then out_i = vec_i - c.
    c is chosen such that sum(vec) == 1.
    """
    vec = np.asarray(vec, dtype=np.float32)

    if normalize:
        vec = vec / vec.sum()
    n = len(vec)
    epsilon = 0.0001
    num_nonzero = np.count_nonzero(vec)
    c = epsilon * (n - num_nonzero) / num_nonzero
    perturbation = (vec == 0)*epsilon - (vec != 0)*c
    return vec + perturbation


def get_bins_and_bincounts(samples, normalized=False):
    """take in samples, create a common set of bins, and compute the counts count(x in bin)
    for each bin and each sample x.
    Parameters
    ------------
    samples : np.array of shape (n,) or shape (k, n).
    - If shape (n,): interpreted as a set of n scalar-valued samples.
    - If shape (k, n): interpreted as k sets of n scalar-valued samples.

    Returns
    --------
    probabilities :
    bins :
    """
    nr_samples = np.prod(samples.shape)
    nr_bins = np.log2(nr_samples)
    nr_bins = int(max(nr_bins, 5))

    lims = [np.min(samples), np.max(samples)]
    bins = np.linspace(*lims, num=nr_bins)

    if samples.ndim == 2:
        out = np.asarray([np.histogram(x, bins=bins, density=normalized)[0] for x in samples])
        return out, bins
    elif samples.ndim == 1:
        return np.histogram(samples, bins=bins, density=normalized)[0], bins
    else:
        raise ValueError(f"Input must have shape (n,) or shape (k,n). Instead received shape {samples.shape}")


def get_histogram_likelihoods(samples):
    """
    Parameters:
    * samples : np.array of scalar-valued samples from a distribution.

    Returns:
    np.array of same length as samples, consisting of a histogram-based approximation of the pdf q(x_i) at the samples x_i
    """
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.squeeze(samples)
    if samples.ndim != 1:
        raise ValueError(f"The shape of samples has to be either (n,) or (n,1). Instead received shape {samples.shape}.")
    n = len(samples)

    bincounts, bins = get_bins_and_bincounts(samples)
    bincounts = np.array(bincounts, dtype=np.int32)
    likelihoods = smooth_and_normalize(bincounts) / np.diff(bins)

    sample_likelihoods = np.repeat(likelihoods, bincounts) # TODO this doesn't play well with jit, cause shape of output depends on values in bincounts
    return sample_likelihoods


# wrapper that prints when the function compiles
def verbose_jit(fun, *jargs, **jkwargs):
    """Does same thing as jax.jit, only that it also inserts a print statement."""
    @wraps(fun)
    def verbose_fun(*args, **kwargs):
        print(f"JIT COMPILING {fun.__name__}...")
        st = time.time()
        out = fun(*args, **kwargs)
        end = time.time()
        print(f"...done compiling {fun.__name__} after {end-st} seconds.")
        return out
    return jit(verbose_fun, *jargs, **jkwargs)


#from haiku._src.data_structures import frozendict
import collections


def isfinite(thing):
    if type(thing) is jax.interpreters.xla.DeviceArray:
        return np.all(np.isfinite(thing))
    elif type(thing) is onp.ndarray:
        return onp.all(onp.isfinite(thing))
    elif isinstance(thing, collections.Mapping):
        for k, v in thing.items():
            isfinite(v)
    elif isiterable(thing):
        for el in thing:
            isfinite(el)
    else:
        warnings.warn(f"Didn't recognize type {type(thing)}. Not checking for NaNs.", RuntimeWarning)
        return None


def warn_if_nonfinite(*args):
    for arg in args:
        if not isfinite(arg):
            warnings.warn(f"Detected NaNs or infs.", RuntimeWarning)
    return None


def is_pd(x):
    """check if matrix is positive defininite"""
    try:
        onp.linalg.cholesky(x)
        return True
    except onp.linalg.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in str(err):
            return False
        else:
            raise


## fori_loop implementation in terms of lax.scan taken from here https://github.com/google/jax/issues/1112
def fori_loop(lower, upper, body_fun, init_val):
    f = lambda x, i: (body_fun(i, x), ())
    result, _ = lax.scan(f, init_val, np.arange(lower, upper))
    return result


# this one from here https://github.com/google/jax/issues/650
# def fori_loop(_, num_iters, fun, init): # added the dummy _
#     dummy_inputs = np.zeros((num_iters, 0))
#     out, _ = lax.scan(lambda x, dummy: (fun(x), dummy), init, dummy_inputs)
#     return out


# this is the python equivalent given in the documentation https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html
def python_fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val

#############################
### better pairwise distances

def squared_distance_matrix(x):
    """
    Parameters:
    * x: np array of shape (n, d) or (n,)
    Returns:
    * np array of shape (n, n):
    consisting of squared distances ||xi - xj||^2
    """
    assert x.ndim < 3 and x.ndim > 0
    n = x.shape[0]
    if x.ndim == 1:
        x = np.reshape(x, (n, 1)) # add dummy dimension
    xx = np.tile(x, (n, 1, 1)) # shape (n, n, d)
    diff = xx - xx.transpose((1, 0, 2))

    def normsq(x):
        return np.inner(x, x)
    v_normsq = vmap(normsq) # outputs vector of norms
    vv_normsq = vmap(v_normsq)
    return vv_normsq(diff)


def getn(l):
    """
    IN: l = n^2 - n / 2
    OUT: n (positive integer solution)
    """
    n = (1 + np.sqrt(1 + 8*l)) / 2
    assert np.equal(np.mod(n, 1), 0) # make sure n is an integer

    n = int(n)
    assert l == n**2 - n / 2
    return n

# @jit
def squareform(distances):
    """
    IN: output from `pairwise_distances`, an array of length l = n^2 - n / 2 with entries d(x1, x2, d(x1, 3), ..., d(xn-1 xn)).
    OUT: a symmetric n x n distance matrix with entries d(x_i, x_j)
    """
    l = distances.shape[0]
    n = getn(l)
    out = np.zeros((n, n))
    out[np.triu_indices(n, k = 1)]

    out = index_update(out, index[np.triu_indices(n, k=1)], distances)
    out = out + out.T
    return out


##########################33
### cartesian product
from jax.ops import index_update, index
# @jit
def cartesian_product(*arrays):
    """
    IN: any number of np arrays of same length
    OUT: cartesian product of the arrays
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
#         arr[...,i] = a
        arr = index_update(arr, index[..., i], a)
    return arr.reshape(-1, la)


def dict_concatenate(dict_list, np_array=False):
    """
    Arguments:
    * dict_list: a list of dictionaries with the same keys. All values must be numeric or a nested dict.

    Returns:
    * a dictionary with the same keys as the input dictionaries. The values are lists
    consisting of the concatenation of the values in the input dictionaries.
    """
    for d in dict_list:
        if not isinstance(d, collections.Mapping):
            raise TypeError("Input has to be a list consisting of dictionaries.")
        elif not all([dict_list[i].keys() == dict_list[i+1].keys() for i in range(len(dict_list)-1)]):
            raise ValueError("The keys of all input dictionaries need to match.")

    keys = dict_list[0].keys()
    out = {key: [d[key] for d in dict_list] for key in keys}

    if np_array:
        for k, v in out.items():
            try:
                out[k] = np.asarray(v)
            except TypeError:
                out[k] = dict_concatenate(v)
    else:
        for k, v in out.items():
            if isinstance(v[0], collections.Mapping):
                out[k] = dict_concatenate(v)
    return out


def dict_mean(dict_list):
    """
    Arguments:
    * dict_list: a list of dictionaries with the same keys. All values must be numeric.

    Returns:
    * a dictionary with the same keys as the input dictionaries. The values are np
    arrays consisting of the mean of the values in the input dictionaries.
    """
    out = dict_concatenate(dict_list)
    for k, v in out.items():
        try:
            out[k] = np.mean(v, axis = 0)
            assert out[k].shape == dict_list[0][k].shape
        except TypeError:
            out[k] = dict_mean(v)
    return out


def dict_divide(da, db):
    """divide numeric dict recursively, a / b."""
    for (k, a), (k, b) in zip(da.items(), db.items()):
        try:
            da[k] = a / b
        except TypeError:
            da[k] = dict_divide(a, b)
    return da


def dict_asarray(dct: dict):
    for k, v in dct.items():
        try:
            dct[k] = np.asarray(v)
        except:
            try:
                dct[k] = dict_asarray(dct[k])
            except:
                pass # be nice if value is neither np-ifiable nor a dictionary.
    return dct


def flatten_dict(d): # TODO use jax.flatten_util.ravel_pytree instead
    """This assumes no name collisions"""
    def visit(subdict):
        flat = []
        for k, v in subdict.items():
            if isinstance(v, collections.Mapping):
                flat.extend(visit(v))
            else:
                flat.append((k, v))
        return flat
    return dict(visit(d))


def dict_cartesian_product(**kwargs):
    """
    >>> [x for x in dict_cartesian_product(chars="ab", nums=[1,2])]
    [{'chars': 'a', 'nums': 1},
     {'chars': 'a', 'nums': 2},
     {'chars': 'b', 'nums': 1},
     {'chars': 'b', 'nums': 2}]
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def nested_dict_contains_key(ndict: collections.Mapping, key):
    if key in ndict:
        return True
    else:
        for k, v in ndict.items():
            if isinstance(v, collections.Mapping):
                if nested_dict_contains_key(v, key):
                    return True
        return False


def dejaxify(array, target="list"):
    if target=="list":
        return onp.asarray(array).tolist()
    elif target=="numpy":
        return onp.asarray(array)
    else:
        raise ValueError("target must be one of 'list', 'numpy'.")


def dict_dejaxify(dictionary, target="list"): # alternatively, just remove the .tolist and make all onp
    """recursively turn all jax arrays into lists or np.arrays.
    target can be one of 'list', 'numpy'."""
    out = {}
    for k, v in dictionary.items():
        if isinstance(v, collections.Mapping):
            out[k] = dict_dejaxify(v, target=target)
        elif type(v) is list and len(v) < 4:
            try:
                out[k] = [dict_dejaxify(element, target=target) for element in v] # v might be a list of dicts...
            except AttributeError: # ...or not.
                out[k] = dejaxify(v, target=target)
        else:
            out[k] = dejaxify(v, target=target)
    return out


def leaf_shapes(pytree):
    """print shapes of pytree leaves"""
    return jax.tree_map(lambda v: v.shape, pytree)


def generate_pd_matrix(dim):
    A = onp.random.rand(dim, dim) * 2
    return onp.matmul(A, A.T)


def generate_parameters_for_gaussian(dim, k=None):
    if k is not None:
        means = onp.random.randint(0, 10, size=(k, dim)) # random means in [0, 10]
        covs = [generate_pd_matrix(dim) for _ in range(k)]
        weights = onp.random.randint(1, 5, k)
        weights = weights / weights.sum()
        return means, covs, weights
    else:
        mean = onp.random.randint(0, 10, size=(dim,)) # random mean in [0, 10]
        cov = generate_pd_matrix(dim)
        return mean.tolist(), cov.tolist()


def subsample(key, array, n_subsamples, replace=True, axis=0):
    """
    Arguments
    ----------

    Returns
    ----------
    np.array of same shape as array except that the specified axis has length n_subsamples.
    consists of random samples from input array.
    """
    subsample_idx = random.choice(key, array.shape[axis], shape=(n_subsamples,), replace=replace)
    subsample = array.take(indices=subsample_idx, axis=axis)
    return subsample


def compute_update_to_weight_ratio(params_pre, params_post):
    """
    Arguments
    ---------
    * params_pre, params_post are frozendicts containing parameters

    Returns
    --------
    dict with same keys containing one scalar per layer: ||dw|| / ||w||
    """
    assert params_pre.keys() == params_post.keys()
    # recurse until we find np arrays
    ratios = dict()
    for k, v in params_pre.items():
        if type(v) is jax.interpreters.xla.DeviceArray:
            try:
                ratios[k] =  np.linalg.norm(params_post[k] - v) / np.linalg.norm(v)
            except FloatingPointError:
                ratios[k] = np.nan
        elif isinstance(v, collections.Mapping):
            ratios[k] = compute_update_to_weight_ratio(v, params_post[k])
    return ratios


def mixture(components: list, weights: list):
    """
    * components: list of functions RNGkey --> sample
    * weights: mixture weights

    Returns function
    mix: RNGkey --> sample,
    where sample is a random sample from the mixture distribution.
    """
    k = len(components)
    assert k == len(weights)
    weights = np.asarray(weights)
    def mix(key):
        key1, key2 = random.split(key)
        component = random.categorical(key1, np.log(weights))
        return components[component](key2)
    return mix

@jit
def kl_of_gaussian(p, q):
    """
    Parameters
    ----------
    p, q : array-like, parameters of gaussian: p = [mu, var], q = [mu2, var2]
    """
    mu1, var1 = p
    mu2, var2 = q
    sigma1, sigma2 = [np.sqrt(v) for v in (var1, var2)]
    out =  np.log(sigma2 / sigma1) + (var1 + (mu1 - mu2)**2) / (2 * var2) - 1/2
    return np.squeeze(out)


def vmv_dot(vec_a, matrix, vec_b):
    """
    Returns x^T A x, the vector-matrix-vector dot product
    """
    return np.einsum("i,ij,j->", vec_a, matrix, vec_b)


def l2_norm_squared(samples, fun):
    """Returns mean of fun^T fun evaluated
    over samples"""
    def fun_norm(x): return np.inner(fun(x), fun(x))
    return np.mean(vmap(fun_norm)(samples))


def l2_normalize(fun: callable, samples, target_norm=1):
    """Rescale function fun so it has L2(q) norm equal to 1 (or target_norm, if supplied).
    samples need to be samples from q (used to compute the expectation)."""
    l2_fun = np.sqrt(l2_norm_squared(samples, fun) + 1e-9)
    def fun_normed(x):
        return fun(x) * target_norm / l2_fun
    return fun_normed


def squeeze_output(fun):
    """fun(...) --> np.squeeze(fun(...))"""
    def fun_with_squeezed_output(*args, **kwargs):
        return np.squeeze(fun(*args, **kwargs))
    return fun_with_squeezed_output


def reshape_input(fun, reshape_input_to=(1,)):
    """fun(a) --> fun(np.reshape(a, newshape=(1,))). Also squeezes output.
    Useful e.g. if you want function to accept scalar input."""
    def fun_accepts_scalar_input(x):
        x = np.reshape(x, newshape=reshape_input_to)
        return np.squeeze(fun(x))
    return fun_accepts_scalar_input


def negative(fun):
    """f --> -f"""
    def negfun(*args, **kwargs):
        return -fun(*args, **kwargs)
    return negfun


def div(fun):
    """return divergence of fun: R^d --> R^d"""
    def divergence(x):
        if x.ndim > 0:
            return np.trace(jacfwd(fun)(x))
        else:
            return grad(fun)(x)
    return divergence


def div_sq(fun):
    def divergence_sq(x):
        if x.ndim > 0:
            j = jacfwd(fun)(x)
            return np.sum(np.diag(j)**2)
        else:
            return grad(fun)(x)**2
    return divergence_sq


def mul(fun, factor):
    """f --> factor * fun"""
    return lambda *args, **kwargs: factor * fun(*args, **kwargs)


def normsq(x):
    return np.inner(x, x)


def qmult(key, b):
    """
    QMULT  Pre-multiply by random orthogonal matrix.
       QMULT(A) is Q*A where Q is a random real orthogonal matrix from
       the Haar distribution, of dimension the number of rows in A.
       Special case: if A is a scalar then QMULT(A) is the same as
                     QMULT(EYE(A)).
       Called by RANDSVD.
       Reference:
       G.W. Stewart, The efficient generation of random
       orthogonal matrices with an application to condition estimators,
       SIAM J. Numer. Anal., 17 (1980), 403-409.
    """
    try:
        n = b.shape[0]
        a = b.copy()
    except AttributeError:
        n = b
        a = np.eye(n)

    d = np.zeros(n)
    for k in range(n - 2, -1, -1):
        # Generate random Householder transformation.
        key, subkey = random.split(key)
        x = random.normal(subkey, (n - k,))
        s = np.linalg.norm(x)

        # Modification to make sign(0) == 1
        sgn = np.sign(x[0]) + float(x[0] == 0)
        s = sgn * s
        d = index_update(d, k, -sgn)
        x = index_update(x, 0, x[0] + s)
        beta = s * x[0]

        # Apply the transformation to a
        y = np.dot(x, a[k:n, :])
        a = index_update(a, index[k:n, :], a[k:n, :] - np.outer(x, (y / beta)))

    # Tidy up signs.
    for i in range(n - 1):
        a = index_update(a, index[i, :], d[i] * a[i, :])

    # Now randomly change the sign (Gaussian dist)
    a = index_update(a, index[n - 1, :], a[n - 1, :] * np.sign(random.normal(key, ())))
    return a


def get_particle_lims(particles):
    """particles is a (n, 2) array of 2d points.
    return lims (-a, a) st particles fit into square with
    corners at +- a."""
    a = np.max(np.abs(particles))
    return (-a, a)


def add_gauss(key: np.ndarray, param: np.ndarray, scale: float):
    return param + random.normal(key, param.shape) * scale


def add_noise(key, updates, scale):
    """updates = updates + scale * z,
    where z is standard normal."""
    num_vars = len(jax.tree_leaves(updates))
    treedef = jax.tree_structure(updates)
    all_keys = jax.random.split(key, num=num_vars + 1)
    noise = jax.tree_multimap(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
        updates, jax.tree_unflatten(treedef, all_keys[1:]))
    updates = jax.tree_multimap(
        lambda g, n: g + scale.astype(g.dtype) * n,
        updates, noise)
    return updates


def return_none_if_none(fun, argnum=0):
    """wrapper to tell function to return none
    when it gets None as argument."""
    def out_fun(*args, **kwargs):
        if args[argnum] is None:
            return
        else:
            return fun(*args, **kwargs)
    return out_fun


def init_scale(fun):
    """fun : R^d --> R^d"""
    stein_discrepancy = stein.stein_discrepancy(
        particles, target_logp, fun, aux=False)
    l2_f_sq = l2_norm_squared(particles, fun)
    return l2_norm_squared / stein_discrepancy


def null_diagonal(matrix):
    """matrix is an (n,n) array.
    output: same matrix with zeros along diagonal"""
    n = matrix.shape[0]
    trace_indices = [list(range(n))]*2
    matrix = index_update(matrix, trace_indices, 0)
    return matrix


def remove_diagonal(matrix):
    """matrix is an (n,n) array.
    ouput: (n, n-1) array, matrix without the diagonal"""
    n = matrix.shape[0]
    offdiag_idx = onp.nonzero(~onp.eye(n, dtype=bool))
    return matrix[offdiag_idx].reshape((n, n-1))


import optax


def polynomial_schedule(step):
    return 1. / (step + 1)**0.55

# def polynomial_schedule(step):
#     return 1. / (step + 1)**0.35 # .25


def sgld(learning_rate: float = 1e-2, random_seed: int = 0):
    return optax.chain(
        optax.scale(-learning_rate),
        optax.add_noise(np.sqrt(2*np.abs(learning_rate)), 0, random_seed),
    )


class ScaledSGLDState(NamedTuple):
    count: int
    key: np.ndarray


def scaled_sgld(key: np.ndarray, schedule_fn: callable = optax.constant_schedule(1.)):
    """
    Scale SGLD the correct way, using a custom schedule for the stepsize.

        an (init_fn, update_fn) Tuple"""
    scaler = optax.scale_by_schedule(schedule_fn)

    def init_fn(params):
        return ScaledSGLDState(count=0, key=key)

    def update_fn(updates, state, params=None):
        """
        returns
        - stepsize * updates + np.sqrt(2 stepsize) * z,
        where z is standard normal.
        """
        count, key = state
        stepsize = schedule_fn(count)
        count += 1
        updates = jax.tree_map(lambda g: -stepsize*g, updates)
        key, subkey = random.split(key)
        # TODO: either throw error when stepsize < 0 or put np.abs(stepsize)
        # under the square root.
        return add_noise(subkey, updates, np.sqrt(2*stepsize)), ScaledSGLDState(count=count, key=key)

    return optax.GradientTransformation(init_fn, update_fn)


optimizer_mapping = {
    "sgd": optax.sgd,
    "adam": optax.adam,
#    "sgld": sgld,
}


# easy way to supress annoying output, from
# https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def vmean(fun):
    """vmap, but computes mean along mapped axis"""
    def compute_mean(*args, **kwargs):
        return np.mean(vmap(fun)(*args, **kwargs), axis=-1)
    return compute_mean


def ravel(tree):
    """just ravel"""
    return jax.flatten_util.ravel_pytree(tree)[0]
