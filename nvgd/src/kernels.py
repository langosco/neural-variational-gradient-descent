import jax.numpy as np
from jax import vmap, grad
from jax.scipy import stats
from . import utils

"""A collection of positive definite kernel functions.
Every kernel takes as input two jax scalars or arrays x, y of shape (d,),
where d is the particle dimension, and outputs a scalar.
"""
def _check_xy(x, y, dim=None):
    """If dim is supplied, also check for correct dimension"""
    x, y = [np.asarray(v) for v in (x, y)]
    if x.shape != y.shape:
        raise ValueError(f"Shapes of particles x and y need to match. "
                         f"Recieved shapes x: {x.shape}, y: {y.shape}")
    elif x.ndim > 1:
        raise ValueError(f"Input particles x and y can't have more than one "
                         f"dimension. Instead they have rank {x.ndim}")
    if dim is not None:
        if dim > 1:
            if x.ndim != 1 or x.shape[0] != dim:
                raise ValueError(f"x must have shape {(dim,)}. Instead received "
                                 f"shape {x.shape}.")
        elif dim == 1:
            if not (x.ndim==0 or x.shape[0]==dim):
                raise ValueError(f"x must have shape (1,) or scalar. Instead "
                                 f"received shape {x.shape}.")
        else: raise ValueError(f"dim must be a natural nr")
    return x, y

def _check_bandwidth(bandwidth, dim=None):
    bandwidth = np.squeeze(np.asarray(bandwidth))
    if bandwidth.ndim > 1:
        raise ValueError(f"Bandwidth needs to be a scalar or a d-dim vector. "
                         f"Instead it has shape {bandwidth.shape}")
    elif bandwidth.ndim == 1:
        pass

    if dim is not None:
        if not (bandwidth.ndim == 0 or bandwidth.shape[0] in (0, 1, dim)):
            raise ValueError(f"Bandwidth has shape {bandwidth.shape}.")
    return bandwidth

def _normalizing_factor(bandwidth):
    if bandwidth.ndim==1 or bandwidth.shape[0]==1:
        return np.sqrt(2*np.pi)
    else:
        d = bandwidth.shape[0]
        return (2*np.pi)**(-d/2) * 1/np.sqrt(np.prod(bandwidth))


def get_rbf_kernel(bandwidth, squared=True, normalize=False, dim=None):
    """If squared, then take the square of bandwidth, ie
    k(x, y) = exp(- (x - y)^2 / bandwidth^2 / 2). Otherwise don't take
    the square (in this case bandwidth must be > 0 always)"""
    bandwidth = _check_bandwidth(bandwidth, dim)
    def rbf(x, y):
        # TODO: add version with param=bandwidth**2 instead of bandwidth
        x, y = _check_xy(x, y, dim)
        h_squared = bandwidth**2 if squared else bandwidth
        if normalize:
            return np.prod(stats.norm.pdf(x, loc=y, scale=np.sqrt(h_squared)))
        else:
            return np.exp(- np.sum((x - y)**2 / h_squared) / 2)
    return rbf

def get_tophat_kernel(bandwidth, normalize=False, dim=None):
    bandwidth = _check_bandwidth(bandwidth, dim)
    volume = np.prod(2*bandwidth)
    def tophat(x, y):
        x, y = _check_xy(x, y, dim)
        if normalize:
            return np.squeeze(np.where(np.all(np.abs(x - y) < bandwidth), 1/volume, 0.))
        else:
            return np.squeeze(np.where(np.all(np.abs(x - y) < bandwidth), 1., 0.))
    return tophat

def get_rbf_kernel_logscaled(logh, normalize=False):
    logh = np.asarray(logh)
    bandwidth = np.exp(logh)
    return get_rbf_kernel(bandwidth, normalize)

def get_multivariate_gaussian_kernel(sigma, dim=None):
    def ard(x, y):
        x, y = _check_xy(x, y, dim)
        s = np.dot(sigma, x-y)
        return np.squeeze(np.exp(-utils.normsq(s)/2))
    return ard

def get_tophat_kernel_logscaled(logh):
    logh = np.asarray(logh)
    bandwidth = np.exp(logh/2) # TODO remove 1/2?
    return get_tophat_kernel(bandwidth)

def constant_kernel(x, y):
    """Returns 1."""
    _check_xy(x, y)
    return np.array(1.)

def char_kernel(x, y):
    """Returns 1 if x==y, else 0"""
    _check_xy(x, y)
    return np.squeeze(np.where(x==y, 1., 0.))

def funnelize(v):
    """If v is standard 2D normal, then
    funnelize(v) is distributed as Neal's Funnel."""
    *x, y = v
    x, y = np.asarray(x), np.asarray(y)
    return np.append(x*np.exp(3*y/2), 3*y)

def defunnelize(z):
    """Inverse of funnelize."""
    *x, y = z
    x, y = np.asarray(x), np.asarray(y)
    return np.append(x*np.exp(-y/2), y/3)

def get_funnel_kernel(bandwidth):
    """Transform input based on `defunnelize` transformation, then apply rbf kernel."""
    rbf = get_rbf_kernel(bandwidth)
    def funnel_kernel(x, y):
        return rbf(defunnelize(x), defunnelize(y))
    return funnel_kernel

def scalar_product_kernel(x, y):
    """k(x, y) = x^T y"""
    return np.inner(x, y)

def get_imq_kernel(alpha: float=1, beta: float=-0.5):
    """
    alpha > 0
    beta \in (-1, 0)
    Returns:
    kernel k(x, y) = (alpha + ||x - y||^2)^beta
    """
    def inverse_multi_quadratic_kernel(x, y):
        return (alpha + utils.normsq(x - y))**beta
    return inverse_multi_quadratic_kernel

def get_inverse_log_kernel(alpha: float):
    def il_kernel(x, y):
        return (alpha + np.log(1 + utils.normsq(x - y)))**(-1)
    return il_kernel

def get_imq_score_kernel(alpha: float, beta: float, logp: callable):
    """
    Arguments:
    alpha > 0
    beta \in (-1, 0)
    logp computes log p(x)

    Returns:
    kernel k(x, y) = (alpha + ||\nabla \log p(x) - \nabla \log p(y)||^2)^beta
    """
    def imq_score_kernel(x, y):
        return (alpha + utils.normsq(grad(logp)(x) - grad(logp)(y)))**beta
    return imq_score_kernel

### Utils
def median_heuristic(x):
    """
    Heuristic for choosing the squared RBF bandwidth.

    IN: np array of shape (n,) or (n,d): set of particles
    OUT: scalar: bandwidth parameter for RBF kernel, based on the
    heuristic from the SVGD paper.
    Note: assumes k(x, y) = exp(- (x - y)^2 / h^2 / 2)
    """
    pairwise_dists = utils.squared_distance_matrix(utils.remove_diagonal(x))
    medsq = np.median(pairwise_dists)
    h = np.sqrt(0.5 * medsq / np.log(x.shape[0] + 1))
    return h
