import jax.numpy as np
from jax import vmap, jacfwd
from scipy.spatial.distance import cdist
import ot

from . import utils, kernels


def append_to_log(dct, update_dict):
    """appends update_dict to dict, entry-wise. Creates list entry
    if it doesn't exist.
    """
    for key, newvalue in update_dict.items():
        dct.setdefault(key, []).append(newvalue)
    return dct


def wasserstein_distance(s1, s2):
    """
    Arguments: samples from two distributions, shape (n, d) (not (n,)).
    Returns: W2 distance inf E[d(X, Y)^2]^0.5 over all joint distributions of
    X and Y such that the marginal distributions are equal those of the input
    samples. Here, d is the euclidean distance."""
    M = cdist(s1, s2, "sqeuclidean")
    a = np.ones(len(s1)) / len(s1)
    b = np.ones(len(s2)) / len(s2)
    return np.sqrt(ot.emd2(a, b, M))


def sqrt_kxx(kernel: callable, particles_a, particles_b):
    """Approximate E[k(x, x)] in O(n^2)"""
    def sqrt_k(x, y):
        return np.sqrt(kernel(x, y))
    sv  = vmap(sqrt_k, (0, None))
    svv = vmap(sv, (None, 0))
    return np.mean(svv(particles_a, particles_b))
#    return np.mean(vmap(sqrt_k)(particles_a, particles_b))


def estimate_kl(logq: callable, logp: callable, samples):
    """KL divergence"""
    return np.mean(vmap(logq)(samples) - vmap(logp)(samples))


def _pushforward_log(logpdf: callable, tinv: callable):
    """
    Arguments
        logpdf computes log(p(_)), where p is a PDF.
        tinv is the inverse of an injective transformation T: R^d --> R^d, x --> z

    Returns
        log p_T(z), where z = T(x). That is, the pushforward log pdf
        log p_T(z) = log p(T^{-1} z) + log det(J_{T^{-1} z})
    """
    def pushforward_logpdf(z):
        det = np.linalg.det(jacfwd(tinv)(z))
#         if np.abs(det) < 0.001:
#             raise LinalgError("Determinant too small: T is not injective.")
        return logpdf(tinv(z)) + np.log(np.abs(det))
    return pushforward_logpdf


def pushforward_loglikelihood(t: callable, loglikelihood, samples):
    """
    Compute log p_T(T(x)) for all x in samples.

    Arguments
        t: an injective transformation T: R^d --> R^d, x --> z
        loglikelihood: np array of shape (n, d)
        samples: samples from p, shape (n, d)

    Returns
        np array of shape (n,): log p_T(z)$, where z = T(x) for all x in samples.
    That is, the pushforward log pdf
        log p_T(z) = log p(x) - log det(J_T x)
    """
    return loglikelihood - np.log(compute_jacdet(t, samples))


def compute_jacdet(t, samples):
    """Just computes the determinants of jacobians.
    Returns np array of shape (n,)"""
    def jacdet(x):
        return np.abs(np.linalg.det(jacfwd(t)(x)))
    return vmap(jacdet)(samples)


def get_mmd(kernel=kernels.get_rbf_kernel(1.)):
    """
    args:
        kernel: callable, computes a scalar kernel function.
    returns:
        mmd: callable, takes two sets of samples of shape (n, d)
    as input and returns the MMD distance between them (scalar).
    """
    kernel_matrix = vmap(vmap(kernel, (0, None)), (None, 0))

    def mmd(xs, ys):
        """Returns unbiased approximation of
        E[k(x, x') + k(y, y') - 2k(x, y)]"""
        kxx = utils.remove_diagonal(kernel_matrix(xs, xs))
        kyy = utils.remove_diagonal(kernel_matrix(ys, ys))
        kxy = kernel_matrix(xs, ys)
        return np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)
    return mmd


# tracers can be plugged into models.Particles to compute metrics
# dependent on particle position at regular intervals during the sampling run.
# A tracer takes as input the current particles (an array of shape (n, d)) and
# outputs a disctionary with scalar values.
def get_mmd_tracer(target_samples, kernel=kernels.get_rbf_kernel(1.)):
    mmd = get_mmd(kernel)

    def compute_mmd(particles):
        return {"mmd": mmd(particles, target_samples)}
    return compute_mmd


def get_funnel_tracer(target_samples):
    rbf_mmd = get_mmd(kernels.get_rbf_kernel(1.))
    funnel_mmd = get_mmd(kernels.get_funnel_kernel(1.))

    def compute_mmd(particles):
        return {"rbf_mmd": rbf_mmd(particles, target_samples),
                "funnel_mmd": funnel_mmd(particles, target_samples)}
    return compute_mmd


def get_squared_error_tracer(target_samples: np.ndarray,
                             statistic: callable,
                             name: str):
    """
    Trace the squared error of a statistic.
    args:
        target_samples: array of shape (n, d)
        statistic: fn to compute scalar from samples
        name: dict key for logging
    (if statistic is not scalar, then report squared error summed over all
    components.)
    """
    target_statistic = statistic(target_samples)

    def compute_summed_error(particles: np.ndarray):
        """compute squared error for each component, then
        sum across components."""
        return {name: np.sum((statistic(particles) - target_statistic)**2)}
    return compute_summed_error


def get_2nd_moment_tracer(target_samples):
    return get_squared_error_tracer(
        target_samples=target_samples,
        statistic=lambda particles: np.mean(particles**2, axis=0),
        name="second_error")


def combine_tracers(*tracers):
    """
    All arguments must be callables returning a dictionary.
    Returns a callable that computes the union of all dicts.
    """
    def combined_tracer(particles: np.ndarray):
        dicts = [tracer(particles) for tracer in tracers]
        return dict(item for d in dicts for item in d.items())
    return combined_tracer
