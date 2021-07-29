import jax.numpy as np
from jax import random, vmap, jacfwd, grad
from jax.scipy import stats
from nvgd.src import utils, plot, stein
import warnings

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

# check wikipedia for computation of higher moments
# https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Higher_moments
# also recall form of characteristic function
# https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)#Examples


class Distribution():
    """Base class for package logpdf + metrics + sampling"""
    def __init__(self):
        self.threadkey = random.PRNGKey(0)

    def sample(self, shape, key=None):
        raise NotImplementedError()

    def logpdf(self, x):
        raise NotImplementedError()

    def pdf(self, x):
        raise NotImplementedError()

    def compute_metrics(self, x, normalize=False):
        """Compute metrics given samples x.
        If normalize = True, then all values are divided by the corresponding
        expected value for a true random sample of the same size."""
        if x.shape[-1] != self.d:
            raise ValueError(f"Particles x need to have shape (n, d), where d = {self.d} is the particle dimension.")
        if normalize:
            n = x.shape[0]
            metrics = self.compute_metrics(x)
            sample_metrics = self.compute_metrics_for_sample(n)
            return utils.dict_divide(metrics, sample_metrics)
        else:
            sample_expectations = [np.mean(value, axis=0) for value in (x, x**2, np.cos(x), np.sin(x))]
            square_errors = [(sample - true)**2 for sample, true in zip(sample_expectations, self.expectations)]
            square_errors = np.array(square_errors)  # shape (4, d)

            metrics_dict = {
                "square_errors": square_errors  # shape (4, d)
            }

            return metrics_dict

    def _checkx(self, x):
        """check if particle (single particle shape (d,)) in right shape, etc"""
        x = np.array(x)
        if x.ndim == 0 and self.d == 1:
            x = x.reshape((1,))
        if x.shape != (self.d,):
            raise ValueError(f"x needs to have shape ({self.d},). Instead, received x of shape {x.shape}.")
        return x

    def get_metrics_shape(self):
        shapes = {
            "square_errors": (4, self.d)
        }
        if self.d == 1:
            shapes["KL Divergence"] = (1,)
        return shapes

    def initialize_metric_names(self):
        self.metric_names = {
            "square_errors": [f"SE for {val}" for val in ["X", "X^2", "cos(X)", "sin(X)"]]
        }
        if self.d == 1:
            self.metric_names["KL Divergence"] = "Estimated KL Divergence"
        return None

    def compute_metrics_for_sample(self, sample_size):
        """For benchmarking. Returns metrics computed for a true random
        sample of size sample_size, averaged over 100 random seeds."""
        if sample_size not in self.sample_metrics:
            def compute():
                sample = self.sample(shape=(sample_size,))
                sample = np.reshape(sample, newshape=(sample_size, self.d))
                return self.compute_metrics(sample)
            self.sample_metrics[sample_size] = utils.dict_mean([compute() for _ in range(100)])
        return self.sample_metrics[sample_size]


class Gaussian(Distribution):
    def __init__(self, mean, cov):
        """
        Possible input shapes for mean and cov:
        1) mean.shape defines dimension of domain: if mean has shape (d,),
        then particles have shape (d,)
        2) if covariance is a scalar, it is reshaped to diag(3 * (cov,))
        3) if covariance is an array of shape (k,), it is reshaped to diag(cov)"""
        super().__init__()

        self.mean, self.cov = self._check_and_reshape_args(mean, cov)
        self.d = len(self.mean)
        self.expectations = self.compute_expectations(self.mean, self.cov)
        self.key = random.PRNGKey(0)
        self.sample_metrics = dict()
        self.initialize_metric_names()

    def _check_and_reshape_args(self, mean, cov):
        mean = np.asarray(mean)
        cov = np.asarray(cov)
        if mean.ndim == 0:
            mean = np.reshape(mean, newshape=(1,))
        elif mean.ndim > 1:
            raise ValueError(f"Recieved inappropriate shape {mean.shape} for"
                             "mean. (Wrong nr of dimensions).")
        d = mean.shape[0]

        if cov.ndim == 0:
            cov = d * (cov,)
            cov = np.asarray(cov)
            cov = np.diag(cov)
        elif cov.ndim == 1:
            assert len(cov) == len(mean)
            cov = np.diag(cov)
        assert mean.ndim == 1 and cov.ndim == 2
        assert mean.shape[0] == cov.shape[0] == cov.shape[1]
        if not utils.is_pd(cov):
            raise ValueError("Covariance must be positive definite.")
        return mean, cov

    def compute_expectations(self, mean, cov):
        """
        returns a list of expected values of the following expressions:
        x, x^2, cos(x), sin(x)
        """
        # characteristic function at [1, ..., 1]:
        t = np.ones(self.d)
        char = np.exp(np.vdot(t, (1j * mean - np.dot(cov, t) / 2)))
        expectations = [mean, np.diagonal(cov) + mean**2, np.real(char), np.imag(char)]
        return expectations

    def sample(self, n_samples, key=None):
        """mutates self.key if key is None"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        out = random.multivariate_normal(key, self.mean, self.cov, shape=(n_samples,))
        shape = (n_samples, self.d)
        return out.reshape(shape)

    def logpdf(self, x):
        x = self._checkx(x)
        return stats.multivariate_normal.logpdf(x, self.mean, self.cov)

    def pdf(self, x):
        x = self._checkx(x)
        return stats.multivariate_normal.pdf(x, self.mean, self.cov)


class GaussianMixture(Distribution):
    def __init__(self, means, covs, weights):
        """
        Arguments:
        means, covs are np arrays or lists of length k, with entries of shape
        (d,) and (d, d) respectively. (e.g. covs can be array of shape (k, d, d))
        """
        means, covs, weights = self._check_and_reshape_args(means, covs, weights)
        self.d = len(means[0])
        self.expectations = self.compute_expectations(means, covs, weights)
        self.mean = self.expectations[0]
        # recall Cov(X) = E[XX^T] - mu mu^T =
        # sum_over_components(Cov(Xi) + mui mui^T) - mu mu^T
        mumut = np.einsum("ki,kj->kij", means, means)  # shape (k, d, d)
        self.cov = np.average(covs + mumut, weights=weights, axis=0) \
            - np.outer(self.mean, self.mean)
        self.threadkey = random.PRNGKey(0)
        self.means = means
        self.covs = covs
        self.weights = weights
        self.num_components = len(weights)
        self.sample_metrics = dict()
        self.initialize_metric_names()
        self.tfp_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=self.weights),
            components_distribution=tfd.MultivariateNormalFullCovariance(
                loc=self.means,
                covariance_matrix=self.covs)
        )

    def _check_and_reshape_args(self, means, covs, weights):
        """
        Shapes:
        means: (k, d) or (k,)
        covs: (k, d, d), (), (k,), (d, d)
        weights: (k,)
        """
        means = np.asarray(means, np.float32)
        covs = np.asarray(covs, np.float32)
        weights = np.asarray(weights, np.float32)
        weights = weights / weights.sum()  # normalize
        k = len(means)  # must equal the number of mixture components
        if means.ndim == 0:
            raise ValueError
        elif means.ndim == 1:
            means = means[:, np.newaxis]  # d = 1
        d = means.shape[1]
        if covs.ndim == 0:
            covs = np.asarray([np.identity(d)*covs]*k)
        elif covs.ndim == 1:  # assume dimension is components
            if len(covs) == k:
                covs = np.asarray([np.identity(d) * cov for cov in covs])
            elif len(covs) == d:
                warnings.warn("Using the same covariance for all"
                              " mixture components")
                covs = np.asarray([np.diag(covs)]*d)
            else:
                raise ValueError("Length of covariance vector must equal the"
                                 "number of mixture components.")
        elif covs.ndim == 2:
            covs = np.asarray([covs]*k)

        assert len(covs) == len(means)
        assert weights.ndim == 1 and len(weights) > 1
        assert means.ndim == 2 and covs.ndim == 3
        assert means.shape[1] == covs.shape[1] == covs.shape[2]

        for cov in covs:
            if not utils.is_pd(cov):
                raise ValueError("Covariance must be positive definite.")
        return means, covs, weights

    def compute_expectations(self, means, covs, weights):
        """
        returns a list of expected values of the following expressions:
        x, x^2, cos(x), sin(x)
        """
        # characteristic function at [1, ..., 1]:
        t = np.ones(self.d)
        chars = np.array([np.exp(np.vdot(t, (1j * mean - np.dot(cov, t) / 2))) for mean, cov in zip(means, covs)])  # shape (k,d)
        char = np.vdot(weights, chars)
        mean = np.einsum("i,id->d", weights, means)
        xsquares = [np.diagonal(cov) + mean**2 for mean, cov in zip(means, covs)]
        expectations = [mean, np.einsum("i,id->d", weights, xsquares),
                        np.real(char), np.imag(char)]
        expectations = [np.squeeze(e) for e in expectations]
        return expectations

    def sample(self, n_samples, key=None):
        """mutates self.rkey"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)

        def sample_from_component(rkey, component):
            return random.multivariate_normal(
                rkey, self.means[component], self.covs[component])

        key, subkey = random.split(key)
        keys = random.split(key, n_samples)
        components = random.categorical(subkey,
                                        np.log(self.weights),
                                        shape=(n_samples,))
        out = vmap(sample_from_component)(keys, components)
        shape = (n_samples, self.d)
        return out.reshape(shape)

    def logpdf(self, x):
        return self.tfp_dist.log_prob(x)

    def pdf(self, x):
        x = np.asarray(x)
        if x.shape != (self.d,) and not (self.d == 1 and x.ndim == 0):
            raise ValueError(f"Input x must be an np.array of length "
                             f"{self.d} and dimension one.")
        pdfs = vmap(stats.multivariate_normal.pdf, (None, 0, 0))(x, self.means, self.covs)
        return np.vdot(pdfs, self.weights)


class Funnel(Distribution):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mean = np.zeros(d)

        self.xcov = np.eye(d-1) * np.exp(9/2)
        self.ycov = 9
        if d == 2:
            self.cov = np.block([
                [self.xcov,          np.zeros((d-1, 1))],
                [np.zeros((1, d-1)), self.ycov         ]
            ])
        else:
            self.cov = None

    def sample(self, n_samples, key=None):
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        key, subkey = random.split(key)
        y = random.normal(subkey, (n_samples, 1)) * 3
        key, subkey = random.split(key)
        x = random.normal(subkey, (n_samples, self.d-1)) * np.exp(y/2)
        return np.concatenate([x, y], axis=1)

    def pdf(self, x):
        x = self._checkx(x)
        *x, y = x
        x, y = np.asarray(x), np.asarray(y)

        xmean = np.zeros(self.d-1)
        xcov  = np.eye(self.d-1)*np.exp(y)
        py = stats.norm.pdf(y, loc=0, scale=3)  # scale=stddev
        px = stats.multivariate_normal.pdf(x, mean=xmean, cov=xcov)
        return np.squeeze(py*px)

    def logpdf(self, x):
        x = self._checkx(x)
        *x, y = x
        x, y = np.asarray(x), np.asarray(y)
        xmean = np.zeros(self.d-1)
        xcov  = np.eye(self.d-1)*np.exp(y)  # Cov(X \given Y=y)
        logpy = stats.norm.logpdf(y, loc=0, scale=3)
        logpx = stats.multivariate_normal.logpdf(x, mean=xmean, cov=xcov)
        return np.squeeze(logpy + logpx)


class FunnelizedGaussian(Gaussian):
    def __init__(self, mean, cov):
        self.mean, self.cov = self._check_and_reshape_args(mean, cov)
        self.d = len(self.mean)
        self.threadkey = random.PRNGKey(0)

    def _check_and_reshape_args(self, mean, cov):
        if len(mean) < 2:
            raise ValueError("Funnel exists only in dimensions > 2."
                             f"Received dimension len(mean) = {len(mean)}")
        return super()._check_and_reshape_args(mean, cov)

    def funnelize(self, v):
        """If v is standard 2D normal, then
        funnelize(v) is distributed as Neal's Funnel."""
        *x, y = v
        x, y = np.asarray(x), np.asarray(y)
        return np.append(x*np.exp(3*y/2), 3*y)

    def defunnelize(self, z):
        """Inverse of funnelize."""
        *x, y = z
        x, y = np.asarray(x), np.asarray(y)
        return np.append(x*np.exp(-y/2), y/3)

    def logpdf(self, z):
        x = self.defunnelize(z)
        *_, y = x
        return super().logpdf(x) + 3 * np.exp(3/2 * y)

    def pdf(self, z):
        x = self.defunnelize(z)
        *_, y = x
        return super().pdf(x) * 3 * np.exp(3/2 * y)

    def sample(self, n_samples, key=None):
        return vmap(self.funnelize)(super().sample(n_samples, key=key))


class Uniform(Distribution):
    def __init__(self, lims):
        """
        lims has shape (d, 2)
        """
        lims = np.asarray(lims)
        if lims.ndim < 2:
            lims = lims[None, :]  # d=1
        elif lims.ndim < 1:
            raise
        self.lims = lims
        self.d = len(lims)
        self.mean = np.mean(self.lims, axis=1)
        self.cov = None

    def sample(self, n_samples, key=None):
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        key, subkey = random.split(key)
        samples = random.uniform(subkey, shape=(n_samples, self.d)) - 0.5
        scales = self.lims[:, 1] - self.lims[:, 0]
        return np.einsum("ij,j->ij", samples, scales) + self.mean

    def pdf(self, x):
        x = self._checkx(x)
        return stats.uniform.pdf(x,
                                 loc=self.lims[:, 0],
                                 scale=self.lims[:, 1] - self.lims[:, 0])

    def logpdf(self, x):
        x = self._checkx(x)
        return stats.uniform.logpdf(x,
                                    loc=self.lims[:, 0],
                                    scale=self.lims[:, 1] - self.lims[:, 0])


class Banana(Gaussian):
    def __init__(self, mean, cov):
        self.gauss_mean, self.gauss_cov = self._check_and_reshape_args(mean, cov)
        self.mean = np.array([0, self.gauss_cov[0, 0]])
        self.cov = np.array([
            [self.gauss_cov[0, 0], 0                                                      ],
            [0, 3*self.gauss_cov[0, 0]**2 + self.gauss_cov[1, 1] - self.gauss_cov[0, 0]**2]])
        self.d = len(self.mean)
        self.threadkey = random.PRNGKey(0)

    def _check_and_reshape_args(self, mean, cov):
        if len(mean) != 2:
            raise ValueError("Banana exists only in 2 dim."
                             f"Received dimension len(mean) = {len(mean)}")
        return super()._check_and_reshape_args(mean, cov)

    def bananify(self, v):
        """If v is 2D normal, then
        bananify(v) is distributed as a Banana."""
        *x, y = v
        x, y = np.asarray(x), np.asarray(y)
        return np.append(x, x**2 + y)

    def debananify(self, z):
        """Inverse of bananify."""
        *x, y = z
        x, y = np.asarray(x), np.asarray(y)
        return np.append(x, y - x**2)

    def logpdf(self, z):
        x = self.debananify(z)
        x = self._checkx(x)
        return stats.multivariate_normal.logpdf(x, self.gauss_mean, self.gauss_cov)

    def pdf(self, z):
        x = self.debananify(z)
        x = self._checkx(x)
        return stats.multivariate_normal.pdf(x, self.gauss_mean, self.gauss_cov)

    def sample(self, n_samples, key=None):
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        out = random.multivariate_normal(key, self.gauss_mean, self.gauss_cov, shape=(n_samples,))
        return vmap(self.bananify)(out.reshape((n_samples, self.d)))


class Ring(Distribution):
    def __init__(self, radius, var):
        """Both args are scalar.
        r ~ N(radius, var), phi ~ Uniform(-pi, pi)"""
        super().__init__()
        self.radius, self.var = self._check_and_reshape_args(radius, var)
        self.d = 2
        self.mean = np.array([0, 0])
        self.cov = np.cov(self.sample(10_000), rowvar=False)
        self.cov = np.diag(np.diag(self.cov))  # remove off-diagonal elements

    def _check_and_reshape_args(self, r, v):
        r, v = [np.array(a) for a in (r, v)]
        if r.ndim > 0 or v.ndim > 0:
            raise ValueError("Both radius and error variance must be scalar.")
        return r, v

    def to_cartesian(self, polar_coords):
        polar_coords = self._checkx(polar_coords)
        r, phi = polar_coords
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        return np.append(x, y)

    def to_polar(self, v):
        v = self._checkx(v)
        x, y = v
        r = np.linalg.norm(v)
        phi = np.arctan2(y, x)
        return np.append(r, phi)

    def logpdf(self, x):
        polar_coords = self.to_polar(x)
        jacdet = np.abs(np.linalg.det(jacfwd(self.to_cartesian)(polar_coords)))
        r, _ = polar_coords
        pr = stats.norm.logpdf(r, loc=self.radius, scale=np.sqrt(self.var))
        return pr - np.log(2*np.pi) - np.log(jacdet)

    def pdf(self, x):
        polar_coords = self.to_polar(x)
        jacdet = np.abs(np.linalg.det(jacfwd(self.to_cartesian)(polar_coords)))
        r, _ = polar_coords
        pr = stats.norm.pdf(r, loc=self.radius, scale=np.sqrt(self.var))
        return pr * 1/(2*np.pi) * 1/jacdet

    def sample(self, n_samples, key=None):
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        keya, keyb = random.split(key)
        phi = random.uniform(keya, minval=-np.pi, maxval=np.pi, shape=(n_samples,))
        r = random.normal(keyb, shape=(n_samples,)) + self.radius
        polar_coords = np.stack([r, phi], axis=1)
        return vmap(self.to_cartesian)(polar_coords)


class Squiggle(Gaussian):
    def __init__(self, mean, cov):
        self.mean, self.gauss_cov = self._check_and_reshape_args(mean, cov)
        self.cov = np.array([[self.gauss_cov[0,0], 0                      ],
                             [0, self.gauss_cov[1,1] + (1 + np.exp(-10))/2 - np.exp(-25)]])
        self.d = len(self.mean)
        self.threadkey = random.PRNGKey(0)

    def _check_and_reshape_args(self, mean, cov):
        if len(mean) != 2:
            raise ValueError("Squiggle exists only in 2 dim."
            f"Received dimension len(mean) = {len(mean)}")
        return super()._check_and_reshape_args(mean, cov)

    def squiggle(self, v):
        """If v is 2D normal, then
        squiggle(v) is distributed as a Squiggle."""
        *x, y = v
        x, y = np.asarray(x), np.asarray(y)
        return np.append(x, np.cos(5*x) + y)

    def desquiggle(self, z):
        """Inverse of squiggle."""
        *x, y = z
        x, y = np.asarray(x), np.asarray(y)
        return np.append(x, y - np.cos(5*x))

    def logpdf(self, z):
        x = self.desquiggle(z)
        x = self._checkx(x)
        return stats.multivariate_normal.logpdf(x, self.mean, self.gauss_cov)

    def pdf(self, z):
        x = self.desquiggle(z)
        x = self._checkx(x)
        return stats.multivariate_normal.pdf(x, self.mean, self.gauss_cov)

    def sample(self, n_samples, key=None):
        """mutates self.key if key is None"""
        if key is None:
            self.threadkey, key = random.split(self.threadkey)
        out = random.multivariate_normal(key, self.mean, self.gauss_cov, shape=(n_samples,))
        return vmap(self.squiggle)(out.reshape((n_samples, self.d)))


### Test distributions for experiments:
class Setup():
    """Simple container for target and proposal densities"""
    def __init__(self, target=None, proposal=None):
        self.target=target
        self.proposal=proposal

    def get(self):
        """Return target, proposal"""
        return self.target, self.proposal

    def plot(self, cmaps=("Blues", "plasma"),
             **kwargs):
        """
        cmaps: sequence of colormaps. First one used for proposal, second one
        for target.
        Proposal is density plot, target is contour plot.
        """
        if self.target.d == 1:
            plot.plot_fun(self.proposal.pdf, label="Proposal", **kwargs)
            plot.plot_fun(self.target.pdf, label="Target", **kwargs)
        elif self.target.d == 2:
            plot.plot_fun_2d(self.proposal.pdf, cmap=cmaps[0], **kwargs)
            plot.plot_fun_2d(self.target.pdf, type="contour", cmap=cmaps[1], **kwargs)
        else:
            raise NotImplementedError()
        return

    def grad_kl(self, x):
        return grad(self.proposal.logpdf)(x) - grad(self.target.logpdf)(x)

    def stein_discrepancy(self, f, samples=None):
        """Compute stein discrepancy to target using witness function f"""
        if samples is None:
            samples = self.proposal.sample(10_000)
        return stein.stein_discrepancy(samples, self.target.logpdf, f)

##########################
## 1-dimensional experiments
target = Gaussian(0, 3)
proposal = GaussianMixture([-5, 5], [1, 9], [1, 1])
simple_mixture = Setup(target, proposal)

# both prop and target are mixtures of different scales
# (situations like this will happen when fitting mixtures)
target = GaussianMixture([-5, 5], [1, 9], [1, 1])
proposal = GaussianMixture([-6, 0, 6], [9, 1, 1], [1,1,1])
double_mixture = Setup(target, proposal)

###########################
## 2-dimensional experiments
target = Funnel(2)
proposal = Gaussian([-3,0], 9)
funnel = Setup(target, proposal)

banana = Banana([0, 0], [4, 1]) # ie y = x**2 + eps; std 2 and 1 respectively
gauss = Gaussian([2, -4], [4, 4])
banana_target = Setup(banana, gauss)
gauss = Gaussian([0, 0], [4, 4])
banana_proposal = Setup(gauss, banana)

ring = Ring(10, .1)
gauss = Gaussian([0,0], 9)
ring_target = Setup(ring, gauss)
ring_proposal = Setup(gauss, ring)

target = Ring(10, .1)
proposal = Ring(15, .1)
double_ring = Setup(target, proposal)

target = Squiggle([0, 0], [1, .1])
proposal = Gaussian([-2, 0], [1, 1])
squiggle_target = Setup(target, proposal)

means = np.array([np.exp(2j*np.pi*x) for x in np.linspace(0, 1, 6)[:-1]])
means = np.column_stack((means.real,means.imag))
target = GaussianMixture(means, .03, np.ones(5))
proposal = Gaussian([-2, 0], [.5, .5])
mix_of_gauss = Setup(target, proposal)

target = Gaussian([0, 0], [1e-4, 9])
proposal = Gaussian([0, 0], [1, 1])
thin_target = Setup(target, proposal)

d = 50
variances = np.logspace(-2, 0, num=d)
target = Gaussian(np.zeros(d), variances)
proposal = Gaussian(np.zeros(d), np.ones(d))
high_d_gaussian = Setup(target, proposal)

# rotated gaussian
subkey = random.PRNGKey(0)
Q = utils.qmult(subkey, d)
cov = Q.T @ np.diag(variances) @ Q
proposal = Gaussian(np.zeros(d), np.ones(d))
target = Gaussian(np.zeros(d), cov)
rotated_gaussian = Setup(target, proposal)


setup_mapping = {
    "funnel": funnel,
    "banana": banana_target,
    "ring": ring_target,
    "squiggle": squiggle_target,
    "mix": mix_of_gauss,
}
