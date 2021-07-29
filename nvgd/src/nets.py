import jax.numpy as jnp
from jax import vmap, grad, random
import jax
import haiku as hk
from . import kernels
from collections import Mapping


def bandwidth_init(shape, dtype=jnp.float32):
    """Init for bandwith matrix"""
    d = shape[0]
    return jnp.identity(d, dtype)


def debug_layer(debug=False, out: str = ''):
    def report_shape(x):
        if debug:
            print(out)
            print(x.shape)
        return x
    return report_shape


class RBFKernel(hk.Module):
    def __init__(self, scale_param=False, parametrization="diagonal", name=None):
        """
        * If params='diagonal', use one scalar bandwidth parameter per dimension,
        i.e. parameters habe shape (d,).
        * If params=log_diagonal, same but parametrize log(bandwidth)
        * If params='full', parametrize kernel using full (d, d) matrix.
        Params are initialized st the three options are equivalent at initialization."""
        super().__init__(name=name)
        self.parametrization = parametrization
        self.scale_param = scale_param

    def __call__(self, xy):
        """xy should have shape (2, d)"""
        d = xy.shape[-1]
        scale = hk.get_parameter("scale", shape=(), init=jnp.ones) if self.scale_param else 1.
        if self.parametrization == "log_diagonal":
            log_bandwidth = hk.get_parameter("log_bandwidth", shape=(d,), init=jnp.zeros)
            log_bandwidth = jnp.clip(log_bandwidth, a_min=-5, a_max=5)
            return scale * kernels.get_rbf_kernel_logscaled(log_bandwidth)(*xy)
        elif self.parametrization == "diagonal":
            bandwidth = hk.get_parameter("bandwidth", shape=(d,), init=jnp.ones)
            return scale * kernels.get_rbf_kernel(bandwidth)(*xy)
        elif self.parametrization == "full":
            sigma = hk.get_parameter("sigma", shape=(d, d), init=bandwidth_init)
            return scale * kernels.get_multivariate_gaussian_kernel(sigma)(*xy)


class DeepKernel(hk.Module):
    def __init__(self, sizes, name=None):
        super().__init__(name=name)
        self.sizes = sizes

    def __call__(self, x):
        """x should have shape (2, d)"""
        k = RBFKernel(scale_param=True, parametrization="full")
        net = hk.nets.MLP(output_sizes=self.sizes,
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.swish,
                          activate_final=False)
        return k(net(x))


def get_norm(init_x):
    mean = jnp.mean(init_x, axis=0)
    std = jnp.std(init_x, axis=0)

    def norm(x):
        return (x - mean) / (std + 1e-5)
    return norm


class MLP(hk.Module):
    def __init__(self, sizes: list, name: str = None):
        """
        Take care to choose sizes[-1] equal to the particle dimension.
        init_x should have shape (n, d)
        """
        super().__init__(name=name)
        self.sizes = sizes

    def __call__(self, x: jnp.ndarray, dropout: bool = False):
        """
        args:
            x: a batch of particles of shape (n, d) or a single particle
        of shape (d,)
            dropout: bool; apply dropout to output?
        """
        mlp = hk.nets.MLP(output_sizes=self.sizes,
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.swish,
                          activate_final=False)
        output = mlp(x)
        if dropout:
            output = hk.dropout(
                rng=hk.next_rng_key(),
                rate=0.2,
                x=output
            )
        return output


class KLGrad(hk.Module):
    def __init__(self, sizes: list, logp: callable, name: str = None):
        """
        Take care to choose sizes[-1] equal to the particle dimension.
        init_x should have shape (n, d)
        """
        super().__init__(name=name)
        self.sizes = sizes
        self.logp = logp

    def __call__(self, x):
        """x is a batch of particles of shape (n, d) or a single particle
        of shape (d,)"""
        assert x.shape[-1] == self.sizes[-1]
        mlp = hk.nets.MLP(output_sizes=self.sizes,
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.swish,
                          activate_final=False)
        s = hk.get_parameter("scale", (), init=jnp.ones)
        if x.ndim == 1:
            return mlp(x) - s*grad(self.logp)(x)
        elif x.ndim == 2:
            return mlp(x) - s*vmap(grad(self.logp))(x)
        else:
            raise ValueError("Input needs to have rank 1 or 2.")


def build_mlp(sizes, name=None, skip_connection=False,
              with_bias=True, activate_final=False):
    """
    * sizes is a list of integers representing layer dimension

    Network uses He initalization; see https://github.com/deepmind/dm-haiku/issues/6
    and https://sonnet.readthedocs.io/en/latest/api.html#variancescaling.
    """
    def mlp(x):
        lin = hk.nets.MLP(output_sizes=sizes,
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.swish,
                          activate_final=activate_final,
                          with_bias=with_bias,
                          name=name)
        if skip_connection is False:
            return lin(x)
        else:
            return lin(x) + x  # make sure sizes fit (ie sizes[-1] == input dimension)
    return hk.transform(mlp)


def hypernet_shape_utilities(conv_dict, misc_dict):
    """
    args:
        conv_dict: dictionary containing conv layer parameters. Importantly,
            every layer must have the same number of parameters (since they
            will be concatenated and fed as batched input to an MLP).
        misc_dict: dictionary containing other parameters. These are typically
            all layers that don't have the right number of parameters, such as
            the first conv layer and the final linear layer.

    returns:
        two functions (ravel, unravel) that map conv_dict and
            misc_dict to arrays and back again.
    """
    conv_params, conv_treedef = jax.tree_flatten(conv_dict)
    conv_kernel_shape = conv_params[0].shape
    _, misc_unravel = jax.flatten_util.ravel_pytree(misc_dict)

    def ravel(conv_dict, misc_dict):
        # dict to arrays
        conv_params, conv_treedef = jax.tree_flatten(conv_dict)
        conv_params = [jnp.ravel(p) for p in conv_params]
        conv_params = jnp.asarray(conv_params)

        misc_params, _ = jax.flatten_util.ravel_pytree(misc_dict)
        return conv_params, misc_params

    def unravel(conv_params_arr, misc_params_arr):
        # arrays to dict
        cp = [jnp.reshape(p, conv_kernel_shape) for p in conv_params_arr]
        conv_dict_new = jax.tree_unflatten(conv_treedef, cp)
        misc_dict_new = misc_unravel(misc_params_arr)
        return conv_dict_new, misc_dict_new

    return ravel, unravel


class StaticHypernet(hk.Module):
    def __init__(self,
                 sizes: list = [256, 256, 256],
                 embedding_size: int = 64,
                 name: str = None):
        """
        args
            sizes: sizes of the fully connected layers
            embedding_size: dimension of z (input to hypernetwork)

        Take care to choose sizes[-1] equal to the particle dimension.
        init_x should have shape (n, d)
        """
        super().__init__(name=name)
        self.sizes = sizes
        self.embedding_size = embedding_size

    def __call__(self,
                 base_params: Mapping,
                 dropout: bool = False):
        """
        args:
            base_params: Neural network parameters in dictionary. Importantly,
                the params must follow this naming scheme:
                - layers containing convnet params must be named 'conv*'
                - all other (misc) layers must be named 'misc*'
                - all layers must have either one those strings (but not both)
                    in their name.
                In addition, the conv layers must all have the same number of 
                parameters, bc they will be stacked and fed in parallel to
                an MLP.
            dropout: apply dropout to output?
        
        returns:
            tuple (conv_out, misc_out) of shape corresponding to 
            the inputs.
        """
        conv_params = {k: v for k, v in base_params.items() if "conv" in k}
        misc_params = {k: v for k, v in base_params.items() if "misc" in k}

        param_ravel, param_unravel = hypernet_shape_utilities(
            conv_params,
            misc_params
        )
        conv_params, misc_params = param_ravel(conv_params, misc_params)

        num_kernels, num_params = conv_params.shape
        h = hk.nets.MLP(output_sizes=self.sizes + [num_params],
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.swish,
                          activate_final=False,
                          name="mlp_conv")

        f_misc = hk.nets.MLP(output_sizes= [128, 128, 128, len(misc_params)],
                          w_init=hk.initializers.VarianceScaling(scale=2.0),
                          activation=jax.nn.swish,
                          activate_final=False,
                          name="mlp_misc")

        z = hk.get_parameter("z",
                             shape=(num_kernels, self.embedding_size),
                             init=hk.initializers.RandomNormal())

        # we want the input to have (batched) shape
        # (num_kernels, self.embedding_size + num_params), and the
        # output to have shape (num_kernels, num_params)
        conv_out = h(jnp.hstack((conv_params, z)))  
        misc_out = f_misc(misc_params)
                                                    
        if dropout:
            conv_out = hk.dropout(
                rng=hk.next_rng_key(),
                rate=0.2,
                x=conv_out,
            )
            misc_out = hk.dropout(
                rng=hk.next_rng_key(),
                rate=0.2,
                x=misc_out,
            )
        conv_out, misc_out = param_unravel(conv_out, misc_out)
        return {**conv_out, **misc_out}


INIT_STDDEV_CNN = 0.15
NUM_CLASSES_CNN = 10
initializer = hk.initializers.RandomNormal(stddev=1 / 100)

class CNN(hk.Module):
    def __init__(self, n_channels=8, n_classes=NUM_CLASSES_CNN, depth=2, name: str = None):
        super().__init__(name=name)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth
        self.initializer = hk.initializers.RandomNormal(stddev=INIT_STDDEV_CNN)

    def __call__(self, image, debug=False): 
        """
        if debug, then print activation shapes
        """
        # TODO: output should have length self.n_classes
#        conv_layers = self.depth * [hk.Conv2D(self.n_channels,
#                                              kernel_shape=3,
#                                              w_init=self.initializer,
#                                              b_init=self.initializer,
#                                              stride=2),
#                                    jax.nn.relu]
#        convnet = hk.Sequential(conv_layers + [hk.Flatten()])

        with_bias = False
        strides = [1,2,1,2,1,2]
        names = ['misc'] + ['conv']*5
        
        conv_layers = [
            [
                hk.Conv2D(self.n_channels,
                        kernel_shape=3,
                        w_init=self.initializer,
                        b_init=self.initializer,
                        with_bias=with_bias,
                        stride=stride,
                        name=name),
                jax.nn.relu,
                debug_layer(debug),
            ]
            for stride, name in zip(strides, names)
        ]

        conv_layers = [l for layer in conv_layers for l in layer]
        convnet = hk.Sequential(conv_layers + [
            hk.Flatten(),
            hk.Linear(self.n_classes,
                      w_init=self.initializer,
                      b_init=self.initializer,
                      name='misc'),
            debug_layer(debug),
        ])

        return convnet(image)

# base convnet
cnn = hk.without_apply_rng(hk.transform(lambda image: CNN()(image)))
