import argparse
import pickle
import time
from math import prod

# Typing functions
from typing import Tuple

# Import JAX and utilities
import jax
import jax.numpy as jnp
from jax.experimental import jet
from jax.tree_util import tree_flatten
from jax.scipy.special import expit as sigmoid
from jax import lax

import numpy as np

# Haiku for Neural networks
import haiku as hk

# Optax for the optimization scheme
import optax

# Import the function to compute the taylor coefficient and remainder
from taylanets.taylanets import taylor_order_n, der_order_n, midpoint_constraints

# Import the datasets to be used
import tensorflow_datasets as tfds


from tqdm.auto import tqdm

def _logaddexp(x1, x2):
  """
  Logaddexp while ignoring the custom_jvp rule.
  """
  amax = lax.max(x1, x2)
  delta = lax.sub(x1, x2)
  return lax.select(jnp.isnan(delta),
                    lax.add(x1, x2),  # NaNs or infinities of the same sign.
                    lax.add(amax, lax.log1p(lax.exp(-lax.abs(delta)))))


softplus = lambda x: _logaddexp(x, jnp.zeros_like(x))

# set up modules
class ConcatConv2D(hk.Module):
    """
    Convolution with extra channel and skip connection for time.
    """

    def __init__(self, *args, **kwargs):
        super(ConcatConv2D, self).__init__()
        self._layer = hk.Conv2D(*args, **kwargs)

    def __call__(self, x, t):
        tt = jnp.ones_like(x[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, x], axis=-1)
        return self._layer(ttx)

def logdetgrad(x, alpha):
    """
    Log determinant grad of the logit function for propagating logpx.
    """
    s = alpha + (1 - 2 * alpha) * x
    logdetgrad_ = -jnp.log(s - s * s) + jnp.log(1 - 2 * alpha)
    return jnp.sum(jnp.reshape(logdetgrad_, (x.shape[0], -1)), axis=1, keepdims=True)


def get_epsilon(key, shape):
    """
    Sample epsilon from the desired distribution.
    """
    # rademacher
    return jax.random.randint(key, shape, minval=0, maxval=2).astype(jnp.float32) * 2 - 1

# def softmax_cross_entropy(logits, labels):
#     """
#     Cross-entropy loss applied to softmax.
#     """
#     one_hot = hk.one_hot(labels, logits.shape[-1])
#     return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


class ForwardPreODE(hk.Module):
    """
    Module applied before the ODE layer.
    """

    def __init__(self, alpha=1e-5):
        super(ForwardPreODE, self).__init__()
        self.alpha = alpha

    def __call__(self, x, logpx):
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = jnp.log(s) - jnp.log(1 - s)
        logpy = logpx - logdetgrad(x, self.alpha)
        return y, logpy


class ReversePreODE(hk.Module):
    """
    Inverse of module applied before the ODE layer.
    """

    def __init__(self, alpha=1e-6):
        super(ReversePreODE, self).__init__()
        self.alpha = alpha

    def __call__(self, y, logpy):
        x = (sigmoid(y) - self.alpha) / (1 - 2 * self.alpha)
        logpx = logpy + logdetgrad(y, self.alpha)
        return x, logpx


class NN_Dynamics(hk.Module):
    """
    NN_Dynamics of the ODENet.
    """

    def __init__(self,
                 hidden_dims=(64, 64, 64),
                 input_shape=(28, 28, 1),
                 strides=(1, 1, 1, 1)):
        super(NN_Dynamics, self).__init__()
        self.input_shape = input_shape
        layers = []
        activation_fns = []
        base_layer = ConcatConv2D
        nonlinearity = softplus

        for layer_num, (dim_out, stride) in enumerate(zip(hidden_dims + (input_shape[-1],), strides)):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"kernel_shape": 3, "stride": 1, "padding": lambda _: (1, 1)}
            elif stride == 2:
                layer_kwargs = {"kernel_shape": 4, "stride": 2, "padding": lambda _: (1, 1)}
            elif stride == -2:
                # note: would need to use convtranspose instead here
                layer_kwargs = {"kernel_shape": 4, "stride": 2, "padding": lambda _: (1, 1), "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            if layer_num == len(strides) - 1:
                layer_kwargs["w_init"] = jnp.zeros  # initialize last layer to zeros (bias is always init to 0)
            layer = base_layer(dim_out, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(nonlinearity)

        self.layers = layers
        self.activation_fns = activation_fns[:-1]

    def __call__(self, x, t):
        x = jnp.reshape(x, (-1, *self.input_shape))
        dx = x
        for l, layer in enumerate(self.layers):
            dx = layer(dx, t)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx


class Midpoint(hk.Module):
    """
    Compute the coefficients in the formula obtained to simplify the learning of the midpoint
    """
    def __init__(self, dim, output_sizes, approx_mid=True):
        """ Build a MLP approximating the coefficient in the midpoint formula
            :param dim : Specifies the flatten dimension (input + time step)
            :param output_sizes : Size of the hidden layer
        """
        super(Midpoint, self).__init__()
        self.approx_mid = approx_mid
        self.dim = dim
        # Initialize the weight to be randomly close to zero
        self.model = hk.nets.MLP(output_sizes=(*output_sizes, dim), 
                        w_init=hk.initializers.RandomUniform(minval=0, maxval=0), 
                        b_init=jnp.zeros, activation=jax.numpy.tanh)

    def __call__(self, xt):
    	if self.approx_mid:
    		return self.model(xt)
    	else:
    		return 0.0



def wrap_module(module, *module_args, **module_kwargs):
    """
    Wrap the module in a function to be transformed.
    """
    def wrap(*args, **kwargs):
        """
        Wrapping of module.
        """
        model = module(*module_args, **module_kwargs)
        return model(*args, **kwargs)
    return wrap

def flatten_image(_image, dim=28*28):
    return _image.reshape((-1, dim))

def inv_flatten_image(_flat_image, dim= (28,28,1)):
    return _flat_image.reshape((-1,*dim))

def aug_init(y):
    """
    Initialize dynamics with 0 for logpx.
    """
    batch_size = y.shape[0]
    return y, jnp.zeros((batch_size, 1))

def standard_normal_logprob(z):
    """
    Log probability of standard normal.
    """
    logz = -0.5 * jnp.log(2 * jnp.pi)
    return logz - jnp.square(z) / 2

@jax.jit
def _loss_fn(z, delta_logp):
    logpz = jnp.sum(jnp.reshape(standard_normal_logprob(z), (z.shape[0], -1)), axis=1, keepdims=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = jnp.sum(logpx) / z.size  # averaged over batches
    bits_per_dim = -(logpx_per_dim - jnp.log(256)) / jnp.log(2)

    return bits_per_dim


# function assumes ts = [0, 1]
def init_model(rng, taylor_order, number_step, batch_size=1, optim=None, 
            midpoint_layers=(12,12), count_nfe= None, pen_midpoint=0.0, 
            pen_remainder= 0.0, approx_mid=True, method='tayla'):
    """ Initialize the model and return the necessary functions to evaluate,
        update, and compute loss given batch of data
        :param rng             : A random key generator used to initialize the haiku module
        :param                 : taylor_order : The order of the taylor expansion : 0 mean the midpoint is evaluated at f,
                                    1 means the midpoint is evaluatated at (df/dx f) and so on ...
        :param number_step     : The number of intermediate point between 0 and 1.0 --> proxy for time step
        :param batch_size      : Initial batch size to initialize the haiku modules
        :param optim           : The optimizer to use to build the update function
        :param midpoint_layers : The hidden layers of the coefficient in the midpoint formula
        :param count_nfe       : if not None, Generate a function counting the number of neural network evaluation  
                                 and solving the ode when using odeint. count_nfe[1] = atol and count_nfe[2] = rtol
        :param pen_midpoint    : Penalty coefficient to ensure this is a midpoint value
        :param pen_remainder   : Penalty coefficient to ensure the remainder is small (in theory it is and this avoid overfitting)
        :param approx_mid      : Specify if this module should approximate the midpoint value
    """
    # Integration time -> initial time is 0 and end time is 1
    ts = jnp.array([0., 1.])

    # Discretize the integration time using the given number of step
    time_step = float((ts[1]-ts[0]) / number_step)
    time_indexes = jnp.array([ time_step * (i+1) for i in range(number_step)])

    # MNIST problem size and dummy random intiial values
    image_shape = (28, 28, 1)
    ode_shape = prod(image_shape)
    midpoint_shape = ode_shape + 1 + 1 # An additional for logp and time
    pre_ode_init = jnp.zeros((batch_size, *image_shape))
    delta_logp_init = jnp.zeros((batch_size, 1))
    ode_init, t_init = jnp.zeros((batch_size, ode_shape)), 0.0
    midpoint_init = jnp.zeros((batch_size, midpoint_shape))

    # Build the pre module
    pre_ode = hk.without_apply_rng(hk.transform(wrap_module(ForwardPreODE)))
    pre_ode_params = pre_ode.init(rng, pre_ode_init, delta_logp_init)
    pre_ode_fn = pre_ode.apply

    # Build the ODE function module
    dynamics = hk.without_apply_rng(hk.transform(wrap_module(NN_Dynamics)))
    dynamics_params = dynamics.init(rng, ode_init, t_init)
    dynamics_wrap = dynamics.apply

    # Build the Midpoint function module
    midpoint = hk.without_apply_rng(hk.transform(wrap_module(Midpoint, midpoint_shape, midpoint_layers, approx_mid)))
    midpoint_params = midpoint.init(rng, midpoint_init)
    midpoint_wrap = midpoint.apply

    def ffjord_dynamics(yp, t, eps, params):
        """
        Dynamics of augmented ffjord state. --> asume the input is flatten
        """
        # y = yp[...,:-1] # remove logp value -> last value on last axis
        y = inv_flatten_image(yp[...,:-1])
        f = lambda y: dynamics_wrap(params, y, t)
        dy, eps_dy = jax.jvp(f, (y,), (eps,))
        div = jnp.sum(jnp.reshape(eps_dy * eps, (y.shape[0], -1)), axis=1, keepdims=True)
        return flatten_image(dy), -div

    def ffjord_dynamics_nfe(yp, t, eps, params):
        """
        Dynamics of augmented ffjord state.
        """
        y, p = yp
        f = lambda y: dynamics_wrap(y, t, params)
        dy, eps_dy = jax.jvp(f, (y,), (eps,))
        div = jnp.sum(jnp.reshape(eps_dy * eps, (y.shape[0], -1)), axis=1, keepdims=True)
        return dy, -div

    # Define the ODE prediction function
    inv_m_fact = jet.fact(jnp.array([i+1 for i in range(taylor_order+1)]))
    inv_m_fact = 1.0 / inv_m_fact
    dt_square_over_2 = time_step*time_step*0.5
    pow_timestep = [time_step]
    for _ in range(taylor_order):
        pow_timestep.append(pow_timestep[-1] * time_step)
    pow_timestep = jnp.array(pow_timestep) * inv_m_fact
    rem_coeff = pow_timestep[-1]
    # Divide the power series of the time step by factorial of the derivative order --> Do some reshaping for broadcasting issue
    pow_timestep = pow_timestep[:-1].reshape((-1,1,1))
    def pred_xnext(params : Tuple[hk.Params, hk.Params], state_t : jnp.ndarray, eps : jnp.ndarray): #, t : float) -> jnp.ndarray:
        """ Predict the next using our Taylor-Lagrange expansion with learned remainder"""
        params_dyn, params_mid = params

        # Define the vector field of the ODE given a pre-process input
        def vector_field(state_val : jnp.ndarray) -> jnp.ndarray:
            """ Function computing the vector field"""
            xstate, curr_t = state_val[...,:-1], state_val[...,-1:]
            dy, dlog = ffjord_dynamics(xstate, curr_t.ravel()[0], eps, params_dyn)# The time is the same for all layers so we reused it
            curr_dt = jnp.ones_like(curr_t)
            return jnp.concatenate((dy, dlog, curr_dt), axis=-1)

        # Merge the state value and time
        # state_t = jnp.concatenate((state, jnp.ones_like(state[...,:1])*t), axis=-1)

        # Compute the vector field as it is used recursively in jet and the midpoint value
        vField = vector_field(state_t)

        # Compute the midpoint coefficient 
        midpoint_val = state_t + midpoint_wrap(params_mid, state_t) * vField

        if taylor_order == 0: # Simple classical single layer DNNN
            rem_term = time_step * vector_field(midpoint_val)
            next_state = state_t

        elif taylor_order == 1: # The first order is faster and easier to implement with jvp
            next_state = state_t + time_step * vField
            rem_term = dt_square_over_2 * jax.jvp(vector_field, (midpoint_val,), (vector_field(midpoint_val),))[1]

        elif taylor_order == 2: # The second order can be easily and fastly encode too
            next_state = state_t + time_step * vField + dt_square_over_2 * jax.jvp(vector_field, (state_t,), (vField,))[1]
            # Add the remainder at the midpoint
            rem_term = rem_coeff * der_order_n(midpoint_val, vector_field, taylor_order)

        else:
            # Compiute higher order derivatives
            m_expansion = taylor_order_n(state_t, vector_field, taylor_order-1, y0=vField)  
            next_state = state_t + jnp.sum(pow_timestep * m_expansion, axis=0)
            # Remainder term
            rem_term = rem_coeff * der_order_n(midpoint_val, vector_field, taylor_order)

        # Add the remainder term of the Taylor expansion at the midpoint
        next_state += rem_term

        return next_state, midpoint_val, rem_term

    # Define a forward function to compute the logits
    # @jax.jit
    def forward_aux(key, params, _images, midpoint_compute):
        (_pre_ode_params, _dynamics_params, _midpoint_params) = params
        eps = get_epsilon(key, _images.shape)
        out_pre_ode, detla_logp = pre_ode_fn(_pre_ode_params, *aug_init(_images))
        out_pre_ode = flatten_image(out_pre_ode)
        state_logp = jnp.concatenate((out_pre_ode, detla_logp), axis=-1)
        # Build the iteration loop for the ode solver when step size is not 1
        def rollout(carry, extra):
            state_t = jnp.concatenate((carry, jnp.ones_like(carry[...,:1])*extra), axis=-1)
            if (not midpoint_compute):
                next_x, _, _ = pred_xnext((_dynamics_params, _midpoint_params), state_t, eps) # carry , extra)
                return next_x[...,:-1], None
            else:
                next_x, currmidpoint, remTerm = pred_xnext((_dynamics_params, _midpoint_params), state_t, eps) #, carry, extra)
                midpoint_constr = midpoint_constraints(currmidpoint, state_t, next_x)
                midpoint_constr_val = jnp.mean(jnp.where(midpoint_constr > 0, 1.0, 0.0) * jnp.abs(midpoint_constr))
                rem_constr = jnp.mean(jnp.abs(remTerm))
                return next_x[...,:-1], jnp.array([midpoint_constr_val, rem_constr])

        if not midpoint_compute:
            # Loop over the grid time step
            out_ode, _ = jax.lax.scan(rollout, state_logp, time_indexes)
            return inv_flatten_image(out_ode[...,:-1]), out_ode[...,-1:]
        else:
            # Loop over the grid time step
            out_ode, m_constr = jax.lax.scan(rollout, state_logp, time_indexes)
            # Post process and return the result
            return inv_flatten_image(out_ode[...,:-1]), out_ode[...,-1:], jnp.mean(m_constr, axis=0)

    # The forward function with
    forward_loss = jax.jit(lambda key, params, _images : forward_aux(key, params, _images, midpoint_compute=True))
    # THe forward function without computing the contraint on the midpoint
    forward_nooverhead = jax.jit(lambda key, params, _images : forward_aux(key, params, _images, midpoint_compute=False))
    m_forward = (forward_nooverhead, forward_loss)
    # Regroup the parameters of this entire module
    m_params = (pre_ode_params, dynamics_params, midpoint_params)

    # Define the loss function
    # @jax.jit
    def loss_fun(params, _images, key):
        """ Compute the loss function of the prediction method
        """
        # Compute the loss function
        z, delta_logp, mpoint_constr = forward_loss(key, params, _images)
        loss_ = _loss_fn(z, delta_logp)
        return loss_ + pen_midpoint * mpoint_constr[0] + pen_remainder * mpoint_constr[1]


    # Define a function to predict next state using fixed time step rk4
    if method != 'tayla':
        from examples_taylanets.jinkelly_lib.lib_ode.ode import odeint_grid, odeint
        if method == 'odeint_grid':
            nodeint_aux = lambda y0, ts, params: odeint_grid(lambda _y, _t, _params : dynamics_wrap(_params, _y, _t), y0, ts, params, step_size=time_step)
        elif method == 'odeint':
            nodeint_aux = lambda y0, ts, params: odeint(lambda _y, _t, _params : dynamics_wrap(_params, _y, _t), y0, ts, params, atol=args.atol, rtol=args.rtol)
        else:
            raise Exception('{} not implemented yet !'.format(method))
        @jax.jit
        def forward(params, _images):
            (_pre_ode_params, _dynamics_params, _post_ode_params) = params
            out_pre_ode = pre_ode_fn(_pre_ode_params, _images)
            out_ode, f_nfe = nodeint_aux(out_pre_ode, ts, _dynamics_params)
            return post_ode_fn(_post_ode_params, out_ode[-1]), jnp.mean(f_nfe)
        # @jax.jit
        def loss_fun(params, _images, _labels):
            logits, _ = forward(params, _images)
            return _loss_fn(logits, _labels)
        m_forward = forward
        # Regroup the parameters of this entire module
        m_params = (pre_ode_params, dynamics_params, post_ode_params)

    # Define the update function
    grad_fun = jax.grad(loss_fun, has_aux=False)

    @jax.jit
    def update(params, opt_state, key, _images):
        grads = grad_fun(params, _images, key)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # If it is required to compute the number of network evaluation 
    # that an adaptive solver will take, then do it
    nfe_fun = None
    if count_nfe is not None:
        from examples_taylanets.jinkelly_lib.lib_ode.ode import odeint
        def dyn_aux(z_px, t, eps, params):
            dz, dlogp = ffjord_dynamics(z_px, t, eps, params)
            return z, dlogp

        @jax.jit
        def nfe_fun(key, params, _images):
            eps = get_epsilon(key, _images.shape)
            out_pre_ode, detla_logp = pre_ode_fn(params[0], *aug_init(_images))
            (out_ode, out_logp), f_nfe = odeint(dyn_aux, (out_pre_ode,detla_logp), ts, eps, params[1], atol=count_nfe[0], rtol=count_nfe[1])
            # -1 here is to take the last element in the integration scheme
            return out_ode[-1], out_logp[-1], jnp.mean(f_nfe)

    return m_params, m_forward, loss_fun, update, nfe_fun


def init_data(train_batch_size, test_batch_size, seed_number=0, shuffle=1000, validation_set=False):
    """
    Initialize data from tensorflow dataset
    """
    # Import and cache the file in the current directory
    ds_data, ds_info = tfds.load('mnist',
                                     data_dir='./tensorflow_data/',
                                     split=['train'] if not validation_set else ['train', 'test'],
                                     shuffle_files=True,
                                     as_supervised=True,
                                     with_info=True,
                                     read_config=tfds.ReadConfig(shuffle_seed=seed_number, try_autocache=False))

    if validation_set:
        ds_train, ds_test = ds_data
    else:
        ds_train, ds_test = ds_data[0], ds_data[0]

    num_train = ds_info.splits['train'].num_examples
    num_test = ds_info.splits['test'].num_examples if validation_set else num_train

    assert num_train % train_batch_size == 0
    num_train_batches = num_train // train_batch_size

    assert num_test % test_batch_size == 0
    num_test_batches = num_test // test_batch_size

    # Make the data set loopable mainly for testing loss evalutaion
    ds_train = ds_train.cache()
    ds_train = ds_train.repeat()
    ds_train = ds_train.shuffle(shuffle, seed=seed_number)

    # ds_test = ds_test.cache()
    # ds_test = ds_test.repeat()
    # ds_test = ds_test.shuffle(shuffle, seed=seed_number)

    ds_train, ds_test = ds_train.batch(train_batch_size), ds_test.batch(test_batch_size).repeat()
    ds_train, ds_test = tfds.as_numpy(ds_train), tfds.as_numpy(ds_test)

    meta = {
        "num_train_batches": num_train_batches,
        "num_test_batches": num_test_batches
    }

    # Return iter element on the training and testing set
    return iter(ds_train), iter(ds_test), meta


@jax.jit
def sep_losses(out_ode, _labels):
    """ Convenience function for calculation the losses
    """
    return _loss_fn(out_ode, _labels), _acc_fn(out_ode, _labels)


if __name__ == "__main__":
    # Parse the command line argument
    parser = argparse.ArgumentParser('FFJORd MNIST')
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--taylor_order', type=int, default=1)
    parser.add_argument('--pen_midpoint', type=float, default=0)
    parser.add_argument('--pen_remainder', type=float, default=0)
    parser.add_argument('--lr_init', type=float, default=1e-2)
    parser.add_argument('--lr_end', type=float, default=1e-2)
    parser.add_argument('--w_decay', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=1e-2)
    parser.add_argument('--nepochs', type=int, default=160)
    parser.add_argument('--test_freq', type=int, default=3000)
    parser.add_argument('--save_freq', type=int, default=3000)
    parser.add_argument('--dirname', type=str, default='neur_train')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--count_odeint_nfe',  action="store_true")
    parser.add_argument('--method',  type=str, default='tayla') # choices odeint or odeint_grid
    # parser.add_argument('--validation_set',  action="store_true")
    parser.add_argument('--no_midpoint',  action="store_false")
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--num_steps', type=int, default=2)
    parser.add_argument('--patience', type=int, default=-1) # For early stopping criteria
    args = parser.parse_args()

    # Generate the random key generator -> A workaround for a weird jax + tensorflow core dumped error if it is placed after init_data
    rng = jax.random.PRNGKey(args.seed)

    # Initialize the MNIST data set
    ds_train, ds_train_eval, meta = init_data(args.train_batch_size, args.test_batch_size, seed_number=args.seed, validation_set=True)
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    print('Meta information learning : \n', meta)
    print(vars(args))

    ############## Build the optimizer ##################
    # Customize the gradient descent algorithm
    chain_list = [optax.scale_by_adam()]

    # Add weight decay if enable
    decay_weight = args.w_decay
    if decay_weight > 0.0:
        chain_list.append(optax.add_decayed_weights(decay_weight))

    # Add scheduler for learning rate decrease --> Linear decrease given bt learning_rate_init and learning_rate_end
    m_schedule = optax.linear_schedule(-args.lr_init, -args.lr_end, args.nepochs*meta['num_train_batches'])
    # m_schedule = optax.piecewise_constant_schedule(-args.lr_init, {50000 : 1e-1, 60000 : 0.5, 70000 : 0.5})
    chain_list.append(optax.scale_by_schedule(m_schedule))

    # Add gradient clipping if enable
    grad_clip_coeff = args.grad_clip
    if grad_clip_coeff > 0.0:
        chain_list.append(optax.adaptive_grad_clip(clipping=grad_clip_coeff))

    # Build the solver finally
    opt = optax.chain(*chain_list)

    # Build the solver
    _count_nfe = None if not args.count_odeint_nfe else (args.atol, args.rtol)
    midpoint_hidden_layer_size = (12,12) # Hidden layers of the midpoint neural network
    m_params, forward_mixture, loss_fun, update, nfe_fun = \
                    init_model(rng, args.taylor_order, args.num_steps, batch_size=train_batch_size, 
                                optim=opt, midpoint_layers=midpoint_hidden_layer_size, count_nfe=_count_nfe, 
                                pen_midpoint=args.pen_midpoint, pen_remainder = args.pen_remainder, 
                                approx_mid = args.no_midpoint, method= args.method)


    # Initialize the state of the optimizer
    opt_state = opt.init(m_params)

    # Save the number of iteration
    itr_count = 0

    # Save the optimal loss_value obtained
    opt_params_dict = None
    opt_loss_train = None
    opt_loss_test = None
    opt_loss_odeint = None
    opt_constr_mid_evol_test = None
    opt_constr_mid_evol_train = None
    opt_constr_rem_evol_test = None
    opt_constr_rem_evol_train = None
    iter_since_opt_test =  None

    total_compute_time = 0.0
    loss_evol_train = list()
    loss_evol_test = list()
    loss_evol_odeint = list()
    predtime_evol_train = list()
    predtime_evol_test = list()
    predtime_evol_odeint = list()
    compute_time_update = list()
    nfe_evol_train = list()
    nfe_evol_test = list()
    nfe_evol_odeint = list()
    constr_mid_evol_train = list()
    constr_mid_evol_test = list()
    constr_rem_evol_train = list()
    constr_rem_evol_test = list()

    m_parameters_dict = vars(args)

    def evaluate_loss(params, forward_fun, key, data_eval, num_iter, is_taylor=True):
        """ 
            Evaluate the loss function, accuracy, number of function evaluation given
            a forward function
            :param params : The parameters of the neural networks
            :param forward_fun : A forward function to compute the logits
            :param data_eval : The data set (should be iteratble) to use when computing the logitsout tayla expansion
            :param data_eval : The number of iteration to go through the full data set
            :param is_taylor : Specify if the forward function is coming from 
        """
        loss_values = list()
        pred_time = list()
        nfe_val = list()
        lossmidpoint = list()
        lossrem = list()
        funEValTaylor = (args.taylor_order+1)*np.log((args.taylor_order+1)) # or should be order**2

        if is_taylor:
            _, forward_loss_ = forward_fun
            def forward_loss(_key, _images):
                x, p, l = forward_loss_(_key, params, _images)
                return x, p, funEValTaylor, l
        else:
            forward_loss = lambda _key, _images : (*forward_fun(_key, params, _images), (0,0))

        for _ in tqdm(range(num_iter),leave=False):
            _key, _key2 = jax.random.split(key, num=2)
            # Extract the current data
            _images, _labels = next(data_eval)
            _images = (_images.astype(jnp.float32) + jax.random.uniform(_key2,
                                                                    minval=1e-15,
                                                                    maxval=1 - 1e-15,
                                                                    shape=_images.shape)) / 256.


            # Compute the loss and accuracy of the of obtained logits
            curr_time = time.time()
            z, probz, number_fun_eval, mconstr_loss = forward_loss(_key, _images)
            z.block_until_ready()
            diff_time  = time.time() - curr_time

            lossval = _loss_fn(z, probz)
            lossmidpoint_val, loss_rem = mconstr_loss[0], mconstr_loss[1]

            # Save the data
            pred_time.append(diff_time)
            loss_values.append(lossval)
            nfe_val.append(number_fun_eval)
            lossmidpoint.append(lossmidpoint_val)
            lossrem.append(loss_rem)
        return jnp.mean(jnp.array(loss_values)), jnp.mean(jnp.array(pred_time)), jnp.mean(jnp.array(nfe_val)), jnp.mean(jnp.array(lossmidpoint)), jnp.mean(jnp.array(lossrem))

    # Open the info file to save the command line print
    outfile = open(args.dirname+'_info.txt', 'w')
    outfile.write('////// Command line messages \n\n\n')
    outfile.close()

    key = rng

    # Start the iteration loop
    for epoch in tqdm(range(args.nepochs)):
        for i in tqdm(range(meta['num_train_batches']), leave=False):
            key, key2 = jax.random.split(key, num=2)
            # Get the next batch of images
            _images, _ = next(ds_train)
            _images = (_images.astype(jnp.float32) + jax.random.uniform(key2,
                                                                    minval=1e-15,
                                                                    maxval=1 - 1e-15,
                                                                    shape=_images.shape)) / 256.

            # Increment the iteration count
            itr_count += 1

            # Convert the image into float 
            # _images = _images.astype(m_dtype)

            # Update the weight of each neural networks and evaluate the compute time
            update_start = time.time()
            m_params, opt_state = update(m_params, opt_state, key, _images,)
            tree_flatten(opt_state)[0][0].block_until_ready()
            update_end = time.time() - update_start

            # [LOG] Update the compute time data
            compute_time_update.append(update_end)

            # [LOG] Total elapsed compute time for update only
            if itr_count >= 5: # Remove the first few steps due to jit compilation
                total_compute_time += update_end

            # Check if it is time to evaluate the function
            if itr_count % args.test_freq == 0:
                # Compute the loss function over the entire training set
                print_str_test = '--------------------------------- Eval on Test Data [epoch={} | num_batch = {}] ---------------------------------\n'.format(epoch, i)
                tqdm.write(print_str_test)
                # loss_values_train, pred_time_train, nfe_train, contr_mid_train, contr_rem_train = evaluate_loss(m_params, forward_mixture, key, ds_train, meta['num_train_batches'], is_taylor = args.method == 'tayla')
                # Compute the loss on the testing set if it is different from the training set
                loss_values_test, pred_time_test, nfe_test, contr_mid_test, contr_rem_test = evaluate_loss(m_params, forward_mixture, key, ds_train_eval, meta['num_test_batches'], is_taylor = args.method == 'tayla')
                # Compute the loss using odeint on the test data
                loss_values_odeint, pred_time_odeint, nfe_odeint = 0, 0, 0
                if nfe_fun is not None:
                     loss_values_odeint, pred_time_odeint, nfe_odeint, _, _ = evaluate_loss(m_params, nfe_fun, key, ds_train_eval, meta['num_test_batches'], is_taylor=False)

                # First time we have a value for the loss function
                if opt_loss_test is None or (opt_loss_test < loss_values_test):
                    opt_loss_test = loss_values_test
                    # opt_loss_train = loss_values_train
                    opt_loss_odeint = loss_values_odeint
                    opt_nfe_test = nfe_test
                    # opt_nfe_train = nfe_train
                    opt_nfe_odeint = nfe_odeint
                    opt_constr_mid_evol_test = contr_mid_test
                    # opt_constr_mid_evol_train = contr_mid_train
                    opt_constr_rem_evol_test = contr_rem_test
                    # opt_constr_rem_evol_train = contr_rem_train
                    opt_params_dict = m_params
                    iter_since_opt_test = epoch # Start counting the number of times we evaluate the learned model

                # # Check if we have improved the loss function on the test only
                # if opt_loss_test > loss_values_test:
                #     opt_loss_test = loss_values_test
                #     opt_loss_train = loss_values_train
                #     opt_loss_odeint = loss_values_odeint
                #     opt_accuracy_test = acc_values_test
                #     opt_accuracy_train = acc_values_train
                #     opt_accuracy_odeint = acc_values_odeint
                #     opt_params_dict = m_params
                #     iter_since_opt_test = epoch

                # Do some printing for result visualization
                print_str = 'Iter {:05d} | Total Update Time {:.2f} | Update time {}\n\n'.format(itr_count, total_compute_time, update_end)
                print_str += 'Loss Test {:.2e} | Loss ODEINT {:.6f}\n'.format(loss_values_test, loss_values_odeint)
                print_str += 'OPT Loss Test {:.2e} | OPT Loss ODEINT {:.2e}\n\n'.format(opt_loss_test, opt_loss_odeint)               
                print_str += 'NFE Test {:.2f} | NFE ODEINT {:.2f}\n'.format(nfe_test, nfe_odeint)
                print_str += 'OPT NFE Test {:.2f} | OPT NFE ODEINT {:.2f}\n\n'.format(opt_nfe_test, opt_nfe_odeint)
                print_str += 'Pred Time Test {:.2e} | Pred Time ODEINT {:.2e}\n\n'.format(pred_time_test, pred_time_odeint)
                print_str += 'Midpoint Constr Test {:.2e} | OPT Midpoint Constr Test {:.2e} \n\n'.format(contr_mid_test, opt_constr_mid_evol_test)
                print_str += 'Remainder Constr Test {:.2e} | OPT Remainder Constr Test {:.2e} \n\n'.format(contr_rem_test, opt_constr_rem_evol_test)
                tqdm.write(print_str)

                # Save all the obtained data
                # loss_evol_train.append(loss_values_train)
                loss_evol_test.append(loss_values_test)
                loss_evol_odeint.append(loss_values_odeint)
                # predtime_evol_train.append(pred_time_train)
                predtime_evol_test.append(pred_time_test)
                predtime_evol_odeint.append(pred_time_odeint)
                # nfe_evol_train.append(nfe_train)
                nfe_evol_test.append(nfe_test)
                nfe_evol_odeint.append(nfe_odeint)
                # constr_mid_evol_train.append(contr_mid_train)
                constr_mid_evol_test.append(contr_mid_test)
                # constr_rem_evol_train.append(contr_rem_train)
                constr_rem_evol_test.append(contr_rem_test)

                # Save these info in a file
                outfile = open(args.dirname+'_info.txt', 'a')
                outfile.write(print_str_test)
                outfile.write(print_str)
                outfile.close()

            if itr_count % args.save_freq == 0 or (epoch == args.nepochs-1 and i == meta['num_train_batches']-1):
                m_dict_res = {'best_params' : opt_params_dict, 'total_update_time' : total_compute_time, 'updatetime_evol' : compute_time_update,
                                'opt_loss_train' : opt_loss_train, 'opt_loss_test' : opt_loss_test, 'opt_loss_odeint' : opt_loss_odeint, 
                                'opt_nfe_train' : opt_nfe_train, 'opt_nfe_test' : opt_nfe_test, 'opt_nfe_odeint' : opt_nfe_odeint, 
                                'loss_evol_train' : loss_evol_train, 'loss_evol_test' : loss_evol_test, 'loss_evol_odeint' : loss_evol_odeint, 
                                'predtime_evol_train' : predtime_evol_train, 'predtime_evol_test' : predtime_evol_test, 'predtime_evol_odeint' : predtime_evol_odeint, 
                                'nfe_evol_train' : nfe_evol_train, 'nfe_evol_test' : nfe_evol_test, 'nfe_evol_odeint' : nfe_evol_odeint, 
                                'constr_mid_evol_train' : constr_mid_evol_train, 'constr_mid_evol_test' : constr_mid_evol_test, 'opt_constr_mid_evol_train' : opt_constr_mid_evol_train, 'opt_constr_mid_evol_test' : opt_constr_mid_evol_test,
                                'constr_rem_evol_train' : constr_rem_evol_train, 'constr_rem_evol_test' : constr_rem_evol_test, 'opt_constr_rem_evol_train' : opt_constr_rem_evol_train, 'opt_constr_rem_evol_test' : opt_constr_rem_evol_test,
                                'training_parameters' : m_parameters_dict}
                outfile = open(args.dirname+'.pkl', "wb")
                pickle.dump(m_dict_res, outfile)
                outfile.close()

        # If early stopping is enabled and no improvement since patience number of epoch stop
        if args.patience > -1 and (epoch - iter_since_opt_test >= args.patience):
            break
