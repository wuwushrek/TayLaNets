import argparse
import pickle
import time

# Typing functions
from typing import Tuple

from pathlib import Path

# Import JAX and utilities
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten
from jax import lax
from jax.config import config

# Haiku for Neural networks
import haiku as hk

# Optax for the optimization scheme
import optax

from tqdm.auto import tqdm

# Import from our module algorithms
from taylanets.tayla import hypersolver, odeint_rev, tayla, wrap_module

from math import prod

import datasets

float_64 = False

config.update("jax_enable_x64", float_64)

def _logaddexp(x1, x2):
    """
    Logad    dexp while ignoring the custom_jvp rule.
    """
    amax = lax.max(x1, x2)
    delta = lax.sub(x1, x2)
    return lax.select(jnp.isnan(delta),
                    lax.add(x1, x2),  # NaNs or infinities of the same sign.
                    lax.add(amax, lax.log1p(lax.exp(-lax.abs(delta)))))


# set up modules
class ConcatSquashLinear(hk.Module):
    """
    ConcatSquash Linear layer.
    """
    def __init__(self, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = hk.Linear(dim_out)
        self._hyper_bias = hk.Linear(dim_out, with_bias=False)
        self._hyper_gate = hk.Linear(dim_out)

    def __call__(self, x, t):
        return self._layer(x) * jax.nn.sigmoid(self._hyper_gate(t)) + self._hyper_bias(t)


def get_epsilon(key, shape):
    """
    Sample epsilon from the desired distribution.
    """
    # normal
    return jax.random.normal(key, shape)


class NN_Dynamics(hk.Module):
    """
    NN_Dynamics of the ODENet.
    """

    def __init__(self,
                 hidden_dims,
                 input_shape):
        super(NN_Dynamics, self).__init__()
        self.input_shape = input_shape
        layers = []
        activation_fns = []
        base_layer = ConcatSquashLinear

        for dim_out in hidden_dims + (input_shape[-1], ):
            layer = base_layer(dim_out)
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
    def __init__(self, dim, dt):
        """ Build a MLP approximating the coefficient in the midpoint formula
            :param dim : Specifies the flatten dimension (input + time step)
        """
        super(Midpoint, self).__init__(name='midpoint_residual')
        self.dim = dim # dim **2 when the midpoint is taken as a matrix
        self.dt = dt
        # Initialize the weight to be randomly close to zero
        self.model = hk.nets.MLP(output_sizes=(32, self.dim), 
                        w_init=hk.initializers.RandomUniform(minval=-1e-4, maxval=1e-4), 
                        b_init=jnp.zeros, activation=jax.nn.relu)

    def __call__(self, xt):
        midcoeff = self.model(xt)
        # We constrainys the time component to be between t and t+dt as given by the mean theorem value
        return jnp.concatenate((midcoeff[...,:-1], jnp.maximum(jnp.minimum(midcoeff[...,-1:], 1.0), 0.) * self.dt) , axis=-1)

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


# Define the loss on the predicted state
@jax.jit
def _pred_loss(eststate, nextstate):
    """ Compute the loss norm between the predicted next state and the next state given by the data 
        :param eststate  : Predicted state of the NN ode
        :param nextstate : Next state from the data
    """
    return jnp.mean(jnp.sum(jnp.abs(eststate - nextstate), axis=-1))


# Define the loss on the remainder term
@jax.jit
def _rem_loss(rem_tt):
    """ Compute the norm of the remainder term (or the residual of the learning)
        :param rem_tt : The remainder (at midpoint for tayla) or residual (for hypersolver)
    """
    return jnp.mean(jnp.abs(rem_tt))

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

    return -jnp.mean(logpx)  # likelihood in nats


# function assumes ts = [0, 1]
def init_model(rng, n_dims, order, n_step, method='tayla', batch_size=1, 
                    pen_remainder= 0, atol=1.4e-8, rtol=1.4e-8, ts=1.0):
    """ Build the model to learn the midpoint or predict trajectories of the system
        :param rng            : A random key generator for initializing the neural networks
        :param order          : The order of the method (This argument is useless for 'rk4', 'dopri5', 'taylor', and 'hypersolver_grid')
        :param n_step         : The number of step for integration
        :param ts             : The final time of integration
        :param batch_size     : The batch size used for training
        :param pen_remainder  : The penalization for the remainder term
        :param atol           : The absolute tolerance error
        :param rtol           : The relative tolerance error
    """
    # Get the time step
    time_step = ts / n_step

    # MNIST problem size and dummy random intiial values
    ode_shape = n_dims
    midpoint_shape = ode_shape + 1 + 1 # An additional for logp and time
    delta_logp_init = jnp.zeros((batch_size, 1))
    ode_init, t_init = jnp.zeros((batch_size, ode_shape)), jnp.zeros((batch_size, 1))
    midpoint_init = jnp.zeros((batch_size, midpoint_shape))


    # Build the ODE function module
    dynamics = hk.without_apply_rng(hk.transform(wrap_module(NN_Dynamics,input_shape=(n_dims,), hidden_dims=(n_dims*args.hdim_factor,)* args.num_layers)))
    dynamics_params = dynamics.init(rng, ode_init, t_init)
    dynamics_wrap_temp = dynamics.apply
    def dynamics_wrap(_xpt, params, eps):
        """ Images last component is the time index"""
        xstate, tval = _xpt[...,:-2], _xpt[...,-1:]
        f = lambda y: dynamics_wrap_temp(params, y, tval)
        derx, eps_dy = jax.jvp(f, (xstate,), (eps,))
        div = jnp.sum(jnp.reshape(eps_dy * eps, (xstate.shape[0], -1)), axis=1, keepdims=True)
        derstate_time = jnp.ones_like(_xpt[...,:1])
        return jnp.concatenate((derx, -div, derstate_time), axis=-1)

    # Build the Midpoint function module
    midpoint = hk.without_apply_rng(hk.transform(wrap_module(Midpoint, midpoint_shape, time_step)))
    midpoint_params = midpoint.init(rng, midpoint_init)
    midpoint_wrap = lambda x, _params : midpoint.apply(_params, x)

    pred_params = (dynamics_params, midpoint_params)

    ####################################################################################
    odeint_eval = odeint_rev(dynamics_wrap, time_step, n_step=n_step, atol=atol, rtol=rtol)
    # Build the prediction function based on the current method
    if method == 'tayla':       # Taylor lagrange with a learned remainder
        pred_fn = tayla((dynamics_wrap, midpoint_wrap), time_step, order=order, n_step=n_step)
        nb_params = 2
    elif method == 'taylor':    # Truncated Taylor expansion
        pred_fn = tayla((dynamics_wrap, ), time_step, order=order, n_step=n_step)
        nb_params = 1
    elif method == 'rk4': # Fixed time-setp RK4 method
        pred_fn = odeint_rev(dynamics_wrap, time_step, n_step=n_step)
        nb_params = 1
    elif method == 'dopri5':
        pred_fn = odeint_eval
        nb_params = 1
    elif method == 'hypersolver':
        pred_fn = hypersolver((dynamics_wrap, midpoint_wrap), time_step, order=order, n_step=n_step)
        nb_params = 2
    elif method == 'hypersolver_grid':
        pred_fn = hypersolver((dynamics_wrap, ), time_step, order=order, n_step=n_step)
        nb_params = 1
    else:
        raise NotImplementedError('Method {} is not implemented yet'.format(method))
    ####################################################################################

    def augment_state(x):
        """ Augment a state with logp and initial time
        """
        return jnp.concatenate((x, jnp.zeros_like(x[...,:2]) ), axis=-1)

    # Define the forward function -> No extra operations on this dynamical system
    def forward_gen(key, params, x, predictor, nb_params):
        """ Compute the forward function """
        (_dynamics_params, _midpoint_params) = params
        # Get epsilon value
        eps = get_epsilon(key, x.shape)
        # Augment the state with time component and solve the ODE
        m_params = (params[0], eps) if nb_params == 1 else (params[0], eps, *params[1:])
        (out_ode, nfe), extra = predictor(augment_state(x), *m_params)
        return (out_ode[...,:-2],out_ode[...,:-2:-1], nfe), extra

    forward = jax.jit(lambda key, params, x : forward_gen(key, params, x, pred_fn, nb_params))
    odeint_forward = jax.jit(lambda key, params, x : forward_gen(key, params, x, odeint_eval, 1))

    def pre_odeint_forward(key, params, x):
        """ Compute the forward function and integrate it"""
        # Augment the state with time component
        eps = get_epsilon(key, x.shape)
        augm_x = augment_state(x)
        (out_ode, nfe), extra = odeint_eval(augm_x, params, eps)
        return augm_x, out_ode, eps

    def corrector_loss(params, xstate, eps, xnextstate):
        (_dynamics_params, _midpoint_params) = params
        (_xnstate, _), _ = pred_fn(xstate, _dynamics_params, eps, _midpoint_params)
        return _pred_loss(_xnstate, xnextstate)


    # Define the forward loss
    def forward_loss(key, params, x):
        """ Compute the loss function from the forward prediction """
        (z, logpz, nfe), extra = forward(key, params, x)
        sloss = _loss_fn(z, logpz)
        if extra is not None and pen_remainder > 0:
            sloss += pen_remainder * _rem_loss(extra)
        return sloss

    return pred_params, forward, forward_loss, odeint_forward, (pre_odeint_forward, corrector_loss)

# Define a function to compute relative error
@jax.jit
def _rel_error(eststate, truestate):
    """ Compute the relative error between two quantities
    """
    eststate = jnp.concatenate(eststate, axis=-1)
    truestate = jnp.concatenate(truestate, axis=-1)
    return jnp.sum(jnp.abs(eststate - truestate)) / jnp.sum(jnp.abs(eststate) + jnp.abs(truestate))

def init_data():
    """
    Initialize data.
    """
    data = datasets.MINIBOONE()

    num_train = data.trn.N
    # num_test = data.trn.N
    num_test = data.val.N

    if float_64:
        convert = jnp.float64
    else:
        convert = jnp.float32

    data.trn.x = convert(data.trn.x)
    data.val.x = convert(data.val.x)
    data.tst.x = convert(data.tst.x)

    num_batches = num_train // args.train_batch_size + 1 * (num_train % args.train_batch_size != 0)
    num_test_batches = num_test // args.test_batch_size + 1 * (num_train % args.test_batch_size != 0)

    # make sure we always save the model on the last iteration
    assert num_batches * args.nepochs % args.save_freq == 0

    def gen_train_data():
        """
        Generator for train data.
        """
        key = rng
        inds = jnp.arange(num_train)

        while True:
            key, = jax.random.split(key, num=1)
            epoch_inds = jax.random.shuffle(key, inds)
            for i in range(num_batches):
                batch_inds = epoch_inds[i * args.train_batch_size: min((i + 1) * args.train_batch_size, num_train)]
                yield data.trn.x[batch_inds]

    def gen_val_data():
        """
        Generator for train data.
        """
        inds = jnp.arange(num_test)
        while True:
            for i in range(num_test_batches):
                batch_inds = inds[i * args.test_batch_size: min((i + 1) * args.test_batch_size, num_test)]
                yield data.val.x[batch_inds]

    def gen_test_data():
        """
        Generator for train data.
        """
        inds = jnp.arange(num_test)
        while True:
            for i in range(num_test_batches):
                batch_inds = inds[i * args.test_batch_size: min((i + 1) * args.test_batch_size, num_test)]
                yield data.tst.x[batch_inds]

    ds_train = gen_train_data()
    ds_test = gen_val_data()

    meta = {
        "dims": data.n_dims,
        "num_batches": num_batches,
        "num_test_batches": num_test_batches
    }

    return ds_train, ds_test, meta


def evaluate_loss(m_params, forward_fn, _key, data_eval, num_iter, fwd_odeint=None):
    """ Compute the metrics for evaluation accross the data set
        :param m_params    : The parameters of the ode and the midpoints (if computed)
        :param forward_fn  : A function that computes the rollout trajectory
        :param data_eval   : The dataset considered for metric computation (iterable)
        :param num_iter    : Number of iterations over each chunck in the dataset
        :param fwd_odeint  : A function to compute the solution via adaptive time step
    """
    # Store metrics using forward_fn
    loss_values, pred_time, nfe_val, lossrem, bs= [np.zeros(num_iter) for i in range(5)]

    # Store metrics using the adaptive time step solver
    ode_loss_values, ode_pred_time, ode_nfe_val, ode_normdiff = [np.zeros(num_iter) for i in range(4)]

    for n_i in tqdm(range(num_iter),leave=False):
        # Extract the current data
        _key, = jax.random.split(_key, num=1)
        test_batch = next(data_eval)

        # Infer the next state values of the system
        curr_time = time.time()
        (z,  delta_logp, nfe) , extra = forward_fn(_key, m_params, test_batch)
        z.block_until_ready()
        diff_time  = time.time() - curr_time

        # Compute the loss function and the remainder loss (norm)
        lossval = _loss_fn(z, delta_logp)
        loss_rem = -1 if extra is None else _rem_loss(extra)

        # Save the data for logging
        pred_time[n_i]=diff_time; loss_values[n_i]=lossval; nfe_val[n_i]=nfe; lossrem[n_i]=loss_rem; bs[n_i] = len(test_batch)

        # If comparisons with odeint are requested
        if fwd_odeint is not None:
            # Infer the next state using the adaptive time step solver
            curr_time = time.time()
            (out_ode_z, dlogp, nfe_odeint) , _ = fwd_odeint(_key, m_params, test_batch)
            out_ode_z.block_until_ready()
            diff_time_odeint  = time.time() - curr_time

            # COmpute the loss function from the adaptive solver solution
            lossval_odeint = _loss_fn(out_ode_z, dlogp)

            # Compare the integration by the adaptive time step and our approach
            diff_predx = _rel_error((out_ode_z,dlogp), (z, delta_logp) )

            # Save the results
            ode_loss_values[n_i]=lossval_odeint; ode_nfe_val[n_i]=nfe_odeint; ode_pred_time[n_i]=diff_time_odeint; ode_normdiff[n_i]=diff_predx

    # Return the solution depending on if the adaptive solver is given or not 
    if fwd_odeint is None:
        return (np.average(loss_values, weights=bs), np.average(pred_time, weights=bs), np.average(nfe_val, weights=bs), np.average(lossrem,weights=bs)),(-1,-1,-1,-1,-1)
    else:
        m_eval = (np.average(loss_values, weights=bs), np.average(pred_time, weights=bs), np.average(nfe_val, weights=bs), np.average(lossrem, weights=bs))
        ode_eval = (np.average(ode_loss_values, weights=bs), np.average(ode_pred_time, weights=bs), np.average(ode_nfe_val, weights=bs), np.average(ode_normdiff, weights=bs))
        return m_eval, ode_eval

if __name__ == "__main__":
    # Parse the command line argument
    # Parse the command line argument
    parser = argparse.ArgumentParser('Learning MNIST classification')
    parser.add_argument('--method',  type=str, default='tayla')

    parser.add_argument('--nepochs', type=int, default=500)
    parser.add_argument('--train_batch_size', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--train_num_batch_eval', type=int, default=-1)

    parser.add_argument('--order', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=2)

    parser.add_argument('--trajdir',  type=str, default='data/')
    parser.add_argument('--no_compare_odeint',  action="store_true")

    parser.add_argument('--pen_remainder', type=float, default=0)

    parser.add_argument('--lr_init', type=float, default=1e-2)
    parser.add_argument('--lr_end', type=float, default=1e-3)
    parser.add_argument('--w_decay', type=float, default=0)
    parser.add_argument('--grad_clip', type=float, default=0)

    parser.add_argument('--mid_freq_update', type=int, default=-1)
    parser.add_argument('--mid_num_grad_iter', type=int, default=1)
    parser.add_argument('--mid_lr_init', type=float, default=1e-4)
    parser.add_argument('--mid_lr_end', type=float, default=1e-7)
    parser.add_argument('--mid_w_decay', type=float, default=0)
    parser.add_argument('--mid_grad_clip', type=float, default=0)

    parser.add_argument('--test_freq', type=int, default=300)
    parser.add_argument('--save_freq', type=int, default=3000)

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--atol', type=float, default=1.4e-8)
    parser.add_argument('--rtol', type=float, default=1.4e-8)

    parser.add_argument('--no_validation_set', action="store_true")

    parser.add_argument('--dur_ending_sched', type=int, default=0)
    parser.add_argument('--ending_lr_init', type=float, default=1e-4)
    parser.add_argument('--ending_lr_end', type=float, default=1e-4)

    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hdim_factor', type=int, default=20)
    parser.add_argument('--nonlinearity', type=str, default="softplus")

    args = parser.parse_args()

    softplus = lambda x: _logaddexp(x, jnp.zeros_like(x))

    nonlinearity = softplus if args.nonlinearity == "softplus" else jnp.tanh

    # Generate the random key generator -> A workaround for a weird jax + tensorflow core dumped error if it is placed after init_data
    rng = jax.random.PRNGKey(args.seed)

    ds_train, ds_test_eval, meta = init_data()
    num_train_batches_eval = meta['num_batches'] if args.train_num_batch_eval <= 0 else args.train_num_batch_eval
    num_train_batches = meta["num_batches"]
    num_test_batches = meta["num_test_batches"]

    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size

    # Some printing
    print('Meta information learning : \n', meta)
    print(vars(args))

    ############### Forward model and Loss functions ##################
    # Compute the function for forward and forward loss and corrections
    pred_params, forward, forward_loss, odeint_eval, (pre_odeint_forward, corrector_loss) = \
            init_model(rng, 43, args.order, args.n_steps, args.method, ts = 1.0, 
                        batch_size=args.train_batch_size, pen_remainder= args.pen_remainder, 
                        atol=args.atol, rtol=args.rtol) # State space dimension is 43

    ##################### Build the optimizer for ode weights #########################
    # Customize the gradient descent algorithm
    chain_list = [optax.scale_by_adam()] # optax.scale_by_adam(b1=0.999,b2=0.9999)

    # Add weight decay if enable
    decay_weight = args.w_decay
    if decay_weight > 0.0:
        chain_list.append(optax.add_decayed_weights(decay_weight))

    # Add scheduler for learning rate decrease --> Linear decrease given bt learning_rate_init and learning_rate_end
    # m_schedule = optax.piecewise_constant_schedule(-args.lr_init, {meta['num_train_batches']*2500 : 1e-1})
    # m_schedule = optax.linear_schedule(-args.lr_init, -args.lr_end, args.nepochs*meta['num_train_batches'])
    # m_schedule = optax.cosine_decay_schedule(-args.lr_init, args.nepochs*meta['num_train_batches'])
    assert args.lr_init >= args.lr_end and args.ending_lr_init >= args.ending_lr_end, 'Ending learning rate should be greater than starting learning rate'

    nepochs_sched1 = args.nepochs - args.dur_ending_sched
    m_schedule_start = optax.exponential_decay(-args.lr_init, nepochs_sched1*num_train_batches, args.lr_end / args.lr_init)
    m_schedule_end = optax.exponential_decay(-args.ending_lr_init, args.dur_ending_sched * num_train_batches, args.ending_lr_init/args.ending_lr_end)
    
    # Merge the two schedulers
    m_schedule = optax.join_schedules((m_schedule_start,m_schedule_end), [nepochs_sched1*num_train_batches])
    # print([float(m_schedule(s)) for s in range(args.nepochs*meta['num_train_batches']) ])
    # exit()

    chain_list.append(optax.scale_by_schedule(m_schedule))

    # Add gradient clipping if enable
    if args.grad_clip > 0.0:
        chain_list.append(optax.adaptive_grad_clip(clipping=args.grad_clip))

    # Check if correction is enable--> Only valid for Tayla
    no_correction = args.mid_freq_update < 0 or args.method != 'tayla'

    # Build the optimizer
    opt = optax.chain(*chain_list)
    opt_state = opt.init(pred_params if no_correction else pred_params[0] ) # We don't update the midpoint when correction enabled

    ##################### Build the optimizer for midpoint/residual weights #########################
    # Customize the gradient descent algorithm for the network parameters
    chain_list_res = [optax.scale_by_adam(b1=0.999,b2=0.9999)]

    # Add weight decay if enable
    decay_weight_res = args.mid_w_decay
    if decay_weight_res > 0.0:
        chain_list_res.append(optax.add_decayed_weights(decay_weight_res))

    # Add scheduler for learning rate decrease --> Linear decrease given bt learning_rate_init and learning_rate_end
    # m_schedule_res = optax.linear_schedule(-args.mid_lr_init, -args.mid_lr_end, int( args.mid_num_grad_iter*((args.nepochs*meta['num_train_batches']) / args.mid_freq_update)) )
    m_schedule_res = optax.exponential_decay(-args.mid_lr_init, int( args.mid_lr_init*((args.nepochs*num_train_batches) / args.mid_freq_update)), args.mid_lr_end / args.mid_lr_init)
    # m_schedule_res = optax.linear_schedule(-args.mid_lr_init, -args.mid_lr_end, int( n_iter*meta['num_train_batches']) )
    chain_list_res.append(optax.scale_by_schedule(m_schedule_res))
    # Add gradient clipping if enable
    if args.mid_grad_clip > 0.0:
        chain_list_res.append(optax.adaptive_grad_clip(clipping=args.mid_grad_clip))
    # Build the optimizer
    mid_opt = optax.chain(*chain_list_res)
    mid_opt_state = mid_opt.init(pred_params[1]) if len(pred_params) == 2 else None # No params in case of noresidual-based solvers

    ######################## Update function ##########################

    @jax.jit
    def update(key, params, _opt_state, x):
        """ Define the update rule for the parameters of the ODE
            :param params         : A tuple containing parameters of ODE and midpoint
            :param _opt_state     : The current state of the optimizer
            :param xstate         : A bactch of trajectories of the system
            :param xstatenext     : A bacth of next state value of the system
        """
        if not no_correction:
            params_dyn = params[0]
            dyn_loss = lambda dyn_params : forward_loss(key, (dyn_params, params[1]), x)
        else:
            params_dyn = params
            dyn_loss = lambda dyn_params : forward_loss(key, dyn_params, x)
        grads = jax.grad(dyn_loss, has_aux=False)(params_dyn)
        updates, _opt_state = opt.update(grads, _opt_state, params_dyn)
        params_dyn = optax.apply_updates(params_dyn, updates)
        if not no_correction:
            return (params_dyn, params[1]), _opt_state
        else:
            return params_dyn, _opt_state

    @jax.jit
    def mid_update(key, params, _opt_state, x):
        """ Update rule for the midpoint parameters
            :param params         : A tuple containing parameters of ODE and midpoint
            :param _opt_state     : The current state of the optimizer
            :param xstate         : A bactch of trajectories of the system
        """
        mid_params = params[1] # Assume params contain the midpoint parameters
        augm_x, xnext, eps = pre_odeint_forward(key, params[0], x)

        # Define the loss function
        residual_loss = lambda params_mid : corrector_loss((params[0],params_mid), augm_x, eps, xnext)

        # Define the gradient function
        grad_fun = jax.grad(residual_loss, has_aux=False)
        def multi_iter(pcarry, extra):
            """ Perform several gradient step"""
            m_params, m_opt_state = pcarry
            grads = grad_fun(m_params)
            updates, m_opt_state = mid_opt.update(grads, m_opt_state, m_params)
            m_params = optax.apply_updates(m_params, updates)
            return (m_params, m_opt_state), None
        # Perform several gradient steps
        (mid_params, _opt_state), _ = lax.scan(multi_iter, (mid_params, _opt_state), xs=None, length=args.mid_num_grad_iter)
        return (params[0], mid_params), _opt_state

    ######################## Main training loop ##########################
    # Save the number of iteration
    itr_count = 0
    itr_count_corr = 0

    # Save the optimal loss_value obtained
    opt_params_dict, opt_loss_test, opt_rem_test, \
        opt_nfe, opt_diff, opt_loss_odeint, opt_predtime_test, opt_predtime_test_odeint = [None] * 8

    # Save the loss evolution and other useful quantities
    total_time, compute_time_update, update_time_average = 0, list(), 0.0
    loss_evol_test, predtime_evol_test, \
        constr_rem_evol_test, loss_evol_odeint, nfe_evol_odeint, \
        err_evol_odeint, predtime_evol_odeint = [list() for i in range(7)]

    # Save all the command line arguments of this script
    m_parameters_dict = vars(args)
    out_data_file = args.trajdir +'dyn_ffjortab_{}_o{}_s{}'.format(args.method, args.order, args.n_steps)

    # Open the info file to save the command line print
    outfile = open(out_data_file+'_info.txt', 'w')
    outfile.write('Training parameters: \n{}'.format(m_parameters_dict))
    outfile.write('////// Command line messages \n\n')
    outfile.close()

    key = rng

    # Start the iteration loop
    for epoch in tqdm(range(args.nepochs)):
        # Iterate on the total number of batches
        for i in tqdm(range(num_train_batches), leave=False):
            key, = jax.random.split(key, num=1)
            batch = next(ds_train)

            # Increment the iteration count
            itr_count += 1

            # Update the weight of the neural network representing the ODE
            update_start = time.time()
            pred_params, opt_state = update(key, pred_params, opt_state, batch)
            tree_flatten(opt_state)[0][0].block_until_ready()
            update_end = time.time() - update_start

            # In case there is an update rule for the midpoint -> Do as following
            if (not no_correction) and mid_opt_state is not None and itr_count % args.mid_freq_update == 0:
                itr_count_corr += 1
                update_start = time.time()
                pred_params, mid_opt_state = mid_update(key, pred_params, mid_opt_state, batch)
                tree_flatten(mid_opt_state)[0][0].block_until_ready()
                corr_time = time.time() - update_start
                if itr_count_corr >= 5:
                    update_end += corr_time

            # Total elapsed compute time for update only
            if itr_count >= 5: # Remove the first few steps due to jit compilation
                update_time_average = (itr_count * update_time_average + update_end) / (itr_count + 1)
                compute_time_update.append(update_end)
                total_time += update_end
            else:
                update_time_average = update_end

             # Check if it is time to compute the metrics for evaluation
            if itr_count % args.test_freq == 0:
                # Print the logging information
                print_str_test = '----------------------------- Eval on Test Data [epoch={} | num_batch = {}] -----------------------------\n'.format(epoch, i)
                tqdm.write(print_str_test)

                # Compute the loss on the entire testing set 
                (loss_values_test, pred_time_test, nfe_test, contr_rem_test), (ode_ltest, ode_predtime, ode_nfe, ode_errdiff) = \
                        evaluate_loss(pred_params, forward,  key, ds_test_eval, meta['num_test_batches'], fwd_odeint=None if args.no_compare_odeint else odeint_eval)

                # First time we have a value for the loss function
                if opt_loss_test is None or (opt_loss_test > loss_values_test):
                    opt_params_dict, opt_loss_test, opt_loss_odeint, opt_rem_test, opt_nfe, opt_diff,\
                        opt_predtime_test, opt_predtime_test_odeint = \
                        pred_params, loss_values_test, ode_ltest, contr_rem_test, ode_nfe, ode_errdiff,\
                            pred_time_test, ode_predtime

                # Do some printing for result visualization
                print_str = 'Iter {:05d} | Total Update Time {:.2e} | Update time {:.2e}\n'.format(itr_count, total_time, update_end)
                print_str += '[    Test      ] Loss = {:.2e} | Pred. Time. = {:.2e} | NFE = {:.3e} |  Rem. Val.   = {:.2e}\n'.format(loss_values_test, pred_time_test, nfe_test,contr_rem_test)
                print_str += '[  ODEINT Test ] Loss = {:.2e} | Pred. Time. = {:.2e} | NFE = {:.3e} | Diff. Pred.  = {:.2e}\n'.format(ode_ltest, ode_predtime, ode_nfe, ode_errdiff)
                print_str += '[  OPT. Value. ] Loss Test = {:.2e} | Loss ODE   = {:.2e} | NFE ODEINT = {:.3e} | Diff. Pred.  = {:.2e} | Pred. Time = {:.2e} | Pred. Time. ODEINT = {:.2e}\n'.format(
                                opt_loss_test, opt_loss_odeint, opt_nfe, opt_diff, opt_predtime_test, opt_predtime_test_odeint)

                tqdm.write(print_str)
                # tqdm.write('{}'.format(pred_params[1]))

                # Save all the obtained data
                loss_evol_test.append(loss_values_test); loss_evol_odeint.append(ode_ltest)
                predtime_evol_test.append(pred_time_test); predtime_evol_odeint.append(ode_predtime)
                nfe_evol_odeint.append(ode_nfe); err_evol_odeint.append(ode_errdiff)
                constr_rem_evol_test.append(contr_rem_test)


                # Save these info of the console in a text file
                outfile = open(out_data_file+'_info.txt', 'a')
                outfile.write(print_str_test)
                outfile.write(print_str)
                outfile.close()

            if itr_count % args.save_freq == 0 or (epoch == args.nepochs-1 and i == num_train_batches-1):
                m_dict_res = {'best_params' : opt_params_dict, 'total_time' : total_time, 'updatetime_evol' : compute_time_update,
                                'opt_loss_test' : opt_loss_test, 'opt_loss_odeint_test' : opt_loss_odeint,
                                'opt_nfe_test' : opt_nfe,  'opt_diff_test' : opt_diff,
                                'opt_predtime_test' : opt_predtime_test, 'opt_predtime_test_odeint' : opt_predtime_test_odeint,
                                'opt_rem_test' : opt_rem_test, 
                                'loss_evol_test' : loss_evol_test, 
                                'predtime_evol_test' : predtime_evol_test, 'predtime_evol_odeint_test' : predtime_evol_odeint,
                                'loss_evol_odeint' : loss_evol_odeint, 'err_evol_odeint' : err_evol_odeint,
                                'nfe_evol_odeint' : nfe_evol_odeint, 'constr_rem_evol_test' : constr_rem_evol_test, 
                                'training_parameters' : m_parameters_dict}
                outfile = open(out_data_file+'_res.pkl', "wb")
                pickle.dump(m_dict_res, outfile)
                outfile.close()