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

# Haiku for Neural networks
import haiku as hk

# Optax for the optimization scheme
import optax

from tqdm.auto import tqdm

# Import from our module algorithms
from taylanets.tayla import hypersolver, odeint_rev, tayla, wrap_module

# Import the datasets to be used
import tensorflow_datasets as tfds

from math import prod

def softmax_cross_entropy(logits, labels):
    """
    Cross-entropy loss applied to softmax.
    """
    one_hot = hk.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


class PreODE(hk.Module):
    """
    Module applied before the ODE layer.
    """
    def __init__(self):
        super(PreODE, self).__init__()
        # Should consider the dtype automatically
        self.model = hk.Sequential([lambda x : (x / 255.0), hk.Flatten()])

    def __call__(self, x):
        return self.model(x)


class MLPDynamics(hk.Module):
    """
    Dynamics for ODE as an MLP.
    """
    def __init__(self, dim):
        """ Build a 2-layer NN where dim specifies the flatten dimension of the input image
        """
        super(MLPDynamics, self).__init__()
        self.dim = dim
        self.hidden_dim = 100
        self.lin1 = hk.Linear(self.hidden_dim)
        self.lin2 = hk.Linear(self.dim)

    def __call__(self, x, t):
        out = jax.nn.sigmoid(x)
        tt = jnp.ones_like(x[..., :1]) * t
        t_out = jnp.concatenate([tt, out], axis=-1)
        out = self.lin1(t_out)

        out = jax.nn.sigmoid(out)
        tt = jnp.ones_like(out[..., :1]) * t
        t_out = jnp.concatenate([tt, out], axis=-1)
        out = self.lin2(t_out)

        return out

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


class PostODE(hk.Module):
    """
    Module applied after the ODE layer.
    """
    def __init__(self):
        super(PostODE, self).__init__()
        self.model = hk.Sequential([
            jax.nn.sigmoid,
            hk.Linear(10)
        ])

    def __call__(self, x):
        return self.model(x)


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


@jax.jit
def _acc_fn(logits, labels):
    """
    Classification accuracy of the model.
    """
    predicted_class = jnp.argmax(logits, axis=1)
    return jnp.mean(predicted_class == labels)


@jax.jit
def _loss_fn(logits, labels):
    """ 
    Compute the cross entropy loss given the logits and labels
    """
    return jnp.mean(softmax_cross_entropy(logits, labels))


# Define a function to compute relative error
@jax.jit
def _rel_error(eststate, truestate):
    """ Compute the relative error between two quantities
    """
    return jnp.sum(jnp.abs(eststate - truestate)) / jnp.sum(jnp.abs(eststate) + jnp.abs(truestate))


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


# function assumes ts = [0, 1]
def init_model(rng, order, n_step, method='tayla', batch_size=1, 
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
    image_shape = (28, 28, 1)
    ode_shape = prod(image_shape)
    midpoint_shape = ode_shape + 1
    pre_ode_init = jnp.zeros((batch_size, *image_shape))
    ode_init, t_init = jnp.zeros((batch_size, ode_shape)), 0.0
    midpoint_init = jnp.zeros((batch_size, midpoint_shape))
    post_ode_init = jnp.zeros((batch_size, ode_shape))

    # Build the pre module
    pre_ode = hk.without_apply_rng(hk.transform(wrap_module(PreODE)))
    pre_ode_params = pre_ode.init(rng, pre_ode_init)
    pre_ode_fn = pre_ode.apply

    # Build the ODE function module
    dynamics = hk.without_apply_rng(hk.transform(wrap_module(MLPDynamics, ode_shape)))
    dynamics_params = dynamics.init(rng, ode_init, t_init)
    dynamics_wrap_temp = dynamics.apply
    def dynamics_wrap(_images, params):
        """ Images last component is the time index"""
        derstate = dynamics_wrap_temp(params, _images[...,:-1], _images[...,-1:])
        derstate_time = jnp.ones_like(_images[...,:1])
        return jnp.concatenate((derstate, derstate_time), axis=-1)

    # Build the Midpoint function module
    midpoint = hk.without_apply_rng(hk.transform(wrap_module(Midpoint, midpoint_shape, time_step)))
    midpoint_params = midpoint.init(rng, midpoint_init)
    midpoint_wrap = lambda x, _params : midpoint.apply(_params, x)
    # print(midpoint_wrap(midpoint_init, midpoint_params))

    # Post processing function module
    post_ode = hk.without_apply_rng(hk.transform(wrap_module(PostODE)))
    post_ode_params = post_ode.init(rng, post_ode_init)
    post_ode_fn = post_ode.apply

    pred_params = (dynamics_params, midpoint_params, pre_ode_params, post_ode_params)

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

    def augment_state(_images):
        return jnp.concatenate((_images, jnp.zeros_like(_images[...,:1])), axis=-1)

    # Define the forward function -> No extra operations on this dynamical system
    def forward_gen(params, _images, predictor, nb_params):
        """ Compute the forward function """
        (_dynamics_params, _midpoint_params, _pre_ode_params, _post_ode_params) = params
        # Pre-processing steps
        out_pre_ode = pre_ode_fn(_pre_ode_params, _images)
        # Augment the state with time component
        # Solve the ode
        (out_ode, nfe), extra = predictor(augment_state(out_pre_ode), *params[:nb_params])
        # Post-processing steps
        out_post_ode = post_ode_fn(_post_ode_params, out_ode[...,:-1])
        return (out_post_ode, nfe), extra

    forward = jax.jit(lambda params, _images : forward_gen(params, _images, pred_fn, nb_params))
    odeint_forward = jax.jit(lambda params, _images : forward_gen(params, _images, odeint_eval, 1))

    def pre_odeint_forward(params, _images):
        """ Compute the forward function and integrate it"""
        (_dynamics_params, _pre_ode_params) = params
        # Pre-processing steps
        out_pre_ode = pre_ode_fn(_pre_ode_params, _images)
        out_pre_ode = augment_state(out_pre_ode)
        # Augment the state with time component
        (out_ode, nfe), extra = odeint_eval(out_pre_ode, _dynamics_params)
        return out_pre_ode, out_ode

    def corrector_loss(params, xstate, xnextstate):
        (_dynamics_params, _midpoint_params) = params
        (_xnstate, _), _ = pred_fn(xstate, _dynamics_params, _midpoint_params)
        return _pred_loss(_xnstate, xnextstate)


    # Define the forward loss
    def forward_loss(params, _images, _labels):
        """ Compute the loss function from the forward prediction """
        (logits, nfe), extra = forward(params, _images)
        sloss = _loss_fn(logits, _labels)
        if extra is not None and pen_remainder > 0:
            sloss += pen_remainder * _rem_loss(extra)
        return sloss

    return pred_params, forward, forward_loss, odeint_forward, post_ode_init.dtype, (pre_odeint_forward, corrector_loss)



### Custom function for uploading the data set
def init_data(train_batch_size, test_batch_size, seed_number=0, shuffle=1000, validation_set=True):
    """
    Initialize data from tensorflow dataset
    """
    # Import and cache the file in the current directory
    ds_data, ds_info = tfds.load('mnist',
                                     data_dir='data/tensorflow_data/',
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

    ds_train, ds_test = ds_train.batch(train_batch_size), ds_test.batch(test_batch_size).repeat()
    ds_train, ds_test = tfds.as_numpy(ds_train), tfds.as_numpy(ds_test)

    meta = {
        "num_train_batches": num_train_batches,
        "num_test_batches": num_test_batches,
        "num_train" : num_train,
        "num_test" : num_test
    }

    # Return iter element on the training and testing set
    return iter(ds_train), iter(ds_test), meta


@jax.jit
def sep_losses(out_ode, _labels):
    """ Convenience function for calculation the losses
    """
    return _loss_fn(out_ode, _labels), _acc_fn(out_ode, _labels)


def evaluate_loss(m_params, forward_fn, data_eval, num_iter, fwd_odeint=None):
    """ Compute the metrics for evaluation accross the data set
        :param m_params    : The parameters of the ode and the midpoints (if computed)
        :param forward_fn  : A function that computes the rollout trajectory
        :param data_eval   : The dataset considered for metric computation (iterable)
        :param num_iter    : Number of iterations over each chunck in the dataset
        :param fwd_odeint  : A function to compute the solution via adaptive time step
    """
    # Store metrics using forward_fn
    loss_values, pred_time, nfe_val, lossrem, acc_values = [np.zeros(num_iter) for i in range(5)]

    # Store metrics using the adaptive time step solver
    ode_loss_values, ode_pred_time, ode_nfe_val, ode_normdiff, ode_accvalues = [np.zeros(num_iter) for i in range(5)]

    for n_i in tqdm(range(num_iter),leave=False):
        # Extract the current data
        _images, _labels = next(data_eval)
        _images = _images.astype(m_dtype)

        # Infer the next state values of the system
        curr_time = time.time()
        (_logits, nfe) , extra = forward_fn(m_params, _images)
        _logits.block_until_ready()
        diff_time  = time.time() - curr_time

        # Compute the loss function and the remainder loss (norm)
        lossval, accval = sep_losses(_logits, _labels)
        loss_rem = -1 if extra is None else _rem_loss(extra)

        # Save the data for logging
        pred_time[n_i]=diff_time; loss_values[n_i]=lossval; nfe_val[n_i]=nfe; lossrem[n_i]=loss_rem; acc_values[n_i] = accval

        # If comparisons with odeint are requested
        if fwd_odeint is not None:
            # Infer the next state using the adaptive time step solver
            curr_time = time.time()
            (logits, nfe_odeint) , _ = fwd_odeint(m_params, _images)
            logits.block_until_ready()
            diff_time_odeint  = time.time() - curr_time

            # COmpute the loss function from the adaptive solver solution
            lossval_odeint, accval_odeint = sep_losses(logits, _labels)

            # Compare the integration by the adaptive time step and our approach
            diff_predx = _rel_error(logits, _logits)

            # Save the results
            ode_loss_values[n_i]=lossval_odeint; ode_nfe_val[n_i]=nfe_odeint; ode_pred_time[n_i]=diff_time_odeint; ode_normdiff[n_i]=diff_predx; ode_accvalues[n_i] = accval_odeint

    # Return the solution depending on if the adaptive solver is given or not 
    if fwd_odeint is None:
        return (np.mean(loss_values), np.mean(pred_time), np.mean(nfe_val), np.mean(lossrem), np.mean(acc_values)),(-1,-1,-1,-1,-1)
    else:
        m_eval = (np.mean(loss_values), np.mean(pred_time), np.mean(nfe_val), np.mean(lossrem), np.mean(acc_values))
        ode_eval = (np.mean(ode_loss_values), np.mean(ode_pred_time), np.mean(ode_nfe_val), np.mean(ode_normdiff), np.mean(ode_accvalues))
        return m_eval, ode_eval

if __name__ == "__main__":
    # Parse the command line argument
    # Parse the command line argument
    parser = argparse.ArgumentParser('Learning MNIST classification')
    parser.add_argument('--method',  type=str, default='tayla')

    parser.add_argument('--nepochs', type=int, default=5000)
    parser.add_argument('--train_batch_size', type=int, default=500)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--train_num_batch_eval', type=int, default=-1)

    parser.add_argument('--order', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=1)

    parser.add_argument('--trajdir',  type=str, default='data/')
    parser.add_argument('--no_compare_odeint',  action="store_true")

    parser.add_argument('--pen_remainder', type=float, default=0)

    parser.add_argument('--lr_init', type=float, default=1e-4)
    parser.add_argument('--lr_end', type=float, default=1e-6)
    parser.add_argument('--w_decay', type=float, default=0)
    parser.add_argument('--grad_clip', type=float, default=0)

    parser.add_argument('--mid_freq_update', type=int, default=-1)
    parser.add_argument('--mid_num_grad_iter', type=int, default=1)
    parser.add_argument('--mid_lr_init', type=float, default=1e-4)
    parser.add_argument('--mid_lr_end', type=float, default=1e-7)
    parser.add_argument('--mid_w_decay', type=float, default=0)
    parser.add_argument('--mid_grad_clip', type=float, default=0)

    parser.add_argument('--test_freq', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=10000)

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--atol', type=float, default=1.4e-5)
    parser.add_argument('--rtol', type=float, default=1.4e-5)

    parser.add_argument('--no_validation_set', action="store_true")

    parser.add_argument('--dur_ending_sched', type=int, default=0)
    parser.add_argument('--ending_lr_init', type=float, default=1e-4)
    parser.add_argument('--ending_lr_end', type=float, default=1e-4)

    args = parser.parse_args()

    # Generate the random key generator -> A workaround for a weird jax + tensorflow core dumped error if it is placed after init_data
    rng = jax.random.PRNGKey(args.seed)

    # Initialize the MNIST data set
    ds_train, ds_train_eval, meta = init_data(args.train_batch_size, args.test_batch_size, seed_number=args.seed, validation_set=not args.no_validation_set)
    num_train_batches_eval = meta['num_train_batches'] if args.train_num_batch_eval <= 0 else args.train_num_batch_eval

    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size

    # Some printing
    print('Meta information learning : \n', meta)
    print(vars(args))

    ############### Forward model and Loss functions ##################
    # Compute the function for forward and forward loss and corrections
    pred_params, forward, forward_loss, odeint_eval, m_dtype, (pre_odeint_forward, corrector_loss) = \
            init_model(rng, args.order, args.n_steps, args.method, ts = 1.0, 
                        batch_size=args.train_batch_size, pen_remainder= args.pen_remainder, 
                        atol=args.atol, rtol=args.rtol)

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
    m_schedule_start = optax.exponential_decay(-args.lr_init, nepochs_sched1*meta['num_train_batches'], args.lr_end / args.lr_init)
    m_schedule_end = optax.exponential_decay(-args.ending_lr_init, args.dur_ending_sched * meta['num_train_batches'], args.ending_lr_init/args.ending_lr_end)
    
    # Merge the two schedulers
    m_schedule = optax.join_schedules((m_schedule_start,m_schedule_end), [nepochs_sched1*meta['num_train_batches']])
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
    opt_state = opt.init(pred_params if no_correction else (pred_params[0], pred_params[2],pred_params[3]) ) # We don't update the midpoint when correction enabled

    ##################### Build the optimizer for midpoint/residual weights #########################
    # Customize the gradient descent algorithm for the network parameters
    chain_list_res = [optax.scale_by_adam(b1=0.999,b2=0.9999)]

    # Add weight decay if enable
    decay_weight_res = args.mid_w_decay
    if decay_weight_res > 0.0:
        chain_list_res.append(optax.add_decayed_weights(decay_weight_res))

    # Add scheduler for learning rate decrease --> Linear decrease given bt learning_rate_init and learning_rate_end
    # m_schedule_res = optax.linear_schedule(-args.mid_lr_init, -args.mid_lr_end, int( args.mid_num_grad_iter*((args.nepochs*meta['num_train_batches']) / args.mid_freq_update)) )
    m_schedule_res = optax.exponential_decay(-args.mid_lr_init, int( args.mid_lr_init*((args.nepochs*meta['num_train_batches']) / args.mid_freq_update)), args.mid_lr_end / args.mid_lr_init)
    # m_schedule_res = optax.linear_schedule(-args.mid_lr_init, -args.mid_lr_end, int( n_iter*meta['num_train_batches']) )
    chain_list_res.append(optax.scale_by_schedule(m_schedule_res))
    # Add gradient clipping if enable
    if args.mid_grad_clip > 0.0:
        chain_list_res.append(optax.adaptive_grad_clip(clipping=args.mid_grad_clip))
    # Build the optimizer
    mid_opt = optax.chain(*chain_list_res)
    mid_opt_state = mid_opt.init(pred_params[1]) if len(pred_params) == 4 else None # No params in case of noresidual-based solvers


    ######################## Update function ##########################

    @jax.jit
    def update(params, _opt_state, _images, _labels):
        """ Define the update rule for the parameters of the ODE
            :param params         : A tuple containing parameters of ODE and midpoint
            :param _opt_state     : The current state of the optimizer
            :param xstate         : A bactch of trajectories of the system
            :param xstatenext     : A bacth of next state value of the system
        """
        if not no_correction:
            params_dyn = (params[0],params[2],params[3])
            dyn_loss = lambda dyn_params : forward_loss((dyn_params[0], params[1], *dyn_params[1:]), _images, _labels)
        else:
            params_dyn = params
            dyn_loss = lambda dyn_params : forward_loss(dyn_params, _images, _labels)
        grads = jax.grad(dyn_loss, has_aux=False)(params_dyn)
        updates, _opt_state = opt.update(grads, _opt_state, params_dyn)
        params_dyn = optax.apply_updates(params_dyn, updates)
        if not no_correction:
            return (params_dyn[0], params[1], *params_dyn[1:]), _opt_state
        else:
            return params_dyn, _opt_state

    @jax.jit
    def mid_update(params, _opt_state, _images):
        """ Update rule for the midpoint parameters
            :param params         : A tuple containing parameters of ODE and midpoint
            :param _opt_state     : The current state of the optimizer
            :param xstate         : A bactch of trajectories of the system
        """
        mid_params = params[1] # Assume params contain the midpoint parameters
        pre_xstate, post_xstate = pre_odeint_forward((params[0], params[2]), _images)

        # Define the loss function
        residual_loss = lambda params_mid : corrector_loss((params[0],params_mid), pre_xstate, post_xstate)

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
        return (params[0], mid_params, *params[2:]), _opt_state

    ######################## Main training loop ##########################
    # Save the number of iteration
    itr_count = 0
    itr_count_corr = 0

    # Save the optimal loss_value obtained
    opt_params_dict, opt_loss_train, opt_accuracy_train, opt_loss_test, opt_accuracy_test, opt_rem_test, opt_rem_train, \
        opt_nfe, opt_diff, opt_loss_odeint, opt_accuracy_odeint_test, opt_predtime_test, opt_predtime_test_odeint = [None] * 13

    # Save the loss evolution and other useful quantities
    total_time, compute_time_update, update_time_average = 0, list(), 0.0
    loss_evol_train, train_accuracy, loss_evol_test, test_accuracy, test_accuracy_odeint, predtime_evol_train, predtime_evol_test, \
        constr_rem_evol_train, constr_rem_evol_test, loss_evol_odeint, nfe_evol_odeint, \
        err_evol_odeint, predtime_evol_odeint = [list() for i in range(13)]

    # Save all the command line arguments of this script
    m_parameters_dict = vars(args)
    out_data_file = args.trajdir +'dyn_mnist_{}_o{}_s{}'.format(args.method, args.order, args.n_steps)

    # Open the info file to save the command line print
    outfile = open(out_data_file+'_info.txt', 'w')
    outfile.write('Training parameters: \n{}'.format(m_parameters_dict))
    outfile.write('////// Command line messages \n\n')
    outfile.close()

    # Start the iteration loop
    for epoch in tqdm(range(args.nepochs)):
        # Iterate on the total number of batches
        for i in tqdm(range(meta['num_train_batches']), leave=False):
            # Get the next batch of images
            _images, _labels = next(ds_train)

            # Convert the image into float 
            _images = _images.astype(m_dtype)

            # Increment the iteration count
            itr_count += 1

            # Update the weight of the neural network representing the ODE
            update_start = time.time()
            pred_params, opt_state = update(pred_params, opt_state, _images, _labels)
            tree_flatten(opt_state)[0][0].block_until_ready()
            update_end = time.time() - update_start

            # In case there is an update rule for the midpoint -> Do as following
            if (not no_correction) and mid_opt_state is not None and itr_count % args.mid_freq_update == 0:
                itr_count_corr += 1
                update_start = time.time()
                pred_params, mid_opt_state = mid_update(pred_params, mid_opt_state, _images)
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

                # Compute the loss on the entire training set 
                (loss_values_train, pred_time_train, nfe_train, contr_rem_train, acc_train), _ = \
                        evaluate_loss(pred_params, forward, ds_train, num_train_batches_eval)

                # Compute the loss on the entire testing set 
                (loss_values_test, pred_time_test, nfe_test, contr_rem_test, acc_test), (ode_ltest, ode_predtime, ode_nfe, ode_errdiff, acc_test_odeint) = \
                        evaluate_loss(pred_params, forward,  ds_train_eval, meta['num_test_batches'], fwd_odeint=None if args.no_compare_odeint else odeint_eval)

                # First time we have a value for the loss function
                if opt_loss_train is None or opt_loss_test is None or (opt_accuracy_test < acc_test):
                    opt_params_dict, opt_loss_train, opt_loss_test, opt_loss_odeint, opt_rem_train, opt_rem_test, opt_nfe, opt_diff, opt_accuracy_train, opt_accuracy_test,\
                        opt_accuracy_odeint_test, opt_predtime_test, opt_predtime_test_odeint = \
                        pred_params, loss_values_train, loss_values_test, ode_ltest, contr_rem_train, contr_rem_test, ode_nfe, ode_errdiff,\
                            acc_train, acc_test, acc_test_odeint, pred_time_test, ode_predtime

                # Do some printing for result visualization
                print_str = 'Iter {:05d} | Total Update Time {:.2e} | Update time {:.2e}\n'.format(itr_count, total_time, update_end)
                print_str += '[    Train     ] Loss = {:.2e} | Accuracy    = {:.3e} | Pred. Time. = {:.2e} | NFE = {:.3e} |  Rem. Val.   = {:.2e}\n'.format(loss_values_train, acc_train, pred_time_train, nfe_train, contr_rem_train)
                print_str += '[    Test      ] Loss = {:.2e} | Accuracy    = {:.3e} | Pred. Time. = {:.2e} | NFE = {:.3e} |  Rem. Val.   = {:.2e}\n'.format(loss_values_test, acc_test, pred_time_test, nfe_test,contr_rem_test)
                print_str += '[  ODEINT Test ] Loss = {:.2e} | Accuracy    = {:.3e} | Pred. Time. = {:.2e} | NFE = {:.3e} | Diff. Pred.  = {:.2e}\n'.format(ode_ltest, acc_test_odeint, ode_predtime, ode_nfe, ode_errdiff)
                print_str += '[  OPT. Value. ] Loss Train = {:.2e} | Loss Test = {:.2e} | Acc Train = {:.3e} | Acc Test = {:.3e} | Acc Test ODEINT = {:.3e}\n'.format(
                                opt_loss_train, opt_loss_test, opt_accuracy_train, opt_accuracy_test, opt_accuracy_odeint_test)
                print_str += '                 Loss ODE   = {:.2e} | NFE ODEINT = {:.3e} | Diff. Pred.  = {:.2e} | Pred. Time = {:.2e} | Pred. Time. ODEINT = {:.2e}\n'.format(opt_loss_odeint, opt_nfe, opt_diff, opt_predtime_test, opt_predtime_test_odeint)

                tqdm.write(print_str)
                # tqdm.write('{}'.format(pred_params[1]))

                # Save all the obtained data
                loss_evol_train.append(loss_values_train); loss_evol_test.append(loss_values_test); loss_evol_odeint.append(ode_ltest)
                train_accuracy.append(acc_train); test_accuracy.append(acc_test); test_accuracy_odeint.append(acc_test_odeint)
                predtime_evol_train.append(pred_time_train); predtime_evol_test.append(pred_time_test); predtime_evol_odeint.append(ode_predtime)
                nfe_evol_odeint.append(ode_nfe); err_evol_odeint.append(ode_errdiff)
                constr_rem_evol_train.append(contr_rem_train); constr_rem_evol_test.append(contr_rem_test)


                # Save these info of the console in a text file
                outfile = open(out_data_file+'_info.txt', 'a')
                outfile.write(print_str_test)
                outfile.write(print_str)
                outfile.close()

            if itr_count % args.save_freq == 0 or (epoch == args.nepochs-1 and i == meta['num_train_batches']-1):
                m_dict_res = {'best_params' : opt_params_dict, 'total_time' : total_time, 'updatetime_evol' : compute_time_update,
                                'opt_loss_train' : opt_loss_train, 'opt_loss_test' : opt_loss_test, 'opt_loss_odeint_test' : opt_loss_odeint,
                                'opt_accuracy_test' : opt_accuracy_test, 'opt_accuracy_train' : opt_accuracy_train, 'opt_accuracy_odeint_test' : opt_accuracy_odeint_test,
                                'opt_nfe_test' : opt_nfe,  'opt_diff_test' : opt_diff,
                                'opt_predtime_test' : opt_predtime_test, 'opt_predtime_test_odeint' : opt_predtime_test_odeint,
                                'opt_rem_test' : opt_rem_test, 'opt_rem_train' : opt_rem_train,

                                'loss_evol_train' : loss_evol_train, 'loss_evol_test' : loss_evol_test, 
                                'accuracy_evol_train' : train_accuracy, 'accuracy_evol_test' : test_accuracy, 'accuracy_evol_odeint' : test_accuracy_odeint,

                                'predtime_evol_train' : predtime_evol_train, 'predtime_evol_test' : predtime_evol_test, 'predtime_evol_odeint_test' : predtime_evol_odeint,
                                'loss_evol_odeint' : loss_evol_odeint, 'err_evol_odeint' : err_evol_odeint,
                                'nfe_evol_odeint' : nfe_evol_odeint, 'constr_rem_evol_train' : constr_rem_evol_train, 'constr_rem_evol_test' : constr_rem_evol_test, 
                                'training_parameters' : m_parameters_dict}
                outfile = open(out_data_file+'_res.pkl', "wb")
                pickle.dump(m_dict_res, outfile)
                outfile.close()
