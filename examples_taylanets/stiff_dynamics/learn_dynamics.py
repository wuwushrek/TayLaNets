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

# This class represents the system's dynamics and need to change from one model to another one
class MLPDynamics(hk.Module):
    """Known dynamics of the system --> Assumed autonomous system (time is included in the state)
    """
    def __init__(self):
        super(MLPDynamics, self).__init__()
        outsize = MLPDynamics.nstate
        self.model = hk.Sequential([hk.Linear(64), hk.Linear(outsize)])

    def __call__(self, x):
        return self.model(x)

    # Number of state of the system
    nstate = 2

# This class represents the midpoint parameterization (for tayla) or the residual (for hypersolver)
class Midpoint(hk.Module):
    """Compute the coefficients in the formula obtained to simplify the learning of the midpoint
    """
    def __init__(self):
        """ Build a MLP approximating the coefficient in the midpoint formula
            :param dim         : Specifies the output dimension of this NN
        """
        super(Midpoint, self).__init__(name='midpoint_residual')
        outsize = MLPDynamics.nstate if args.method == 'hypersolver' else MLPDynamics.nstate
        # Initialize the weight to be randomly close to zero
        # The output size of the neural network must be ns or ns**2 (multiplicative vector or Matrix)
        # And in case, there is a time dependency it should be (ns+1) or (ns+1)**2
        self.model = hk.nets.MLP(output_sizes=(16, outsize), 
                        w_init=hk.initializers.RandomUniform(minval=-1e-2, maxval=1e-2), 
                        b_init=jnp.zeros, activation=jax.nn.relu)

    def __call__(self, x):
        return self.model(x)

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
    return jnp.mean(jnp.sum(jnp.abs(rem_tt), axis=-1))


# Define a function to compute relative error
@jax.jit
def _rel_error(eststate, truestate):
    """ Compute the relative error between two quantities
    """
    return jnp.sum(jnp.abs(eststate - truestate)) / jnp.sum(jnp.abs(eststate) + jnp.abs(truestate))


# function assumes ts = [0, 1]
def init_model(rng, order, n_step, method='tayla', ts = 1.0, batch_size=1, 
                    pen_remainder= 0, atol=1.4e-8, rtol=1.4e-8):
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

    # Number of state of the system (should include time if dynamics is time-dependent)
    nstate = MLPDynamics.nstate

    # Build the ODE function module
    dynamics = hk.without_apply_rng(hk.transform(wrap_module(MLPDynamics)))
    ode_init = jnp.zeros((batch_size, nstate))
    dynamics_params = dynamics.init(rng, ode_init)
    dynamics_wrap = lambda x, _params : dynamics.apply(_params, x)

    # Build the Midpoint or remainder function module
    midpoint = hk.without_apply_rng(hk.transform(wrap_module(Midpoint)))
    midpoint_init = jnp.zeros((batch_size, nstate))
    midpoint_params = midpoint.init(rng, midpoint_init)
    midpoint_wrap = lambda x, _params : midpoint.apply(_params, x)

    ####################################################################################
    odeint_eval = odeint_rev(dynamics_wrap, time_step, n_step=n_step, atol=atol, rtol=rtol)
    # Build the prediction function based on the current method
    if method == 'tayla':       # Taylor lagrange with a learned remainder
        pred_fn = tayla((dynamics_wrap, midpoint_wrap), time_step, order=order, n_step=n_step)
        pred_params = (dynamics_params, midpoint_params)
    elif method == 'taylor':    # Truncated Taylor expansion
        pred_fn = tayla((dynamics_wrap, ), time_step, order=order, n_step=n_step)
        pred_params = (dynamics_params, )
    elif method == 'rk4': # Fixed time-setp RK4 method
        pred_fn = odeint_rev(dynamics_wrap, time_step, n_step=n_step)
        pred_params = (dynamics_params, )
    elif method == 'dopri5':
        pred_fn = odeint_eval
        pred_params = (dynamics_params, )
    elif method == 'hypersolver':
        pred_fn = hypersolver((dynamics_wrap, midpoint_wrap), time_step, order=order, n_step=n_step)
        pred_params = (dynamics_params, midpoint_params)
    elif method == 'hypersolver_grid':
        pred_fn = hypersolver((dynamics_wrap, ), time_step, order=order, n_step=n_step)
        pred_params = (dynamics_params, )
    else:
        raise NotImplementedError('Method {} is not implemented yet'.format(method))
    ####################################################################################

    # Define the forward function -> No extra operations on this dynamical system
    def forward_gen(params, x0, predictor):
        """ Compute the forward function """
        return predictor(x0, *params)

    forward = jax.jit(lambda params, x0 : forward_gen(params, x0, pred_fn))
    odeint_forward = jax.jit(lambda params, x0 : forward_gen(params, x0, odeint_eval))

    # Define the forward loss
    def forward_loss(params, batch_x, batch_nextx):
        """ Compute the loss function from the forward prediction """
        (est_nextx, nfe), extra = forward(params, batch_x)
        sloss = _pred_loss(est_nextx, batch_nextx)
        if extra is not None and pen_remainder > 0:
            sloss += pen_remainder * _rem_loss(extra)
        return sloss

    return pred_params, forward, forward_loss, odeint_forward

def init_data():
    """ Initialize data from a file containing trajectories """
    # Load the file containing the trajectory
    mFile = open(args.trajfile, 'rb')
    mData = pickle.load(mFile)
    mFile.close()

    # Get the training data set
    xTrainList = np.asarray(mData.xTrainList)
    xTrainNext = np.asarray(mData.xNextTrainList)

    # Get the testing data set
    xTestList = np.asarray(mData.xTestList)
    xTestNext = np.asarray(mData.xNextTestList)

    # Get the number of trajectories in the data set
    num_traj_data = mData.trajectory_length

    # Get the integration time step in the data set
    time_step = mData.time_step

    # Get the total number of training and testing data points
    num_train = xTrainList.shape[0]
    num_test = xTestList.shape[0]
    print(num_train, num_test)

    assert num_train % args.train_batch_size == 0
    num_train_batches = num_train // args.train_batch_size

    assert num_test % args.test_batch_size == 0
    num_test_batches = num_test // args.test_batch_size

    meta = {
        "num_train_batches": num_train_batches,
        "num_test_batches": num_test_batches,
        "num_train" : num_train,
        "num_test" : num_test,
        "time_step" : time_step
    }

    # Return iter element on the training and testing set
    return (xTrainList, xTrainNext[0,:,:]) , (xTestList, xTestNext[0,:,:]), meta

def shuffle_and_split(rng, x, y, num_split, shuffle=True):
    """ Shuffle a set of data and split them into chunck of batches """
    if not shuffle:
        return np.split(x, num_split), np.split(y, num_split)
    indx = np.arange(x.shape[0])
    rng.shuffle(indx)
    return np.split(x[indx,:], num_split), np.split(y[indx,:], num_split)

def evaluate_loss(m_params, forward_fn, data_eval, num_iter, fwd_odeint=None):
    """ Compute the metrics for evaluation accross the data set
        :param m_params    : The parameters of the ode and the midpoints (if computed)
        :param forward_fn  : A function that computes the rollout trajectory
        :param data_eval   : The dataset considered for metric computation (iterable)
        :param num_iter    : Number of iterations over each chunck in the dataset
        :param fwd_odeint  : A function to compute the solution via adaptive time step
    """
    # Store metrics using forward_fn
    loss_values, pred_time, nfe_val, lossrem = [np.zeros(num_iter) for i in range(4)]

    # Store metrics using the adaptive time step solver
    ode_loss_values, ode_pred_time, ode_nfe_val, ode_normdiff = [np.zeros(num_iter) for i in range(4)]

    for n_i in tqdm(range(num_iter),leave=False):
        # Extract the current data
        xstate, xstatenext = next(data_eval[0]), next(data_eval[1])

        # Infer the next state values of the system
        curr_time = time.time()
        (est_nextx, nfe) , extra = forward_fn(m_params, xstate)
        est_nextx.block_until_ready()
        diff_time  = time.time() - curr_time

        # Compute the loss function and the remainder loss (norm)
        lossval = _pred_loss(est_nextx, xstatenext)
        loss_rem = -1 if extra is None else _rem_loss(extra)

        # Save the data for logging
        pred_time[n_i]=diff_time; loss_values[n_i]=lossval; nfe_val[n_i]=nfe; lossrem[n_i]=loss_rem

        # If comparisons with odeint are requested
        if fwd_odeint is not None:
            # Infer the next state using the adaptive time step solver
            curr_time = time.time()
            (est_nextx_odeint, nfe_odeint) , _ = fwd_odeint(m_params[:1], xstate)
            est_nextx_odeint.block_until_ready()
            diff_time_odeint  = time.time() - curr_time

            # COmpute the loss function from the adaptive solver solution
            lossval_odeint = _pred_loss(est_nextx_odeint, xstatenext)

            # Compare the integration by the adaptive time step and our approach
            diff_predx = _rel_error(est_nextx_odeint, est_nextx)

            # Save the results
            ode_loss_values[n_i]=lossval_odeint; ode_nfe_val[n_i]=nfe_odeint; ode_pred_time[n_i]=diff_time_odeint; ode_normdiff[n_i]=diff_predx

    # Return the solution depending on if the adaptive solver is given or not 
    if fwd_odeint is None:
        return (np.mean(loss_values), np.mean(pred_time), np.mean(nfe_val), np.mean(lossrem)),(-1,-1,-1,-1)
    else:
        m_eval = (np.mean(loss_values), np.mean(pred_time), np.mean(nfe_val), np.mean(lossrem))
        ode_eval = (np.mean(ode_loss_values), np.mean(ode_pred_time), np.mean(ode_nfe_val), np.mean(ode_normdiff))
        return m_eval, ode_eval

if __name__ == "__main__":
    # python learn_dynamics.py --train_batch_size 500 --test_batch_size 250 --lr_init 1e-2 --lr_end 1e-3 --test_freq 1000 --save_freq 20000 --n_steps 1 --pen_remainder 1e-2 --nepochs 5000 --w_decay 1e-3 --grad_clip 0 --method tayla --order 1 --atol 1e-5 --rtol 1e-5 --mid_lr_init 1e-3 --mid_lr_end 1e-5 --mid_freq_update 500 --mid_num_grad_iter 1 --mid_grad_clip 0 --trajfile data/vanderpool_dt0.04.pkl
    
    # Parse the command line argument
    parser = argparse.ArgumentParser('Learning Dynamics of a stiff linear system')

    parser.add_argument('--method',  type=str, default='tayla')

    parser.add_argument('--nepochs', type=int, default=5000)
    parser.add_argument('--train_batch_size', type=int, default=500)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--train_num_batch_eval', type=int, default=-1)

    parser.add_argument('--order', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=1)

    parser.add_argument('--trajfile',  type=str, default='data/stifflinear_dt0.05.pkl')
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

    args = parser.parse_args()

    # Extrat the directory of the data file and the file name
    trajdir_ = Path(args.trajfile)
    trajdir = str(trajdir_.parent)+'/'
    trajfile = trajdir_.name

    # Random number generator for numpy variables
    m_numpy_rng = np.random.default_rng(args.seed)

    # Generate the random key generator -> A workaround for a weird jax + tensorflow core dumped error if it is placed after init_data
    rng = jax.random.PRNGKey(args.seed)

    # Initialize the data set
    (ds_train_x, ds_train_xnext), (ds_train_eval_x, ds_train_eval_xnext), meta = init_data()
    num_train_batches_eval = meta['num_train_batches'] if args.train_num_batch_eval <= 0 else args.train_num_batch_eval

    # Print some meta information
    print('Meta information learning : \n', meta)
    print(vars(args))

    ############### Forward model and Loss functions ##################
    # Compute the function for forward and forward loss and corrections
    pred_params, forward, forward_loss, odeint_eval = \
            init_model(rng, args.order, args.n_steps, args.method, ts = meta['time_step'], 
                        batch_size=args.train_batch_size, pen_remainder= args.pen_remainder, 
                        atol=args.atol, rtol=args.rtol)

    ##################### Build the optimizer for ode weights #########################
    # Customize the gradient descent algorithm
    chain_list = [optax.scale_by_adam(b1=0.999,b2=0.9999)]

    # Add weight decay if enable
    decay_weight = args.w_decay
    if decay_weight > 0.0:
        chain_list.append(optax.add_decayed_weights(decay_weight))

    # Add scheduler for learning rate decrease --> Linear decrease given bt learning_rate_init and learning_rate_end
    # m_schedule = optax.piecewise_constant_schedule(-args.lr_init, {meta['num_train_batches']*2500 : 1e-1})
    # m_schedule = optax.linear_schedule(-args.lr_init, -args.lr_end, args.nepochs*meta['num_train_batches'])
    # m_schedule = optax.cosine_decay_schedule(-args.lr_init, args.nepochs*meta['num_train_batches'])
    assert args.lr_init >= args.lr_end, 'Ending learning rate should be greater than starting learning rate'
    m_schedule = optax.exponential_decay(-args.lr_init, args.nepochs*meta['num_train_batches'], args.lr_end / args.lr_init)

    chain_list.append(optax.scale_by_schedule(m_schedule))

    # Add gradient clipping if enable
    if args.grad_clip > 0.0:
        chain_list.append(optax.adaptive_grad_clip(clipping=args.grad_clip))

    # Check if correction is enable--> Only valid for Tayla
    no_correction = args.mid_freq_update < 0 or args.method != 'tayla'

    # Build the optimizer
    opt = optax.chain(*chain_list)
    opt_state = opt.init(pred_params if no_correction else pred_params[0] ) # Here we only update the parameters of the dynamics function

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
    mid_opt_state = mid_opt.init(pred_params[1]) if len(pred_params) == 2 else None # No params in case of noresidual-based solvers
    mid_init  = pred_params[1] if len(pred_params) == 2 else None

    ######################## Update function ##########################

    @jax.jit
    def update(params, _opt_state, xstate, xstatenext):
        """ Define the update rule for the parameters of the ODE
            :param params         : A tuple containing parameters of ODE and midpoint
            :param _opt_state     : The current state of the optimizer
            :param xstate         : A bactch of trajectories of the system
            :param xstatenext     : A bacth of next state value of the system
        """
        if not no_correction:
            params_dyn = params[0]
            dyn_loss = lambda dyn_params : forward_loss((dyn_params,*params[1:]), xstate, xstatenext)
        else:
            params_dyn = params
            dyn_loss = lambda dyn_params : forward_loss(dyn_params, xstate, xstatenext)
        grads = jax.grad(dyn_loss, has_aux=False)(params_dyn)
        updates, _opt_state = opt.update(grads, _opt_state, params_dyn)
        params_dyn = optax.apply_updates(params_dyn, updates)
        if not no_correction:
            return (params_dyn, *params[1:]), _opt_state
        else:
            return params_dyn, _opt_state
        # return (params_dyn, *params[1:]), _opt_state

    @jax.jit
    def mid_update(params, _opt_state, xstate):
        """ Update rule for the midpoint parameters
            :param params         : A tuple containing parameters of ODE and midpoint
            :param _opt_state     : The current state of the optimizer
            :param xstate         : A bactch of trajectories of the system
        """
        mid_params = params[1] # Assume params contain the midpoint parameters
        (x_odeint, nfe_), _ = odeint_eval(params[:1], xstate)
        # Define the loss function
        residual_loss = lambda params_mid : forward_loss((params[0],params_mid), xstate, x_odeint)
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
    opt_params_dict, opt_loss_train, opt_loss_test, opt_rem_test, opt_rem_train, opt_nfe, opt_diff, opt_ode_loss = [None] * 8

    # Save the loss evolution and other useful quantities
    total_time, compute_time_update, update_time_average = 0, list(), 0.0
    loss_evol_train, loss_evol_test, predtime_evol_train, predtime_evol_test, \
        constr_rem_evol_train, constr_rem_evol_test, loss_evol_odeint, nfe_evol_odeint, \
        err_evol_odeint, predtime_evol_odeint= [list() for i in range(10)]

    # Save all the command line arguments of this script
    m_parameters_dict = vars(args)
    out_data_file = trajdir +'dyn_{}_{}_o{}_s{}'.format(trajfile.split('.pkl')[0], args.method, args.order, args.n_steps)

    # Open the info file to save the command line print
    outfile = open(out_data_file+'_info.txt', 'w')
    outfile.write('Training parameters: \n{}'.format(m_parameters_dict))
    outfile.write('////// Command line messages \n\n')
    outfile.close()

    # When evaluation the data, we consider a non-shuffle data set
    t_train, t_train_next = shuffle_and_split(m_numpy_rng, ds_train_x, ds_train_xnext, meta['num_train_batches'], shuffle=False)
    ds_test_c_, ds_test_c_next_ = shuffle_and_split(m_numpy_rng, ds_train_eval_x, ds_train_eval_xnext, meta['num_test_batches'], shuffle=False)

    # Start the iteration loop
    for epoch in tqdm(range(args.nepochs)):
        # Shuffle the entire data set at each epoch and return iterables
        ds_train_, ds_train_c_next_ = shuffle_and_split(m_numpy_rng, ds_train_x, ds_train_xnext, meta['num_train_batches'], shuffle=True)
        # Build the iteratble
        ds_train_c, ds_train_c_next = iter(ds_train_), iter(ds_train_c_next_)

        # Iterate on the total number of batches
        for i in tqdm(range(meta['num_train_batches']), leave=False):
            # Get the next batch of images
            xTrain, xTrainNext = next(ds_train_c), next(ds_train_c_next)

            # Increment the iteration count
            itr_count += 1

            # Update the weight of the neural network representing the ODE
            update_start = time.time()
            pred_params, opt_state = update(pred_params, opt_state, xTrain, xTrainNext)
            tree_flatten(opt_state)[0][0].block_until_ready()
            update_end = time.time() - update_start

            # In case there is an update rule for the midpoint -> Do as following
            if (not no_correction) and mid_opt_state is not None and itr_count % args.mid_freq_update == 0:
                itr_count_corr += 1
                update_start = time.time()
                pred_params, mid_opt_state = mid_update(pred_params, mid_opt_state, xTrain)
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
                (loss_values_train, pred_time_train, nfe_train, contr_rem_train), _ = \
                        evaluate_loss(pred_params, forward, (iter(t_train),iter(t_train_next)), num_train_batches_eval)

                # Compute the loss on the entire testing set 
                (loss_values_test, pred_time_test, nfe_test, contr_rem_test), (ode_ltest, ode_predtime, ode_nfe, ode_errdiff) = \
                        evaluate_loss(pred_params, forward,  (iter(ds_test_c_), iter(ds_test_c_next_)), meta['num_test_batches'],
                                        fwd_odeint=None if args.no_compare_odeint else odeint_eval)

                # First time we have a value for the loss function
                if opt_loss_train is None or opt_loss_test is None or (opt_loss_test > loss_values_test):
                    opt_params_dict, opt_loss_train, opt_loss_test, opt_rem_train, opt_rem_test, opt_nfe, opt_diff, opt_ode_loss = \
                        pred_params, loss_values_train, loss_values_test, contr_rem_train, contr_rem_test, ode_nfe, ode_errdiff, ode_ltest 

                # Do some printing for result visualization
                print_str = 'Iter {:05d} | Total Update Time {:.2e} | Update time {:.2e}\n'.format(itr_count, total_time, update_end)
                print_str += '[    Train     ] Loss = {:.2e} | Pred. Time. = {:.2e} | NFE = {:.3e} |  Rem. Val.   = {:.2e}\n'.format(loss_values_train, pred_time_train, nfe_train, contr_rem_train)
                print_str += '[    Test      ] Loss = {:.2e} | Pred. Time. = {:.2e} | NFE = {:.3e} |  Rem. Val.   = {:.2e}\n'.format(loss_values_test, pred_time_test, nfe_test, contr_rem_test)
                print_str += '[  ODEINT Test ] Loss = {:.2e} | Pred. Time. = {:.2e} | NFE = {:.3e} | Diff. Pred.  = {:.2e}\n'.format(ode_ltest, ode_predtime, ode_nfe, ode_errdiff)
                print_str += '[  OPT. Value. ] Loss Train = {:.2e} | Loss Test = {:.2e} | Rem Train = {:.2e} | Rem Test = {:.2e}\n'.format(
                                opt_loss_train, opt_loss_test, opt_rem_train, opt_rem_test)
                print_str += '                 Loss ODE   = {:.2e} | NFE ODEINT = {:.3e} | Diff. Pred.  = {:.2e}\n'.format(opt_ode_loss, opt_nfe, opt_diff)
                # w_ode = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(pred_params[0]))
                # w_mid = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(pred_params[1])) if mid_opt_state is not None else 0
                # print_str += '[  Weigh loss  ] Weight Mid = {:.3e} | Weight ODE  = {:.3e}\n'.format(w_mid, w_ode)
                tqdm.write(print_str)

                # Save all the obtained data
                loss_evol_train.append(loss_values_train); loss_evol_test.append(loss_values_test); predtime_evol_train.append(pred_time_train)
                predtime_evol_test.append(pred_time_test); constr_rem_evol_train.append(contr_rem_train); constr_rem_evol_test.append(contr_rem_test)
                loss_evol_odeint.append(ode_ltest); nfe_evol_odeint.append(ode_nfe), err_evol_odeint.append(ode_errdiff); predtime_evol_odeint.append(ode_predtime)

                # Save these info of the console in a text file
                outfile = open(out_data_file+'_info.txt', 'a')
                outfile.write(print_str_test)
                outfile.write(print_str)
                outfile.close()

            if itr_count % args.save_freq == 0 or (epoch == args.nepochs-1 and i == meta['num_train_batches']-1):
                m_dict_res = {'best_params' : opt_params_dict, 'total_time' : total_time, 'compute_time_update' : compute_time_update,
                                'opt_loss_train' : opt_loss_train, 'opt_loss_test' : opt_loss_test, 'opt_nfe_test' : opt_nfe, 
                                'opt_rem_test' : opt_rem_test, 'opt_rem_train' : opt_rem_train, 'opt_diff_test' : opt_diff,
                                'loss_evol_train' : loss_evol_train, 'loss_evol_test' : loss_evol_test, 
                                'predtime_evol_train' : predtime_evol_train, 'predtime_evol_test' : predtime_evol_test, 'predtime_evol_odeint' : predtime_evol_odeint,
                                'loss_evol_odeint' : loss_evol_odeint, 'err_evol_odeint' : err_evol_odeint,
                                'nfe_evol_odeint' : nfe_evol_odeint, 'constr_rem_evol_train' : constr_rem_evol_train, 'constr_rem_evol_test' : constr_rem_evol_test, 
                                'training_parameters' : m_parameters_dict}
                outfile = open(out_data_file+'_res.pkl', "wb")
                pickle.dump(m_dict_res, outfile)
                outfile.close()