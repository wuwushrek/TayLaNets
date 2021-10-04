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


class MLPDynamics(hk.Module):
    """
    Dynamics for ODE as an MLP.
    """
    def __init__(self, dim):
        """ Build a 2-layer NN where dim specifies the flatten dimension of the input image
        """
        super(MLPDynamics, self).__init__()
        self.dim = dim
        self.model = hk.nets.MLP(output_sizes=(256,256, self.dim), b_init=jnp.zeros, activation=jax.nn.sigmoid,
                                    w_init = hk.initializers.RandomUniform(minval=-0.1, maxval=0.1), )

    def __call__(self, x):
        return self.model(x)

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


# function assumes ts = [0, 1]
def init_model(rng, taylor_order, batch_size=1, midpoint_layers=(12,12), approx_mid=True, time_step=0.1):
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
    ts = jnp.array([0., time_step])

    # Discretize the integration time using the given number of step
    # time_indexes = jnp.array([ time_step * (i+1) for i in range(number_step)])

    # MNIST problem size and dummy random intiial values
    num_state = 2
    midpoint_shape = num_state
    ode_init = jnp.zeros((batch_size, num_state))
    midpoint_init = jnp.zeros((batch_size, midpoint_shape))

    # Build the ODE function module
    dynamics = hk.without_apply_rng(hk.transform(wrap_module(MLPDynamics, num_state)))
    dynamics_params = dynamics.init(rng, ode_init)
    dynamics_wrap = dynamics.apply

    # Build the Midpoint function module
    midpoint = hk.without_apply_rng(hk.transform(wrap_module(Midpoint, midpoint_shape, midpoint_layers, approx_mid)))
    midpoint_params = midpoint.init(rng, midpoint_init)
    midpoint_wrap = midpoint.apply

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
    def pred_xnext(params : Tuple[hk.Params, hk.Params], state_t : jnp.ndarray): #, t : float) -> jnp.ndarray:
        """ Predict the next using our Taylor-Lagrange expansion with learned remainder"""
        params_dyn, params_mid = params

        # Define the vector field of the ODE given a pre-process input
        def vector_field(state_val : jnp.ndarray) -> jnp.ndarray:
            return dynamics_wrap(params_dyn, state_val)

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

    @jax.jit
    def m_forward(params, state):
        (_dynamics_params, _midpoint_params) = params
        next_x, _, _ = pred_xnext((_dynamics_params, _midpoint_params), state)
        return next_x

    # Regroup the parameters of this entire module
    m_params = (dynamics_params, midpoint_params)

    return m_params, m_forward, dynamics_wrap

from generate_sample import system_ode, numeric_solution, load_data_yaml



if __name__ == "__main__":
    # Parse the command line argument
    parser = argparse.ArgumentParser('Neural ODE MNIST')
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
    parser.add_argument('--n_rollout', type=int, default=-1)
    parser.add_argument('--dirname', type=str, default='neur_train')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--count_odeint_nfe',  action="store_true")
    parser.add_argument('--method',  type=str, default='tayla') # choices odeint or odeint_grid
    parser.add_argument('--validation_set',  action="store_true")
    parser.add_argument('--no_midpoint',  action="store_false")
    parser.add_argument('--atol', type=float, default=1.4e-8)
    parser.add_argument('--rtol', type=float, default=1.4e-8)
    parser.add_argument('--num_steps', type=int, default=2)
    parser.add_argument('--patience', type=int, default=-1) # For early stopping criteria
    args = parser.parse_args()

    # Load the config file
    mdata_log, _ = load_data_yaml('dataconfig.yaml', {})
    print(mdata_log)

    # Number of training trajectories

    (xtrain_lb, utrain_lb) = mdata_log.xu_train_lb
    (xtrain_ub, utrain_ub) = mdata_log.xu_train_ub 
    (xtest_lb, utest_lb) = mdata_log.xu_test_lb
    (xtest_ub, utest_ub) = mdata_log.xu_test_ub 

    # System dimension
    nstate = 2
    ncontrol = 0


    n_rollout = None if args.n_rollout == -1 else args.n_rollout
    m_numpy_rng = np.random.default_rng(args.seed)
    # Generate the random key generator -> A workaround for a weird jax + tensorflow core dumped error if it is placed after init_data
    rng = jax.random.PRNGKey(args.seed)

    # Store the data in each results file
    file_name = {'rk4' : 'results/rk4_ode.pkl', 'odeint' : 'results/odeint_ode.pkl', 'tayla' : 'results/tayla_ode.pkl'}
    m_best_params = dict()
    dynamic_models = dict()
    problem_params = None

    # Build the solver
    _count_nfe = None if not args.count_odeint_nfe else (args.atol, args.rtol)
    midpoint_hidden_layer_size = (32,32) # Hidden layers of the midpoint neural network
    _, m_forward, dynamics_wrap = \
                    init_model(rng, args.taylor_order, midpoint_layers=midpoint_hidden_layer_size, 
                                approx_mid = args.no_midpoint, time_step=mdata_log.time_step)

    # results file
    for name, f in file_name.items():
        mFile = open(f, 'rb')
        l_data = pickle.load(mFile)
        mFile.close()
        dynamic_models[name] =jax.jit( lambda state, t : dynamics_wrap(l_data['best_params'][0], state))
        dynamic_models[name](jnp.array([0,1.0]), 0.0)
        # m_best_params[name]  = copy.deepcopy(l_data['best_params'])
        problem_params = l_data['training_parameters']
        print(problem_params,'\n\n')

    # # My dynamics
    # dynamic_models = dict()
    # for name, params in m_best_params.items():
    #     print(name)
    #     dynamic_models[name] =jax.jit( lambda state, t : dynamics_wrap(params[0], state))

    # Solve the dynamics using scipy
        # Set of initial states training 
    rng, subkey = jax.random.split(rng)
    num_traj_data = 10
    m_init_train_x = jax.random.uniform(subkey, (num_traj_data, nstate), minval = jnp.array(xtest_lb), maxval=jnp.array(xtest_ub))

    # Generate the training trajectories
    integ_time_step = 0.01 # mdata_log.time_step
    trajectory_length = 1000 # mdata_log.trajectory_length
    trueTraj, _ = numeric_solution(system_ode, m_init_train_x, integ_time_step, trajectory_length, 1, merge_traj=False)
    rk4Traj, _ = numeric_solution(dynamic_models['rk4'], m_init_train_x, integ_time_step, trajectory_length, 1, merge_traj=False)
    odeint, _ = numeric_solution(dynamic_models['odeint'], m_init_train_x, integ_time_step, trajectory_length, 1, merge_traj=False)
    tayla, _ = numeric_solution(dynamic_models['tayla'], m_init_train_x, integ_time_step, trajectory_length, 1, merge_traj=False)

    # x_init = [m_init_train_x]
    # for _ in range(stop)

    time_index = [ i * integ_time_step for i in range(trajectory_length)]

    import matplotlib.pyplot as plt

    # Do some plotting
    state_label = [r'$x_{0}$', r'$x_{1}$']
    for i in range(nstate):
        plt.figure()
        plt.plot(time_index, rk4Traj[0][:,i], color='blue')
        plt.plot(time_index, odeint[0][:,i], color='green')
        plt.plot(time_index, tayla[0][:,i], color='magenta')
        plt.plot(time_index, trueTraj[0][:,i], color='red')
        plt.xlabel('Time (s)')
        plt.ylabel(state_label[i])
        plt.grid(True)

    # 2D dimensional plot
    plt.figure()
    plt.plot(rk4Traj[0][:,0], rk4Traj[0][:,1], linewidth=2, color='blue')
    plt.plot(odeint[0][:,0], odeint[0][:,1], linewidth=2, color='green')
    plt.plot(trueTraj[0][:,0], trueTraj[0][:,1], linewidth=2, color='red')
    plt.plot(tayla[0][:,0], tayla[0][:,1], linewidth=2, color='magenta')
    plt.xlabel(state_label[0])
    plt.ylabel(state_label[1])
    plt.grid(True)

    plt.show()