# Import JAX and utilities
import jax
import jax.numpy as jnp
from jax.experimental import jet
from jax.experimental.ode import odeint as jax_odeint

from jax import lax

def taylor_order_n(vector_field_fn, state : jnp.ndarray, order : int):
    """ Compute higher-order Taylor expansion and return the high-order derivatives
        :param vector_field_fn : The function of the state defining the vector field of the ODE
        :param state : The state at which to evaluate the Taylor expansion
        :param order: The order of the Taylor expansion --> Order i means term f^(i) wuth f^(0) = f, f^(1) = f'....
    """
    y0 = vector_field_fn(state)
    # In case it is a zero order, we return only f(x0) = vector_field_fn(state)
    if order == 0:
        return y0[None]
    y1 = jax.jvp(vector_field_fn, (state,), (y0, ))[1]
    # In the case it is first order, just stop here
    if order == 1:
        return jnp.stack((y0,y1))
    # For higher order derivatives of the vector field, use the previously computed y0 and y1
    yn = [y1]
    order = order-1
    # TODO use lax.scan or lax.for for avoiding unrolling this loop
    for _ in range(order):
        (y0, [*yn]) = jet.jet(vector_field_fn, (state,), ((y0, *yn), ))
    return jnp.stack((y0, *yn))


def midpoint_constraints(midpoint_val: jnp.ndarray, state : jnp.ndarray, nextstate : jnp.ndarray):
    """ Provide the required constraints to enforce such that the midpoint estimation is between current state
        and possible next state value -> Returns f <= 0 where f concatenate the two inequalities constraints
        :param midpoint_val : The midpoint value in the Taylor Lagrange expansion
        :param state : The current state of the system
        :nextstate : The next state of the system
    """
    min_point = jnp.minimum(state, nextstate)
    max_point = jnp.maximum(state, nextstate)
    lb_constr = min_point - midpoint_val
    ub_constr = midpoint_val -  max_point
    return jnp.hstack((lb_constr, ub_constr))


def ode_truncated_taylor(vector_field_fn, time_step, taylor_order):
    """ Compute a function that returns a truncated taylor expansion around state_t of the ode 
        \dot{state_t} = vector_field_fn(state_t) with a fixed time step given by time_step and 
        the taylor_order of the Taylor expansion given by order.
        Besides, state_t = [state, t] if the dynamics is time dependent
        :param vector_field_fn : A function representing the dynamics function/vector field
        :param time_step : The fixed time step of the taylor expansion
        :taylor_order : The order of the Taylor expansion
    """
    assert taylor_order > 0, 'The Taylor order = {} is not positive'.format(taylor_order)
    # Compute the coefficient in the taylor expansion without the derivatives coefficients
    pow_timestep = []
    start_val = 1.0 # Store the current Taylor coefficient dt^j / facto(j) until taylor_order+1
    for i in range(taylor_order+1):
        m_coeff = start_val * (time_step / (i+1))
        pow_timestep.append(m_coeff)
        start_val = m_coeff

    # Extract the remainder coefficient
    rem_coeff = pow_timestep[-1]

    # Extract the truncated coefficient and spawn the array to match the expansion derivatives and batch size
    pow_timestep = jnp.array(pow_timestep[:-1]).reshape((-1,1,1))

    # Truncated Taylor expansion
    def ode_tt(state_t, *args):
        """ Compute the solution of the ODE expansion around the current state state_t
        """
        vector_field_fn_ =  lambda x : vector_field_fn(x, *args)
        m_expansion = taylor_order_n(vector_field_fn_, state_t, taylor_order-1)
        return state_t + jnp.sum(pow_timestep * m_expansion, axis=0), m_expansion[0]

    # Remainder term of the truncated taylor expansion given the midpoint
    def rem_tt(midpoint_t, *args):
        """ Compute the remainder of the ODE expansion given the midpoint
        """
        vector_field_fn_ =  lambda x : vector_field_fn(x, *args)
        return rem_coeff * taylor_order_n(vector_field_fn_, midpoint_t, taylor_order)[-1]
    return ode_tt, rem_tt

def euler_step(func, y0, time_step, *args):
    """ Compute a Euler step (2nd order method) with fixed time step
        :param func : The ODE function -> The function is vectorized
        :param y0 : The the state at time step 0
        :param time_step : The time step of the integration scheme
    """
    return y0 + time_step * func(y0, *args)

def heun_step(func, y0, time_step, *args):
    """ Compute a Heun step (2nd order method) with fixed time step
        :param func : The ODE function -> It is assumed to not be vectorized
        :param y0 : The the state at time step 0
        :param time_step : The time step of the integration scheme
    """
    k1 =  func(y0, *args)
    k2 = func(y0 + time_step * k1, *args)
    return y0 + 0.5 * time_step * (k1 + k2)

def midpoint_step(func, y0, time_step, *args):
    """ Compute a Midpoint step (2nd order method) with fixed time step
        :param func : The ODE function -> It is assumed to not be vectorized
        :param y0 : The the state at time step 0
        :param time_step : The time step of the integration scheme
    """
    k1 =  func(y0, *args)
    k2 = func(y0 + 0.5 * time_step * k1, *args)
    return y0 + time_step * k2

def bosh_step(func, y0, time_step, *args):
    """ Compute a Bosh step (3rd order method) with fixed time step
        :param func : The ODE function -> The function is vectorized
        :param y0 : The the state at time step 0
        :param time_step : The time step of the integration scheme
    """
    k1 = func(y0, *args)
    k2 = func(y0 + time_step * k1, *args)
    k3 = func(y0 + 0.5*time_step*(k1 + k2)*0.5, *args)
    return y0 + (k1 + k2 + 4*k3)*(time_step/6.0)

def rk4_step(func, y0, time_step, *args):
    """ Compute a Bosh step (3rd order method) with fixed time step
        :param func : The ODE function -> The function is vectorized
        :param y0 : The the state at time step 0
        :param time_step : The time step of the integration scheme
    """
    k1 = func(y0, *args)
    k2 = func(y0 + 0.5 * time_step * k1, *args)
    k3 = func(y0 + 0.5 * time_step * k2, *args)
    k4 = func(y0 + time_step * k3, *args)
    return y0 + (k1 + 2*k2 + 2*k3 + k4)*(time_step/6.0)

def midpoint_eval(time_step, mid_fn):
    """ Return a function that computes the remainder of the 
        :param time_step : The time step of the integration scheme
        :param mid_fn    : A function to compute the coefficient in the midpoint
    """
    def full_midpoint(x0, f0, *params):
        # Compute the midpoint coefficient
        mid_coeff = mid_fn(x0, *params)
        # The midpoint is computed differently depending on if the coefficient is a matrix or vector
        ns = x0.shape[-1]
        if ns != mid_coeff.shape[-1]: 
            mid_p = x0 + jax.vmap(jax.numpy.matmul)(mid_coeff.reshape(-1,ns,ns), f0)
        else:
            mid_p = x0 + mid_coeff * f0
        return mid_p
    return full_midpoint

def tayla(dyn_fn, time_step, order=1, n_step=1):
    """ Provide a function to predict the state values at future time step using Taylor expansion or Taylor-Lagrange expansion
        :param dyn_fn       : A tuple containing either the function describing the dynamics or both
                              the function describing the dynamics and the function to compute the midpoint
        :param time_step    : The integration time step used in the Taylor expansion
        :param order        : The taylor expansion order
        :param n_step       : Number of steps in the integration scheme
    """
    order = int(order)
    assert n_step > 0 , 'Number of steps n_step = {} , must be positive'.format(n_step)
    time_indexes = jnp.array([time_step * (i+1) for i in range(n_step)])

    # Get the function to compute the higher order expansion and the remainder term
    ode_tt, rem_tt = ode_truncated_taylor(dyn_fn[0], time_step, order)

    # Function to compute the midpoint
    mid_fn = None if len(dyn_fn) ==1 else midpoint_eval(time_step, dyn_fn[1])

    # Get the number of function evaluations
    nfe = order**2 if len(dyn_fn) ==1 else (order+1)**2

    def pred_next(state, *params_args):
        """ Compute the state value at different time indexes
            :param state        : The current state of the system
            :param params_args  : Extra parameters used to call the functions of the dynamics or midpoint
        """
        def rollout(state_t, time_val):
            # Compute the taylor expansion and the function evaluation at initial state
            x_tt, f0 = ode_tt(state_t, *params_args[:1])
            # Compute the midpoint and the remainder if dyn_fn contains a function to provide the midpoint
            if len(dyn_fn) == 2:
                # Compute the midpoint
                mid_p = mid_fn(state_t, f0, *params_args[1:])
                # Compute the reaminder term
                r_tt = rem_tt(mid_p, *params_args[:1])
                state_next  = x_tt + r_tt
                # return state_next, (state_next, mid_p, r_tt)
                return state_next, r_tt
            else:
                # return x_tt, (x_tt, )
                return x_tt, None
        # Do a rollout over the time_indexes
        statenext, mid_rem = jax.lax.scan(rollout, state, time_indexes)
        return (statenext, nfe*(time_indexes.shape[0])), mid_rem
    return pred_next

def tayla_predcorr(dyn_fn, time_step, order=1, n_step=1, rtol=1.4e-8, atol=1.4e-8):
    """ Provide functions to compute truncated Taylor expansion, remainder term, and true next state from odeint
        :param dyn_fn       : A tuple containing either the function describing the dynamics or both
                              the function describing the dynamics and the function to compute the midpoint
        :param time_step    : The integration time step used in the Taylor expansion
        :param order        : The taylor expansion order
        :param n_step       : Number of steps in the integration scheme
        :param rtol         : relative tolerance error for the odeint solver
        :param atol         : absolute tolerance error for the odeint solver
    """
    order = int(order)

    # Get the function to compute the higher order expansion and the remainder term
    ode_tt, rem_tt = ode_truncated_taylor(dyn_fn[0], time_step, order)

    # Function to compute the midpoint
    mid_fn = None if len(dyn_fn) ==1 else midpoint_eval(time_step, dyn_fn[1])

    # TODO: vmap instead of scan here since no carry variable
    def trunc_pred(x0, *params):
        """ Return the truncated taylor expansion at a set of state inputs
        """
        def rollout(_useless, state_t):
            x_tt, f0 = ode_tt(state_t, *params)
            return _useless, (x_tt, f0)
        _, (trunc_x, f0v) = jax.lax.scan(rollout, None, x0)
        return trunc_x, f0v

    def corrector(x0, f0, *params):
        """ Return the remainder of the taylor expansion at a set of state inputs
        """
        def rollout(_useless, xs):
            (state_t, ft) = xs
            mid_p = mid_fn(state_t, ft, *params[1:])
            r_tt = rem_tt(mid_p, *params[:1])
            return _useless, r_tt
        _, r_tt = jax.lax.scan(rollout, None, (x0, f0))
        return r_tt

    assert n_step > 0 , 'Number of steps n_step = {} , must be positive'.format(n_step)
    time_indexes = jnp.array([time_step * i for i in range(n_step+1)])
    truth_pred = lambda y0, *params: jax_odeint(lambda _y, _t, *_params : dyn_fn[0](_y, *_params), 
                                                    y0, time_indexes, *params, atol=atol, rtol=rtol)

    return trunc_pred, corrector, truth_pred

def solve_with_jax_odeint(dyn_fn, time_step, n_step=1, rtol=1.4e-8, atol=1.4e-8):
    """ Provide a function to solve the ode (autonomous) for n_step, each of them spaced with a 
        timestep value given by time_step
        :param dyn_fn     : The function to integrate
        :param time_step  : The time_step of the integration scheme
        :param n_step     : The number of step to perform
        :param atol       : The abosulte tolerance error of the adaptive solver
        :param rtol       : The relative tolerance error of the adaptive solver
    """
    assert n_step > 0 , 'Number of steps n_step = {} , must be positive'.format(n_step)
    time_indexes = jnp.array([time_step * i for i in range(n_step+1)])
    truth_pred = lambda y0, *params: jax_odeint(lambda _y, _t, *_params : dyn_fn[0](_y, *_params), 
                                            y0, time_indexes, *params, atol=atol, rtol=rtol)[-1]
    return truth_pred


def hypersolver(dyn_fn, time_step, order=1, n_step=1):
    """ Provide a function to predict the state values at future time step using fixed rk methods
        and the concept of hypersolver (learning the approximation error)
        :param dyn_fn       : A tuple containing either the function describing the dynamics or both
                              the function describing the dynamics and the function to compute the midpoint
        :param time_step    : The integration time step used in the RK integration scheme
        :param order        : The taylor expansion order
        :param n_step       : Number of steps in the integration scheme
    """
    # Number of function evaluation
    nfe = int(order)
    # Get the actual RK method to apply
    fn2order = {1 : euler_step, 2 : heun_step, 2.5 : midpoint_step, 3 : bosh_step, 4 : rk4_step}
    m_step_fn = fn2order[order]
    time_indexes = jnp.array([time_step * (i+1) for i in range(n_step)])
    # Compute the function for future state prediction
    def pred_next(state, *params_args):
        """ Compute the state value at different time indexes
            :param state : The current state of the system
            :param time_indexes : An array with the different time indexes at which to evaluate the solution
            :param params_args : Extra parameters used to call the functions of the dynamics or midpoint
        """
        def rollout(state_t, time_val):
            # Compute the next state
            x_next = m_step_fn(dyn_fn[0], state_t, time_step, *params_args[:1])
            # Compute the midpoint and the remainder if dyn_fn contains a function to provide the midpoint
            if len(dyn_fn) == 2:
                # Compute the residual value
                residual = time_step**(nfe+1) * dyn_fn[1](state_t, *params_args[1:])
                state_next  = x_next +  residual
                return state_next, residual
            else:
                return x_next, None
        # Do a rollout over the time_indexes
        statenext, mid_rem = jax.lax.scan(rollout, state, time_indexes)
        return (statenext, nfe*(time_indexes.shape[0])), mid_rem
    return pred_next


# ODE integration with adjoint method for reverse differentiation
from ode import odeint, odeint_grid
def odeint_rev(dyn_fn, time_step, n_step=1, **args):
    """ Provide a function to predict the state values at future time step using fixed rk4 methods with memory efficient
        reverse autodiff, and an adaptive time step integration
        :param dyn_fn    : A function providing the vector field of the system
        :args : A set a parameter specifying if this a fixed-time step rk4 or adaptive dopri5
                atol, rtol or time_step
    """
    if 'atol' in args and 'rtol' in args:
        m_integ_fn = lambda y0, ts, *params: odeint(lambda _y, _t, *_params : dyn_fn(_y, *_params), y0, ts, *params, atol=args['atol'], rtol=args['rtol'])
    else:
        m_integ_fn = lambda y0, ts, *params: odeint_grid(lambda _y, _t, *_params : dyn_fn(_y, *_params), y0, ts, *params, step_size=time_step)
    time_indexes = jnp.array([time_step * (i+1) for i in range(n_step)])
    def pred_next(state, *params_args):
        out_ode, f_nfe = m_integ_fn(state, jnp.array([0.0, time_indexes[-1]]), *params_args)
        return (out_ode[-1], jnp.mean(f_nfe)), None
    return pred_next


def wrap_module(module, *module_args, **module_kwargs):
    """Wrap the module in a function to be transformed.
    """
    def wrap(*args, **kwargs):
        """
        Wrapping of module.
        """
        model = module(*module_args, **module_kwargs)
        return model(*args, **kwargs)
    return wrap



# m_fun = jax.vmap(lambda x : jnp.sin(x))

# x0 = jnp.array([[0,0.14,1.1,0.15]])
# order = 2
# time_step = 0.1
# nstep = 2
# # time_indexes = jnp.array([(i+1)*time_step for i in range(nstep)])

# m_val = lambda x : jnp.zeros(x0.shape)

# import time

# hyper_ode = hypersolver((m_fun,), time_step, n_step=nstep, order=order)
# print(hyper_ode(x0))

# tayla_ode = tayla((m_fun,), time_step, n_step=nstep, order=order)
# print(tayla_ode(x0))

# pred_ode = odeint_rev(m_fun, time_step, n_step=nstep, atol=1e-8, rtol=1e-8)
# print(pred_ode(x0))

# pred, corr, truth = tayla_predcorr((m_fun, m_val), time_step, n_step=nstep, order=1, rtol=1e-8, atol=1e-8)
# xn, f0 = pred(x0)
# print(xn)
# print(corr(x0, f0))

# truth = jax.jit(truth)
# truth(x0)
# c = time.time()
# truth(x0)
# print(time.time() - c)
# # print(truth(x0))

# pred_ode = odeint_rev(m_fun, n_step=nstep, atol=1e-8, rtol=1e-8)
# pred_ode = jax.jit(pred_ode)
# pred_ode(x0)


# c = time.time()
# pred_ode(x0)
# print(time.time() - c)

# # res = bosh_step(m_fun, x0, time_step)
# res = bosh_step(m_fun, x0, time_step)
# print(res)
# taylor_res_ = taylor_order_n(m_fun, x0, order)
# taylor_res = taylor_order_n_prev(x0, m_fun, order)

# print(jnp.sum(taylor_res_ - taylor_res))
# print(taylor_res)
# print(taylor_res_)

# exit()

# time_step = 0.2
# taylor_order = order
# rem, pow_ser = ode_truncated_taylor(m_fun, time_step, taylor_order)

# inv_m_fact = jet.fact(jnp.array([i+1 for i in range(taylor_order+1)]))
# print(inv_m_fact-1.0)
# inv_m_fact = 1.0 / inv_m_fact
# dt_square_over_2 = time_step*time_step*0.5
# pow_timestep = [time_step]
# for _ in range(taylor_order):
#     pow_timestep.append(pow_timestep[-1] * time_step)
# pow_timestep = jnp.array(pow_timestep) * inv_m_fact
# rem_coeff = pow_timestep[-1]
# print(rem - rem_coeff)
# print(jnp.min(jnp.abs(pow_timestep[:-1] - pow_ser)))
# print(pow_timestep[:-1] - pow_ser)
# print(pow_timestep)
# print(pow_ser)