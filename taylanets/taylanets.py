# Jax imports
import functools
import jax
import jax.numpy as jnp

from jax.experimental import jet
from jax.experimental.ode import odeint

# Haiku for Neural networks
import haiku as hk

# Optax for the optimization scheme
import optax

# Typing functions
from typing import Optional, Tuple

########################################################################################################################
# Define a structural neural network to learn unknown dynamics from trajectory based on a Taylor Lagrange expansion with 
#	both the vector field and the mid-point value (in the Mean value Theorem) approximated by Deep Neural Networks.
#	By learning such mid-point value, we provide improved accuracy over current differentiable ODE solver that does not
#	learn the mid-point and propagate the approximation error in the learning process.
#######################################################################################################################

def taylor_order_n(state : jnp.ndarray, vector_field_fn, order : int, y0 : Optional[jnp.ndarray] = None):
	""" Compute higher-order Taylor expansion and return the high-order derivatives
		:param state : The state at which to evaluate the Taylor expansion
		:param vector_field_fn : The function of the state defining the vector field of the ODE
		:param order: The order of the Taylor expansion --> Order i means term f^(i) wuth f^(0) = f, f^(1) = f'....
	"""
	if y0 is None:
		y0 = vector_field_fn(state)
	# y1 = jax.jvp(vector_field_fn, (state,), (y0, ))[1]
	# yn = [y1]
	# order = order-1
	yn = []
	# TODO use lax.scan or lax.for for avoiding unrolling this loop
	for _ in range(order):
		(y0, [*yn]) = jet.jet(vector_field_fn, (state,), ((y0, *yn), ))
	return jnp.array([y0, *yn])


def der_order_n(state : jnp.ndarray, vector_field_fn, order : int):
	""" Compute higher-order Taylor expansion and return the high-order derivatives
		:param state : The state at which to evaluate the Taylor expansion
		:param vector_field_fn : The function of the state defining the vector field of the ODE
		:param order: The order of the Taylor expansion --> Order i means term f^(i) wuth f^(0) = f, f^(1) = f'....
	"""
	y0 = vector_field_fn(state)
	yn = []
	# TODO use lax.scan or lax.for for avoiding unrolling this loop
	for _ in range(order):
		(y0, [*yn]) = jet.jet(vector_field_fn, (state,), ((y0, *yn), ))
	flist = [y0, *yn]
	return flist[-1]

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


# Physics-based neural networks
class VectorFieldNets(hk.Module):
	""" Structural Deep Neural Network with side information and constraints to provide an approximation of the unknown vector field
		The midpoint neural network in nn_params can approximate either the midpoint value or the entired remainder of the Taylor expansion.
		This paper approximates the midpoint value but we show here that the accuracy is improved when approximating such a point than the entire
		remainder as unicity problems can appear in the latter scenario.
	"""
	def __init__(self, ns, nu, nn_params={}, known_dynamics=None, name='vfield'):
		""" This initialization assumes the state is fully observable
			:params ns 						: The number of states of the system
			:params nu 						: The number of inputs of the system
			:params nn_params 				: Dictionary containing the parameters of the NN of each unknown terms + the midpoint value parameterization if it is needed
												nn_params = {'unknown_f_name' : {'input_index' : , 'output_sizes' : , 'w_init' : , 'b_init' : , 'with_bias' : , 'activation' :, 'activate_final' :}, 
															 ... , 'midpoint' : {'input_index' : , 'output_sizes' : , 'w_init' : , 'b_init' : , 'with_bias' : , 'activation' :, 'activate_final' :} }
			:params known_dynamics			: \dot{x} = f(x, u, g_1(x,u), g_2(x,u), ...) where f is the known part and the functions g_1(.), g_2(.) of the susbset of variables 
												and controls are potentially unknown. 'input_index' specifies the input dependency when merging the arrays x and u.
												g_1, g_2, ... must be present (with the same name) in the nn_params dictionary.
												THIS FUNCTION MUST BE VECTORIZED TO HANDLE BATCH INPUTS.
		"""
		# Initialoize the params with the current name
		super().__init__(name=name)

		# Sanity check 
		assert not (known_dynamics is None and len(nn_params) == 0), \
				'The vector field is fully unknown but not parameterized. In this case nn_params should not be empty and should provide a parameterization of the vector field'

		# In case the dynamics is not known, nn_params should not contain only midpoint
		assert not (known_dynamics is None and len(nn_params) == 1 and 'midpoint' in nn_params), 'In case the dynamics is completely unknown, nn_params should not contain only midpoint'

		# Save the number of states and the number of control inputs
		self._ns = ns
		self._nu = nu

		# Save the known part of the dynamics
		self._known_dynamics = known_dynamics

		# A dictionary to save the unknown parts of the dynamics
		self._unknown_terms = dict()

		# Build all the unknown networks and the midpoint value approximator
		self.build_unknown_nets(nn_params)

	def build_unknown_nets(self, nn_params):
		""" Define the neural network for each unknown variables and for the midpoint approximator (if required)
			and save it as attributes of this class
			:params nn_params : Define a dictionary with the parameters of each neural networks
		"""

		# Extract the neural network for the midpoint value approximation
		midpoint_term = nn_params.get('midpoint', None)
		if midpoint_term is not None:
			dictParams_cpy = {key : val for key, val in midpoint_term.items() if key != 'input_index'}
			outSize = (*dictParams_cpy['output_sizes'], self._ns)
			dictParams_cpy['output_sizes'] = outSize
			# Ignore the input index set as we now the dimension of the input for the remainder/midpoint value
			input_size = self._ns + self._nu # + 1 # The +1 is for the time step
			default_params = {'name' : 'midpoint', 'w_init' : hk.initializers.TruncatedNormal(1. / jnp.sqrt(input_size)), 
								'b_init' : jnp.zeros, 'activation' : jax.nn.relu, 'activate_final': False}
			self.midpoint = hk.nets.MLP(**({**default_params,**dictParams_cpy}))
		else:
			self.midpoint = None

		# Build the unknown terms in the vector field expression (if there is any)
		for var_name, dictParams in nn_params.items():
			if var_name == 'midpoint':
				continue
			# The size of the neural network should be specified
			assert 'output_sizes' in dictParams and 'input_index' in dictParams, 'Size of the output or input layers should be specified with the key <output_sizes>, <input_size>'
			dictParams_cpy = {key : val for key, val in dictParams.items() if key != 'input_index'}
			input_size = dictParams['input_index'].shape[0]
			default_params = {'name' : var_name, 'w_init' : hk.initializers.TruncatedNormal(1. / jnp.sqrt(input_size)), 
								'b_init' : jnp.zeros, 'activation' : jax.nn.relu, 'activate_final': False}
			self._unknown_terms[var_name] = (hk.nets.MLP(**({**default_params,**dictParams_cpy})), dictParams['input_index'])


	# @abstractmethod
	def vector_field_and_unknown(self, x, u=None, extra_args=None):
		""" This function defines the underlying dynamics of the system and encodes the known terms. 
			Specifically, we assume that \dot{x} = f(x, u, g_1(x,u), g_2(x,u), ...) where f is known but
			the functions g_1(.), g_2(.) of the susbset of variables and controls are potentially
			unknown. We encode these unknown functions as a set of small size feed forward neural networks 
			for which we develop a custom forward and back propagation to train their parameters.

			The function returns both the resulting vector field and the estimate of the unknown terms

			:params x : The current state vector
			:params u : The current control value
			:param extra_args	: State dependent extra parameters used in side information and constraints
		"""

		# Fuse the vector x and vector u into one vector (assume to be row vectors)
		fus_xu = jnp.hstack((x,u)) if u is not None else x

		# Need to handle the case where the inputs are one dimensional compared to batch inputs
		dictEval = { var_name : nn_val(fus_xu[...,xu_index])  \
						for (var_name, (nn_val, xu_index)) in self._unknown_terms.items()}

		# Estimate the midpoint value coefficient
		if self.midpoint is not None:
			# fus_xtu = jnp.hstack((time_step,x,u)) if u is not None else jnp.hstack((time_step,x))
			midpoint_coeff = self.midpoint(fus_xu)
		else:
			midpoint_coeff = None

		# If none about the dynamics are known
		if self._known_dynamics is None:
			(fName, fValue), = dictEval.items()
			return fValue, midpoint_coeff, dictEval

		# If extra arguments are given then the known function should take as input the extra arguments
		if extra_args is not None:
			return self._known_dynamics(x, u, extra_args=extra_args, **dictEval), midpoint_coeff, dictEval
		else: # If no extra arguments are given then don't inclue it in the required arguments of _known_dynamics
			return self._known_dynamics(x, u, **dictEval), midpoint_coeff, dictEval

	def vector_field(self, x, u=None, extra_args=None):
		""" This function defines the underlying dynamics of the system and encodes the known terms. 
			Specifically, we assume that \dot{x} = f(x, u, g_1(x,u), g_2(x,u), ...) where f is known but
			the functions g_1(.), g_2(.) of the susbset of variables and controls are potentially
			unknown. We encode these unknown functions as a set of small size feed forward neural networks 
			for which we develop a custom forward and back propagation to train their parameters.

			This function only returns the estimate value of the vector field.
			THIS FUNCTION ASSUMES THAT THE INPUTS ARE BATCHES, I.E., TWO DIMENSIONAL ARRAYS

			:params x : The current state vector
			:params u : The current control value
			:param extra_args	: State dependent extra parameters used in side information and constraints
		"""
		vfield, _, _ = self.vector_field_and_unknown(x, u, extra_args)
		return vfield


	def __call__(self, x : jnp.ndarray, u : Optional[jnp.ndarray] = None, extra_args : Optional[jnp.ndarray] = None):
		""" This function provides the vector field, all the unknown terms in the parameterization of the vector field
			and the midpoint value if it was also parameterized
			:param x : Current step of the system
			:param u : The control input value
			:param extra_args	: State dependent extra parameters used in side information
		"""
		return self.vector_field_and_unknown(x, u, extra_args) # vField, midpoint_coeff, unknown_terms


def build_taylanets(rng_key, nstate, ncontrol, time_step, baseline_params, nn_params = {}, optim = None, 
					model_name = 'tayla', known_dynamics=None, pen_constr={}, batch_size=1, 
					extra_args_init=None, normalize=False):
	""" This function builds a set of functions to train and and estimate future state values of unknown dynamical systems with side information about 
		the dynamics. Specifically, it returns a function to estimate next state, to compute the loss function, and a function to update each network parameters.
		:param rng_key						: A key for random initialization of the parameters of the neural networks
		:param nstate 						: The number of state of the system
		:param ncontrol 					: The number of control inputs of the system
		:param time_step 					: The fixed time step for the ode integrator to use --> TODO : Add varying step size to enable data points not sampled regurlarly 
		:param baseline_params 				: This is a dictionary providing the baseline ODESolver and teh corresponding parameters to use during the training process.
												The dictionary must contains at least the keys 'name', and possibly the keys 'order', 'midpoint' , 'remainder' and other parameters
												The key 'name' specifies the ODESolver to use ('rk4', 'euler', 'base', 'taylor', 'tayla', 'odeint')
													<rk4> 		implements Runge-Kutta of order 4 with fixed time step
													<euler> 	implements a first order euler scheme 
													<base> 		implements a single neural network predicting next state given current state and current control signal
													<taylor> 	implements a prediction using a fixed time-step taylor expansion of order n
													<tayla> 	implements a prediction using a fixed time-step taylor lagrange expansion of order n with learned remainder/midpoint value
													<odeint> 	implements a prediction using the adaptive time-step integration scheme with adjoint method provided by jax
												The key 'order' specifies the order of the taylor expansion if the method involves taylor expansion
												The key 'midpoint' (only for <tayla>) specifies that we learn the midpoint value instead of the remainder and provide the neural nets params for the quantity
												The key 'remainder' (only for <tayla>) specifies that we learn the remainder term instead of the midpoint and provide the neural nets params for the quantity
		:paramn nn_params 					: Dictionary containing the parameters of the NN of each unknown terms
												nn_params = {'g_1' : {'input_index' : , 'output_sizes' : , 'w_init' : , 'b_init' : , 'with_bias' : , 'activation' :, 'activate_final' :}, ...}.
												The keys of this dictionary should matches the arguments of the function 'known_dynamics' below.
		:param optim 						: The optimizer for the update of the neural networks parameters 
		:param model_name 					: A name for the model of interest. This must be unique as it is useful to load and save parameters of the model
		:param known_dynamics				: \dot{x} = known_dynamics(x, u, g_1(x,u), g_2(x,u), ...) where 'known_dynamics' is the known part but the functions g_1(.), g_2(.) of the susbset of variables 
												and controls are potentially unknown. 'input_index' specifies the input dependency when merging the arrays x and u.
												The extra arguments name g_1, g_2 pf this function should match the keys of the dictionary of 'nn_params' above.
		:param pen_constr					: The penalty coefficient parameters to take into account for the constraints on the midpoint value ( or the remainder if any)
		:param batch_size 					: The batch size for initial compilation of the haiku pure function
		:param extra_args_init 				: The size of the last dimension of the extra argument to use in known_dynamics for intialization of the VectorField module
		:param normalize					: Loss function is divided by the batch size and the number of states (similar reasoning applies for constraints on the midpoint)
	"""
	######## First, extract all the baselines parameters
	assert time_step > 0 , 'The time step {} must be greater than 0'.format(time_step)
	# Extract ode solver name
	print(baseline_params)
	odesolver_name = baseline_params.get('name', None)
	assert odesolver_name is not None, 'The name of the odesolver to use must be specified in baseline_params'

	# Extract the order of the expansion
	taylor_order = baseline_params.get('order', None)

	# Extract (specifically for tayla) if we learn the full remainder or just the midpoint value
	midpoint_nn = baseline_params.get('midpoint', None)
	remainder_nn = baseline_params.get('remainder', None)
	assert midpoint_nn is None or remainder_nn is None, 'midpoint or remainder cannot be both specified at the same time'

	# Make a copy of nn_params
	nn_params_copy = nn_params.copy()
	assert 'midpoint' not in nn_params_copy, 'The midpoint neural network parameters should not be specified in nn_params'

	# Some more parsing of baseline_params dict values
	is_midpoint = None
	if odesolver_name == 'tayla' and (midpoint_nn is not None or remainder_nn is not None):
		assert taylor_order is not None and taylor_order >= 1, 'midpoint/remainder neural net is only valid for taylor lagrange nets with taylor_order >= 1'
		is_midpoint = midpoint_nn is not None
		nn_params_copy['midpoint'] = midpoint_nn if is_midpoint else remainder_nn
	assert not (is_midpoint is None and odesolver_name == 'tayla'), 'Midpoint or Remainder should only be specified for tayla ode integration scheme' 


	######### Build the functions to compute outputs of the neural networks
	def vectorfield_and_midpoint_fn(x: jnp.ndarray, u : Optional[jnp.ndarray] = None, extra_args : Optional[jnp.ndarray] = None):
		""" This function estimates the vector field and the remainder/midpoint value by calling the neural networks approximator
			:param x 				: The current state of the system
			:param u 				: The current control of the system
			:param extra_args		: State dependent extra parameters used in side information and constraints	
		"""
		# assert (u is None or x.shape[0] == u.shape[0]), 'The (batch size of u) should be equal to (batch size of x)'
		objNN = VectorFieldNets(nstate, ncontrol, nn_params=nn_params_copy, known_dynamics=known_dynamics, name=model_name)
		return objNN(x, u, extra_args)

	# Initialize this function
	dummy_x_init = jax.numpy.zeros((batch_size, nstate))
	dummy_u_init = None if ncontrol == 0 else jax.numpy.zeros((batch_size,ncontrol))
	dummy_args = None if extra_args_init is None else jax.numpy.zeros((batch_size, extra_args_init))

	# Build the prediction function
	pred_fn_pure = hk.without_apply_rng(hk.transform(vectorfield_and_midpoint_fn))

	# Initialize the parameters for the prediction function
	params_init = pred_fn_pure.init(rng_key, x=dummy_x_init, u=dummy_u_init, extra_args=dummy_args)
	vectorfield_and_midpoint = pred_fn_pure.apply

	############ Define each method for ODE solve
	if odesolver_name == 'euler': # Euler integration scheme
		def pred_xnext(params : hk.Params, state : jnp.ndarray, u : Optional[jnp.ndarray] = None, extra_args : Optional[jnp.ndarray] = None):
			vField, mPoint, unknown_terms = vectorfield_and_midpoint(params, state, u, extra_args)
			return state + time_step * vField, (vField, mPoint, unknown_terms)
	elif odesolver_name == 'rk4': # RK4 integration scheme
		dt_over_2 = 0.5 * time_step
		dt_over_6 = time_step / 6.0
		def pred_xnext(params : hk.Params, state : jnp.ndarray, u : Optional[jnp.ndarray] = None, extra_args : Optional[jnp.ndarray] = None):
			k1, mPoint, unknown_terms = vectorfield_and_midpoint(params, state, u, extra_args)
			k2, _, _ = vectorfield_and_midpoint(params, state + dt_over_2 * k1, u, extra_args)
			k3, _, _ = vectorfield_and_midpoint(params, state + dt_over_2 * k2, u, extra_args)
			k4, _, _ = vectorfield_and_midpoint(params, state + time_step * k3, u, extra_args)
			return state + dt_over_6 * (k1 + 2*k2 + 2*k3 + k4), (k1, mPoint, unknown_terms)
	elif odesolver_name == 'taylor': # Taylor expansion of fixed order without remainder estimation
		assert taylor_order is not None, 'A taylor expansion order must be specified'
		# Compute the factorial term of each taylor order
		inv_m_fact = jet.fact(jnp.array([i+1 for i in range(taylor_order+1)]))
		inv_m_fact = 1.0 / inv_m_fact
		# Compute dt^2/2
		dt_square_over_2 = time_step*time_step*0.5
		# Compute the power series of the time step
		pow_timestep = [time_step]
		for _ in range(taylor_order):
			pow_timestep.append(pow_timestep[-1] * time_step)
		pow_timestep = jnp.array(pow_timestep) * inv_m_fact
		# Divide the power series of the time step by factorial of the derivative order --> Do some reshaping for broadcasting issue
		pow_timestep = pow_timestep.reshape((-1,1,1))
		# Define the predictor function
		def pred_xnext(params : hk.Params, state : jnp.ndarray, u : Optional[jnp.ndarray] = None, extra_args : Optional[jnp.ndarray] = None):
			vField, mPoint, unknown_terms = vectorfield_and_midpoint(params, state, u, extra_args)
			if taylor_order == 0:
				return state + time_step * vField, (vField, mPoint, unknown_terms)
			elif taylor_order == 1:
				temp_fn = lambda x_state : vectorfield_and_midpoint(params, x_state, u, extra_args)[0]
				return state + time_step * vField + dt_square_over_2 * jax.jvp(temp_fn, (state,), (vField,))[1], (vField, mPoint, unknown_terms)
			else:
				# Define the vector field function
				temp_fn = lambda x_state : vectorfield_and_midpoint(params, x_state, u, extra_args)[0]

				# Compute the higher order derivatives
				m_expansion = taylor_order_n(state, temp_fn, taylor_order, y0=vField)
				return state + jnp.sum(pow_timestep * m_expansion, axis=0), (vField, mPoint, unknown_terms)
	elif odesolver_name == 'odeint': # Adaptative ODE solver from JAX -> RK4 + Dormand Price
		def pred_xnext(params : hk.Params, state : jnp.ndarray,  u : Optional[jnp.ndarray] = None, extra_args : Optional[jnp.ndarray] = None):
			# Little trict to use vectorfield_and_midpoint since the vector field is already vectorized
			temp_fn = lambda x_state, t=0 : vectorfield_and_midpoint(params, x_state.reshape(1,-1), u, extra_args)[0][0]
			@jax.vmap
			def op(m_state):
				time_steps = jnp.array([0.0, time_step])
				next_state = odeint(temp_fn, m_state, t=time_steps)
				return next_state[-1,:]
			return op(state), (None, None, None)
	elif odesolver_name == 'tayla':
		assert taylor_order is not None, 'A taylor expansion order must be specified'
		# Compute the factorial term of each taylor order
		inv_m_fact = jet.fact(jnp.array([i+1 for i in range(taylor_order+1)]))
		inv_m_fact = 1.0 / inv_m_fact
		# Compute dt^2/2
		dt_square_over_2 = time_step*time_step*0.5
		# Compute the power series of the time step
		pow_timestep = [time_step]
		for _ in range(taylor_order):
			pow_timestep.append(pow_timestep[-1] * time_step)
		pow_timestep = jnp.array(pow_timestep) * inv_m_fact
		rem_coeff = pow_timestep[-1]
		# Divide the power series of the time step by factorial of the derivative order --> Do some reshaping for broadcasting issue
		pow_timestep = pow_timestep[:-1].reshape((-1,1,1))
		# Define the predictor function
		def pred_xnext(params : hk.Params, state : jnp.ndarray, u : Optional[jnp.ndarray] = None, extra_args : Optional[jnp.ndarray] = None):
			temp_fn = lambda x_state : vectorfield_and_midpoint(params, x_state, u, extra_args)[0]
			vField, mPoint, unknown_terms = vectorfield_and_midpoint(params, state, u, extra_args)
			if taylor_order == 1: # The first order is faster and easier to implement with jvp
				next_state = state + time_step * vField
				if is_midpoint:
					actual_midpoint = state + mPoint * vField
					return next_state + dt_square_over_2 * jax.jvp(temp_fn, (actual_midpoint,), (temp_fn(actual_midpoint),))[1], (vField, actual_midpoint, unknown_terms)
				else:
					return next_state + dt_square_over_2 * mPoint, (vField, mPoint, unknown_terms)
			else:
				if taylor_order == 2:
					next_state = state + time_step * vField + dt_square_over_2 * jax.jvp(temp_fn, (state,), (vField,))[1]
				else:
					# Compute the higher order derivatives
					m_expansion = taylor_order_n(state, temp_fn, taylor_order-1, y0=vField)	
					next_state = state + jnp.sum(pow_timestep * m_expansion, axis=0)
				
				if is_midpoint:
					actual_midpoint = state + mPoint * vField
					rem_term = der_order_n(actual_midpoint, temp_fn, taylor_order)	
					return  next_state + rem_coeff * rem_term, (vField, actual_midpoint, unknown_terms)
				else:
					return  next_state + rem_coeff * mPoint, (vField, mPoint, unknown_terms)
	else:
		raise Exception('{} not implemented yet !'.format(odesolver_name))

	############# Define the penalty terms for the midpoint constraint
	beta_ineq, pen_ineq_shape, m_pen_ineq_k, m_lagr_ineq_k = None, None, None, None
	if is_midpoint is not None and is_midpoint: 
		assert type(pen_constr) == dict, 'Penalty constraints should be a dictionary'
		pen_ineq_init, beta_ineq = pen_constr['pen_ineq_init'], pen_constr['beta_ineq']
		pen_ineq_shape = 2 * nstate # The lower bound and the upper bound inequality constraints
		total_constr = pen_constr['coloc_set_size']
		# Parameters for the constant penalty coefficient of the inequality constraints
		m_pen_ineq_k = None if total_constr <= 0 else pen_ineq_init
		# Parameters for the lagrangian multiplier of the inequality constraints
		m_lagr_ineq_k = None if total_constr <= 0 else jnp.zeros((total_constr, pen_ineq_shape))

	############ Define the loss function
	# Then define a function to compute the loss function needed to train the model
	def loss_fun(	params 			: hk.Params, 
					xnext 			: jnp.ndarray, 
					x 				: jnp.ndarray, 
					u 				: Optional[jnp.ndarray] = None, 
					extra_args 		: Optional[jnp.ndarray] = None, 
					pen_ineq_k 		: Optional[float] = None, 
					lagr_ineq_k 	: Optional[jnp.ndarray] = None,
					coloc_points 	: Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] =(None, None, None)
				):
		""" Compute the loss function given the current parameters of the custom neural network
			:param params 		: Weights of all the neural networks
			:param xnext 		: The target next state of the system
			:param x 			: The state for which to estimate the next state value
			:param u 			: The control signal applied at each state x
			:param extra_args	: State dependent extra parameters used in side information and constraints
			:param pen_ineq_k 	: Penalty coefficient for the inequality constraints Phi(x,u,...) <= 0
			:param lagr_ineq_k 	: Lagrange multiplier for the inequality constraints
			:param coloc_points : Point used in order to enforce the constraints on unlabbelled data
		"""
		# assert u is None or x.shape[0]==u.shape[0], 'The (batch size of u) should be equal to (batch size of x)'
		assert  (len(x.shape) == len(xnext.shape)-1) and (u is None or (u.shape[0] == xnext.shape[0])), \
			'Mismatch ! xnext and u should have one more dimension than x of size roolout'

		# Scan function for rolling out the dynamics
		def rollout(carry, extra):
			""" Rollout the computation of the next state
			"""
			curr_x, params = carry
			curr_u, true_nextx = extra

			# Predict the next state
			next_x, (_, midpoint_value, _) = pred_xnext(params, curr_x, curr_u, extra_args)

			# Measure the mean squared difference
			meanSquredDiff = jnp.sum(jnp.square(next_x - true_nextx)) / (x.shape[0]*(1 if not normalize else x.shape[-1]))

			# Compute the total loss
			totalLoss =  meanSquredDiff

			########################################################################################################
			# Compute the cost associated to the constraints
			cost_midpoint = 0.0
			if m_pen_ineq_k is not None: # Check if midpoint computation is required with constraints on the midpoint
				ineqTerms = midpoint_constraints(midpoint_value, curr_x, next_x)
				cost_midpoint += jnp.sum(jnp.where(ineqTerms > 0, 1.0, 0.0) * jnp.square(ineqTerms)) /  (ineqTerms.shape[0]*(1 if not normalize else ineqTerms.shape[-1]))
				totalLoss += pen_ineq_k * cost_midpoint
			########################################################################################################

			return (next_x, params), jnp.array([totalLoss, meanSquredDiff, cost_midpoint])

		# Rollout and compute the rolled out error term
		_, m_res = jax.lax.scan(rollout, (x, params), (u, xnext))

		# Compute the loss associated to the collocation points 
		coloc_cost = 0.0
		cTerm_ineq = 0.0
		if m_pen_ineq_k is not None:
			assert coloc_points[0] is not None, 'At least the state and the time step in the colocation data sets must be non None'
			next_x, (_, midpoint_value, _) = pred_xnext(params, *coloc_points)
			ineqTerms = midpoint_constraints(midpoint_value, coloc_points[0], next_x)
			# TODO : The lagrangian term associated to this constraint should be included in the test ineqTerms > 0
			cTerm_ineq = jnp.sum(jnp.where(ineqTerms > 0, 1.0, 0.0) * jnp.square(ineqTerms)) /  (ineqTerms.shape[0]*(1 if not normalize else ineqTerms.shape[-1]))
			cTerm_lagineq = jnp.sum(lagr_ineq_k * ineqTerms) / (ineqTerms.shape[0]*(1 if not normalize else ineqTerms.shape[-1]))
			coloc_cost += pen_ineq_k * cTerm_ineq + cTerm_lagineq

		# Total cost function
		m_total_cost = jnp.mean(m_res[:,0]) + coloc_cost
		# Return the composite
		return  m_total_cost, (m_res, jnp.array([m_total_cost, coloc_cost, cTerm_ineq]))

	############## Define the gradient of the loss function
	grad_fun = jax.grad(loss_fun, has_aux=True)

	############## Define the update function for the neural networks weights
	# Define the update step
	def update(	params 			: hk.Params, 
				opt_state 		: optax.OptState, 
				xnext 			: jnp.ndarray, 
				x 				: jnp.ndarray, 
				u 				: Optional[jnp.ndarray] = None, 
				extra_args 		: Optional[jnp.ndarray] = None, 
				pen_ineq_k 		: Optional[float] = None, 
				lagr_ineq_k 	: Optional[jnp.ndarray] = None,
				coloc_points 	: Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] =(None, None, None)
			) -> Tuple[hk.Params, optax.OptState, Tuple[jnp.ndarray, jnp.ndarray]]:
		"""Update the parameters of the neural netowkrs via one of the gradient descent optimizer
			:param params 		: The current weight of the neural network
			:param opt_state 	: The current state of the optimizer
			:param xnext 		: The target next state of the system
			:param x 			: The state for which to estimate the next state value
			:param u 			: The control signal applied at each state x
			:param extra_args	: State dependent extra parameters used in side information and constraints
			:param pen_ineq_k 	: Penalty coefficient for the inequality constraints Phi(x,u,...) <= 0
			:param lagr_ineq_k 	: Lagrange multiplier for the inequality constraints
			:param coloc_points : Point used in order to enforce the constraints on unlabbelled data
		"""
		grads, m_aux = grad_fun(params, xnext, x, u, extra_args, pen_ineq_k, lagr_ineq_k, coloc_points)
		updates, opt_state = optim.update(grads, opt_state, params)
		params = optax.apply_updates(params, updates)
		return params, opt_state, m_aux


	############## Define the update function for the lagrangian parameters
	# Define the update step for the lagrangier multipliers term
	def update_lagrange(params, 
						pen_ineq_k 		: Optional[float] = None, 
						lagr_ineq_k 	: Optional[jnp.ndarray] = None,
						coloc_points 	: Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] =(None, None, None)):
		""" This function defines the update rule for the lagrangian multiplier for satisfaction of constraints
			:param params 		: The current weight of the neural network
			:param pen_ineq_k 	: Penalty coefficient for the inequality constraints Phi(x,u,...) <= 0
			:param lagr_ineq_k 	: Lagranfier multiplier for the inequality constraints
			:param coloc_points : Point used in order to enforce the constraints on unlabbelled data
		"""
		# Sanity check
		if m_pen_ineq_k is None or beta_ineq is None:
			return None, None

		# In case the constraints are enabled
		assert coloc_points[0] is not None, 'At least the state and the time step in the colocation data sets must be non None'

		# Compute the constraints at the colocations points
		next_x, (_, midpoint_value, _) = pred_xnext(params, *coloc_points)
		ineqTerms = midpoint_constraints(midpoint_value, coloc_points[0], next_x)

		# Update lagrangian term for inequality constraints
		n_lagr_ineq_k = jnp.maximum(lagr_ineq_k + 2 * pen_ineq_k * ineqTerms, 0)
		return pen_ineq_k * beta_ineq, n_lagr_ineq_k

	return (params_init, m_pen_ineq_k, m_lagr_ineq_k) , pred_xnext, loss_fun, update, update_lagrange