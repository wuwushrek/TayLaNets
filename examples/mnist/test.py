"""
Neural ODEs on MNIST with no downsampling before ODE, implemented with Haiku.
"""
import argparse
import collections
import os
import pickle
import time
from math import prod

import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from jax import lax
from jax.config import config
from jax.experimental import optimizers
from jax.experimental.jet import jet
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten

from lib.ode import odeint, odeint_aux_one, odeint_sepaux, odeint_grid, odeint_grid_sepaux_one, odeint_grid_aux

float64 = False
config.update("jax_enable_x64", float64)

from tqdm.auto import tqdm


# some primitive functions
def sigmoid(z):
	"""
	Numerically stable sigmoid.
	"""
	return 1/(1 + jnp.exp(-z))


def softmax_cross_entropy(logits, labels):
	"""
	Cross-entropy loss applied to softmax.
	"""
	one_hot = hk.one_hot(labels, logits.shape[-1])
	return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)

# set up modules
class Flatten(hk.Module):
	"""
	Flatten all dimensions except batch dimension.
	"""

	def __init__(self):
		super(Flatten, self).__init__()

	def __call__(self, x):
		return jnp.reshape(x, (x.shape[0], -1))

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


class PreODE(hk.Module):
	"""
	Module applied before the ODE layer.
	"""

	def __init__(self):
		super(PreODE, self).__init__()
		if float64:
			self.model = hk.Sequential([
				lambda x: x.astype(jnp.float64) / 255.,
				Flatten()
			])
		else:
			self.model = hk.Sequential([
				lambda x: x.astype(jnp.float32) / 255.,
				Flatten()
			])

	def __call__(self, x):
		return self.model(x)


class MLPDynamics(hk.Module):
	"""
	Dynamics for ODE as an MLP.
	"""

	def __init__(self, input_shape):
		super(MLPDynamics, self).__init__()
		self.input_shape = input_shape
		self.dim = prod(input_shape[1:])
		self.hidden_dim = 100
		self.lin1 = hk.Linear(self.hidden_dim)
		self.lin2 = hk.Linear(self.dim)

	def __call__(self, x, t):
		# vmapping means x will be a single batch element, so need to expand dims at 0
		x = jnp.reshape(x, (-1, self.dim))

		out = sigmoid(x)
		tt = jnp.ones_like(x[:, :1]) * t
		t_out = jnp.concatenate([tt, out], axis=-1)
		out = self.lin1(t_out)

		out = sigmoid(out)
		tt = jnp.ones_like(out[:, :1]) * t
		t_out = jnp.concatenate([tt, out], axis=-1)
		out = self.lin2(t_out)

		return out

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


def _acc_fn(logits, labels):
	"""
	Classification accuracy of the model.
	"""
	predicted_class = jnp.argmax(logits, axis=1)
	return jnp.mean(predicted_class == labels)


def _loss_fn(logits, labels):
	return jnp.mean(softmax_cross_entropy(logits, labels))


def initialization_data(input_shape, ode_shape):
	"""
	Data for initializing the modules.
	"""
	ode_shape = (1, ) + ode_shape[1:]
	ode_dim = prod(ode_shape)
	data = {
		"pre_ode": jnp.zeros(input_shape),
		"ode": (jnp.zeros(ode_dim), 0.),
		"post_ode": jnp.zeros(ode_dim)
	}
	return data


def init_model():
	"""
	Instantiates transformed submodules of model and their parameters.
	"""
	ts = jnp.array([0., 1.])

	input_shape = (1, 28, 28, 1)
	ode_shape = (-1, 28, 28, 1)

	initialization_data_ = initialization_data(input_shape, ode_shape)

	pre_ode = hk.without_apply_rng(hk.transform(wrap_module(PreODE)))
	pre_ode_params = pre_ode.init(rng, initialization_data_["pre_ode"])
	pre_ode_fn = pre_ode.apply

	dynamics = hk.without_apply_rng(hk.transform(wrap_module(MLPDynamics, ode_shape)))
	dynamics_params = dynamics.init(rng, *initialization_data_["ode"])
	dynamics_wrap = lambda x, t, params: dynamics.apply(params, x, t)


# # function assumes ts = [0, 1]
# def init_model(rng, taylor_order, number_step, batch_size=1, optim=None, midpoint_layers=(12,12)):
# 	# Integration time -> initial time is 0 and end time is 1
# 	ts = jnp.array([0., 1.])

# 	# Discretize the integration time using the given number of step
# 	time_step = (ts[1]-ts[0]) / number_step
# 	time_indexes = jnp.array([ time_step * (i+1) for i in range(number_step)])

# 	# MNIST problem size and dummy random intiial values
# 	image_shape = (28, 28, 1)
# 	ode_shape = prod(image_shape)
# 	midpoint_shape = ode_shape + 1
# 	pre_ode_init = jnp.zeros((batch_size, *image_shape))
# 	ode_init, t_init = jnp.zeros((batch_size, ode_shape)), 0.0
# 	midpoint_init = jnp.zeros((batch_size, midpoint_shape))
# 	post_ode_init = jnp.zeros((batch_size, ode_shape))

# 	# Build the pre module
# 	pre_ode = hk.without_apply_rng(hk.transform(wrap_module(PreODE)))
# 	pre_ode_params = pre_ode.init(rng, pre_ode_init)
# 	pre_ode_fn = pre_ode.apply

# 	# Build the ODE function module
# 	dynamics = hk.without_apply_rng(hk.transform(wrap_module(MLPDynamics, ode_shape)))
# 	dynamics_params = dynamics.init(rng, ode_init, t_init)
# 	dynamics_wrap = dynamics.apply

# 	# Build the Midpoint function module
# 	midpoint = hk.without_apply_rng(hk.transform(wrap_module(Midpoint, midpoint_shape, midpoint_layers)))
# 	midpoint_params = midpoint.init(rng, midpoint_init)
# 	midpoint_wrap = midpoint.apply

# 	# Post processing function module
# 	post_ode = hk.without_apply_rng(hk.transform(wrap_module(PostODE)))
# 	post_ode_params = post_ode.init(rng, post_ode_init)
# 	post_ode_fn = post_ode.apply

# 	# Regroup the parameters of this entire module
# 	m_params = (pre_ode_params, dynamics_params, midpoint_params, post_ode_params)

# 	# Define the ODE prediction function
# 	inv_m_fact = jet.fact(jnp.array([i+1 for i in range(taylor_order+1)]))
# 	inv_m_fact = 1.0 / inv_m_fact
# 	dt_square_over_2 = time_step*time_step*0.5
# 	pow_timestep = [time_step]
# 	for _ in range(taylor_order):
# 		pow_timestep.append(pow_timestep[-1] * time_step)
# 	pow_timestep = jnp.array(pow_timestep) * inv_m_fact
# 	rem_coeff = pow_timestep[-1]
# 	# Divide the power series of the time step by factorial of the derivative order --> Do some reshaping for broadcasting issue
# 	pow_timestep = pow_timestep[:-1].reshape((-1,1,1))
# 	def pred_xnext(params : Tuple[hk.Params, hk.Params], state : jnp.ndarray, t : float) -> jnp.ndarray:
# 		""" Predict the next using our Taylor-Lagrange expansion with learned remainder"""
# 		params_dyn, params_mid = params

# 		# Define the vector field of the ODE given a pre-process input
# 		def vector_field(state_val : jnp.ndarray) -> jnp.ndarray:
# 			""" Function computing the vector field"""
# 			xstate, curr_t = state_val[...,:-1], state_val[...,-1:]
# 			curr_dt = jnp.ones_like(curr_t)
# 			vField = dynamics_wrap(params_dyn, xstate, curr_t)
# 			return jnp.concatenate((vField, curr_dt), axis=-1)

# 		# Merge the state value and time
# 		state_t = jnp.concatenate((state, jnp.ones_like(state[...,:1])*t), axis=-1)

# 		# Compute the vector field as it is used recursively in jet and the midpoint value
# 		vField = vector_field(state_t)

# 		# Compute the midpoint coefficient 
# 		midpoint_val = state_t + midpoint_wrap(params_mid, state_t) * vField

# 		if taylor_order == 0: # Simple classical single layer DNNN
# 			next_state = state_t + time_step * vector_field(midpoint_val)

# 		elif taylor_order == 1: # The first order is faster and easier to implement with jvp
# 			next_state = state_t + time_step * vField \
# 							+ dt_square_over_2 * jax.jvp(vector_field, (midpoint_val,), (vector_field(midpoint_val),))[1]

# 		elif taylor_order == 2: # The second order can be easily and fastly encode too
# 			next_state = state_t + time_step * vField + dt_square_over_2 * jax.jvp(vector_field, (state_t,), (vField,))[1]
# 			# Add the remainder at the midpoint
# 			next_state += rem_coeff * der_order_n(midpoint_val, vector_field, taylor_order)

# 		else:
# 			# Compiute higher order derivatives
# 			m_expansion = taylor_order_n(state_t, vector_field, taylor_order-1, y0=vField)	
# 			next_state = state_t + jnp.sum(pow_timestep * m_expansion, axis=0)
# 			# Remainder term
# 			next_state += rem_coeff * der_order_n(midpoint_val, vector_field, taylor_order)
# 		return next_state[...,:-1]

# 	# Define the loss function
# 	@jax.jit
# 	def loss_fun(params, _images, _labels):
# 		""" Compute the loss function of the prediction method
# 		"""
# 		# Load the different params
# 		(_pre_ode_params, _dynamics_params, _midpoint_params, _post_ode_params) = params
# 		out_pre_ode = pre_ode_fn(_pre_ode_params, _images)

# 		# Build the iteration loop for the ode solver when step size is not 1
# 		def rollout(carry, extra):
# 			next_x = pred_xnext((_dynamics_params, _midpoint_params), carry, extra)
# 			return next_x, None

# 		# Loop over the grid time step
# 		out_ode, _ = jax.lax.scan(rollout, out_pre_ode, time_indexes)

# 		# Post process
# 		out_ode = post_ode_fn(_post_ode_params, out_ode)

# 		# Compute the loss function
# 		return _loss_fn(out_ode, _labels)

# 	# Define the update function
# 	grad_fun = jax.grad(loss_fun, has_aux=False)

# 	@jax.jit
# 	def update(params, opt_state, _images, _labels):
# 		grads = grad_fun(params, _images, _labels)
# 		updates, opt_state = optim.update(grads, opt_state, params)
# 		params = optax.apply_updates(params, updates)
# 		return params, opt_state

# 	return m_params, loss_fun, update, post_ode_init.dtype


# def init_data(train_batch_size, test_batch_size, seed_number=0, shuffle=1000):
# 	"""
# 	Initialize data from tensorflow dataset
# 	"""
# 	# Import and cache the file in the current directory
# 	(ds_train,), ds_info = tfds.load('mnist',
# 									 data_dir='./tensorflow_data/',
# 									 split=['train'],
# 									 shuffle_files=True,
# 									 as_supervised=True,
# 									 with_info=True,
# 									 read_config=tfds.ReadConfig(shuffle_seed=seed_number))

# 	num_train = ds_info.splits['train'].num_examples

# 	assert num_train % train_batch_size == 0
# 	num_train_batches = num_train // train_batch_size

# 	assert num_train % test_batch_size == 0
# 	num_test_batches = num_train // test_batch_size

# 	# Make the data set loopable mainly for testing loss evalutaion
# 	ds_train = ds_train.cache()
# 	ds_train = ds_train.repeat()
# 	ds_train = ds_train.shuffle(shuffle, seed=seed_number)
# 	ds_train, ds_train_eval = ds_train.batch(train_batch_size), ds_train.batch(test_batch_size)
# 	ds_train, ds_train_eval = tfds.as_numpy(ds_train), tfds.as_numpy(ds_train_eval)

# 	meta = {
# 		"num_train_batches": num_train_batches,
# 		"num_test_batches": num_test_batches
# 	}

# 	# Return iter element on the training and testing set
# 	return iter(ds_train), iter(ds_train_eval), meta

# if __name__ == "__main__":
# 	# Parse the command line argument
# 	parser = argparse.ArgumentParser('Neural ODE MNIST')
# 	parser.add_argument('--train_batch_size', type=int, default=100)
# 	parser.add_argument('--test_batch_size', type=int, default=1000)
# 	parser.add_argument('--taylor_order', type=int, default=1)
# 	parser.add_argument('--lr_init', type=float, default=1e-2)
# 	parser.add_argument('--lr_end', type=float, default=1e-2)
# 	parser.add_argument('--w_decay', type=float, default=1e-3)
# 	parser.add_argument('--grad_clip', type=float, default=1e-2)
# 	parser.add_argument('--nepochs', type=int, default=160)
# 	parser.add_argument('--test_freq', type=int, default=3000)
# 	parser.add_argument('--save_freq', type=int, default=3000)
# 	parser.add_argument('--dirname', type=str, default='neur_train')
# 	parser.add_argument('--seed', type=int, default=0)
# 	# parser.add_argument('--no_count_nfe', action="store_true")
# 	parser.add_argument('--num_steps', type=int, default=2)
# 	args = parser.parse_args()

# 	# Initialize the MNIST data set
# 	ds_train, ds_train_eval, meta = init_data(args.train_batch_size, args.test_batch_size)
# 	train_batch_size = args.train_batch_size
# 	test_batch_size = args.test_batch_size

# 	############## Build the optimizer ##################
# 	# Customize the gradient descent algorithm
# 	chain_list = [optax.scale_by_adam()]

# 	# Add weight decay if enable
# 	decay_weight = args.w_decay
# 	if decay_weight > 0.0:
# 		chain_list.append(optax.add_decayed_weights(decay_weight))

# 	# Add scheduler for learning rate decrease --> Linear decrease given bt learning_rate_init and learning_rate_end
# 	m_schedule = optax.linear_schedule(-args.lr_init, -args.lr_end, args.nepochs*meta['num_train_batches'])
# 	chain_list.append(optax.scale_by_schedule(m_schedule))

# 	# Add gradient clipping if enable
# 	grad_clip_coeff = args.grad_clip
# 	if grad_clip_coeff > 0.0:
# 		chain_list.append(optax.adaptive_grad_clip(clipping=grad_clip_coeff))

# 	# Build the solver finally
# 	opt = optax.chain(*chain_list)

# 	# Build the solver
# 	rng = jax.random.PRNGKey(args.seed)
# 	midpoint_hidden_layer_size = (24,24)
# 	m_params, loss_fun, update, m_dtype = init_model(rng, args.taylor_order, args.num_steps, batch_size=train_batch_size, optim=opt, midpoint_layers=midpoint_hidden_layer_size)

# 	# Initialize the state of the optimizer
# 	opt_state = opt.init(m_params)

# 	# Save the number of iteration
# 	itr_count = 0

# 	# Save the optimal loss_value obtained
# 	opt_params_dict = None
# 	opt_loss = None
# 	total_compute_time = 0.0
# 	loss_evol = list()
# 	time_evol = list()

# 	for epoch in tqdm(range(args.nepochs)):
# 		for i in tqdm(range(meta['num_train_batches']), leave=False):
# 			# Get the next batch of images
# 			_images, _labels = next(ds_train)

# 			# Increment the iteration count
# 			itr_count += 1

# 			# Convert the image into float 
# 			_images = _images.astype(m_dtype)

# 			# Update the weight of each neural networks and evaluate the compute time
# 			update_start = time.time()
# 			m_params, opt_state = update(m_params, opt_state, _images, _labels)
# 			tree_flatten(opt_state)[0][0].block_until_ready()
# 			update_end = time.time() - update_start
# 			if itr_count >= 5: # Begining jit time
# 				total_compute_time += update_end

# 			# Check if it is time to evaluate the function
# 			if itr_count % args.test_freq == 0:
# 				# Compute the loss function over the entire training set
# 				loss_values = list()
# 				for _ in range(meta['num_test_batches']):
# 					_images, _labels = next(ds_train_eval)
# 					_images = _images.astype(m_dtype)
# 					loss_values.append(loss_fun(m_params, _images, _labels))
# 				loss_values = jnp.mean(jnp.array(loss_values))

# 				# First time we have a value for the loss function
# 				if opt_loss is None:
# 					opt_loss = loss_values
# 					opt_params_dict = m_params

# 				# Check if we have improved the loss function
# 				if opt_loss > loss_values:
# 					opt_loss = loss_values
# 					opt_params_dict = m_params

# 				# Do some printing for result visualization
# 				print_str = '[Epoch {} x {}\t| Time {:.3f} \t| Loss {}\t| Best Loss {}]\n'.format(epoch, i, total_compute_time, loss_values, opt_loss)
# 				tqdm.write(print_str)

# 				# Save the results in a file
# 				loss_evol.append(loss_values)
# 				time_evol.append(total_compute_time)

# 				# Open saving file
# 				outfile = open(args.dirname+'.pkl', "wb")
# 				mData = {'loss_evol' : loss_evol, 'time_evol' : time_evol, 'total_compute_time' : total_compute_time,
# 							'opt_loss' : opt_loss, 'params' : opt_params_dict}
# 				pickle.dump(mData, outfile)
# 				outfile.close()