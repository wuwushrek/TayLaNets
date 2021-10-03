from jax.config import config
config.update("jax_enable_x64", True)

import yaml
import pickle

# Import JAX
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit

from taylanets.utils import SampleLog, load_data_yaml

from tqdm.auto import tqdm

from scipy.integrate import odeint as scipy_ode
import numpy as np


def system_ode(state, t=0, mu=100):
	""" Define the ordinary differential equation of the system
		:param state :	The current state of the system
		:param t :	The current time instant
	"""
	x, y = state[...,0], state[...,1] 
	xdot = y
	ydot = mu * (1 - np.square(x)) * y - x
	return np.hstack((xdot,ydot))


def numeric_solution(fun_ode, state_init, time_step, traj_length, n_rollout, merge_traj=True):
	""" Solve the differential equation via sicpy solve (approximative solution)
		:param state_init :	Set of initial state of the system
		:param time_step :	The time step of the integration method
		:param traj_length :	The length of the trajectory
		:param n_rollout : The rollout length 
		:param merge_traj :	Specify if the trajectories are merged to form a two dimensional array
	"""
	t_indexes = np.array([i * time_step for i in range(traj_length+n_rollout)])
	res_state = []
	res_rollout = [ [] for r in range(n_rollout)]
	for i in tqdm(range(state_init.shape[0])):
		x_init = state_init[0,:]
		x_traj = scipy_ode(fun_ode, x_init, t_indexes, full_output=False)
		if merge_traj:
			res_state.extend(x_traj[:traj_length])
		else:
			res_state.append(x_traj[:traj_length])
		for j, r in enumerate(res_rollout):
			r.extend(x_traj[(j+1):(j+1+traj_length),:])
	return res_state, res_rollout


def main_fn(path_config_file, extra_args={}):
	""" Root function in the main file.
		:param path_config_file : Path to the adequate yaml file
		:param extra_args : Extra argument from the command line
	"""
	# Load the config file
	mdata_log, (num_data_test, num_data_colocation, extra_noise_colocation) = load_data_yaml(path_config_file, extra_args)
	print(mdata_log)

	# Number of training trajectories
	num_traj_data = mdata_log.num_traj_data[-1]

	(xtrain_lb, utrain_lb) = mdata_log.xu_train_lb
	(xtrain_ub, utrain_ub) = mdata_log.xu_train_ub 
	(xtest_lb, utest_lb) = mdata_log.xu_test_lb
	(xtest_ub, utest_ub) = mdata_log.xu_test_ub 

	# Initial random seed
	m_rng = jax.random.PRNGKey(seed = mdata_log.seed_number)

	# System dimension
	nstate = 2
	ncontrol = 0

	# Set of initial states training 
	m_rng, subkey = jax.random.split(m_rng)
	m_init_train_x = jax.random.uniform(subkey, (num_traj_data, nstate), minval = jnp.array(xtrain_lb), maxval=jnp.array(xtrain_ub))

	# Generate the training trajectories
	xTrain, xNextTrain = numeric_solution(system_ode, m_init_train_x, mdata_log.time_step, mdata_log.trajectory_length, mdata_log.n_rollout)

	# Set of initial states testing 
	m_rng, subkey = jax.random.split(m_rng)
	m_init_test_x = jax.random.uniform(subkey, (num_data_test, nstate), minval = jnp.array(xtest_lb), maxval=jnp.array(xtest_ub))

	# Generate the testing trajectories
	xTest, xNextTest = numeric_solution(system_ode, m_init_test_x, mdata_log.time_step, mdata_log.trajectory_length, mdata_log.n_rollout)

	# Set of colocation poits
	coloc_points = (None, None, None)
	if num_data_colocation > 0:
		m_rng, subkey = jax.random.split(m_rng)
		m_init_coloc_x = jax.random.uniform(subkey, (num_data_colocation, nstate), minval = jnp.array(xtest_lb)-extra_noise_colocation , maxval=jnp.array(xtest_ub)+extra_noise_colocation)
		xcoloc, _ = numeric_solution(system_ode, m_init_coloc_x, mdata_log.time_step, mdata_log.trajectory_length, mdata_log.n_rollout)
		coloc_points = (xcoloc, None, None)

	# Save the log using pickle
	mSampleLog = SampleLog(xTrain, xNextTrain, None, None, mdata_log.xu_train_lb, mdata_log.xu_train_ub, xTest, xNextTest, None, None, mdata_log.xu_test_lb, mdata_log.xu_test_ub, 
					coloc_points, mdata_log.num_traj_data, mdata_log.trajectory_length, mdata_log.env_name, mdata_log.env_extra_args, mdata_log.seed_number, nstate, ncontrol, 
					mdata_log.time_step, mdata_log.n_rollout, mdata_log.data_set_file, mdata_log.others)
	mFile = open(mdata_log.data_set_file, 'wb')
	pickle.dump(mSampleLog, mFile)
	mFile.close()

	# Do some plotting for illustration
	import matplotlib.pyplot as plt
	n_traj = 1
	indx_list = [i for i in range(n_traj)]
	traj_set_list = list()
	for indx_trajectory in indx_list:
		traj_set_list.append(np.array(xTrain[indx_trajectory*mdata_log.trajectory_length:(indx_trajectory+1)*mdata_log.trajectory_length]))
	time_index = [ i * mdata_log.time_step for i in range(mdata_log.trajectory_length)]
	
	# Do some plotting
	state_label = [r'$x_{0}$', r'$x_{1}$']
	for i in range(nstate):
		plt.figure()
		for traj_set in traj_set_list:
			plt.plot(time_index, traj_set[:,i], linewidth=2)
		plt.xlabel('Time (s)')
		plt.ylabel(state_label[i])
		plt.grid(True)

	# 2D dimensional plot
	plt.figure()
	for traj_set in traj_set_list:
		plt.plot(traj_set[:,0], traj_set[:,1], linewidth=2)
	plt.xlabel(state_label[0])
	plt.ylabel(state_label[1])
	plt.grid(True)

	# plt.show()


if __name__ == "__main__":
	import time
	import argparse
	# python generate_sample.py --cfg reacher_brax/dataset_gen.yaml --output_file reacher_brax/testdata --seed 101 --disable_substep 0 --save_video 1
	# Command line argument for setting parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', required=True, type=str, help='yaml configuration file for training/testing information: see reacher_cfg1.yaml for more information')
	parser.add_argument('--output_file', type=str, default='', help='File to save the generated trajectories')
	parser.add_argument('--n_rollout', type=int, default=0, help='Number of rollout step')
	parser.add_argument('--seed', type=int, default=-1, help='The seed for the trajetcories generation')
	parser.add_argument('--trajectory_length', type=int, default=-1, help='Length of each trajectory')
	parser.add_argument('--num_data_train', nargs='+', help='A list containing the number of trajectories for each training set')
	parser.add_argument('--num_data_test', type=int, default=0, help='A list containing the number of trajectories for each testing set')
	args = parser.parse_args()
	args = parser.parse_args()
	m_config_aux = {'cfg' : args.cfg}
	if args.output_file != '':
		m_config_aux['output_file']  = args.output_file
	if args.n_rollout > 0:
		m_config_aux['n_rollout']  = args.n_rollout
	if args.seed >= 0:
		m_config_aux['seed']  = args.seed
	if args.trajectory_length > 0:
		m_config_aux['trajectory_length']  = args.trajectory_length
	if args.num_data_test > 0:
		m_config_aux['num_data_test'] = args.num_data_test
	if args.num_data_train is not None and len(args.num_data_train) > 0:
		m_config_aux['num_data_train'] = [int(val) for val in args.num_data_train]
	# print(m_config_aux)
	main_fn(args.cfg, m_config_aux)
