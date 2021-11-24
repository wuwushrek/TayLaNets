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

def system_ode(state, t=0, A=1.0, B=3.5):
	""" Define the ordinary differential equation of the system
		:param state :	The current state of the system
		:param t :	The current time instant
	"""
	x, y = state[...,0], state[...,1] 
	xdot = A + jnp.square(x) * y - B * x - x
	ydot = B * x - jnp.square(x) * y
	return jnp.hstack((xdot,ydot))


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
		x_init = state_init[i]
		x_traj = scipy_ode(fun_ode, x_init, t_indexes, full_output=False)
		if merge_traj:
			res_state.extend(x_traj[:traj_length])
		else:
			res_state.append(x_traj[:traj_length])
		for j, r in enumerate(res_rollout):
			r.extend(x_traj[(j+1):(j+1+traj_length),:])
	return res_state, res_rollout


def main_fn(path_config_file, extra_args={}, coeff_dur_test=1):
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
	test_traj_length = mdata_log.trajectory_length*coeff_dur_test
	xTest, xNextTest = numeric_solution(system_ode, m_init_test_x, mdata_log.time_step, test_traj_length, mdata_log.n_rollout)

	# Set of colocation poits
	coloc_points = (None, None, None)

	# Save the log using pickle
	mSampleLog = SampleLog(xTrain, xNextTrain, None, None, mdata_log.xu_train_lb, mdata_log.xu_train_ub, xTest, xNextTest, None, None, mdata_log.xu_test_lb, mdata_log.xu_test_ub, 
					coloc_points, mdata_log.num_traj_data, mdata_log.trajectory_length, mdata_log.env_name, mdata_log.env_extra_args, mdata_log.seed_number, nstate, ncontrol, 
					mdata_log.time_step, mdata_log.n_rollout, mdata_log.data_set_file, mdata_log.others)
	mFile = open(mdata_log.data_set_file, 'wb')
	pickle.dump(mSampleLog, mFile)
	mFile.close()

	# Do some plotting for illustration
	import matplotlib.pyplot as plt

	n_traj = num_traj_data
	indx_list = [i for i in range(n_traj)]
	traj_set_list = list()
	for indx_trajectory in indx_list:
		traj_set_list.append(np.array(xTrain[indx_trajectory*mdata_log.trajectory_length:(indx_trajectory+1)*mdata_log.trajectory_length]))

	n_traj_test = num_data_test
	indx_list_test = [i for i in range(n_traj_test)]
	trajt_set_list = list()
	for indx_trajectory in indx_list_test:
		trajt_set_list.append(np.array(xTest[indx_trajectory*(test_traj_length):(indx_trajectory+1)*(test_traj_length)]))

	time_index = [ i * mdata_log.time_step for i in range(mdata_log.trajectory_length)]
	time_index_test = [ i * mdata_log.time_step for i in range(test_traj_length)]
	
	# Do some plotting
	state_label = [r'$x_{0}$', r'$x_{1}$']
	for i in range(nstate):
		plt.figure()
		for traj_set in trajt_set_list:
			plt.plot(time_index_test, traj_set[:,i], color='r', linewidth=2)
		for traj_set in traj_set_list:
			plt.plot(time_index, traj_set[:,i], color='b', linewidth=2)
		plt.xlabel('Time (s)')
		plt.ylabel(state_label[i])
		plt.grid(True)
		plt.savefig('data/evol_x{}.svg'.format(i), dpi=300)

	# 2D dimensional plot
	plt.figure()
	for traj_set in trajt_set_list:
		plt.plot(traj_set[:,0], traj_set[:,1], color='r', linewidth=1, zorder=1)
	for traj_set in traj_set_list:
		plt.scatter(traj_set[:,0], traj_set[:,1], color='b', s=6, zorder=2)
	plt.xlabel(state_label[0])
	plt.ylabel(state_label[1])
	plt.grid(True)
	plt.savefig('data/xy_plane.svg', dpi=300)
	plt.show()


if __name__ == "__main__":
	import time
	import argparse
	# Command line argument for setting parameters
	# python generate_sample.py --dt 0.04
	parser = argparse.ArgumentParser()
	parser.add_argument('--dt', type=float, default=0.01)
	args = parser.parse_args()
	traj_duration = 20 # Duration of the trajectory
	num_trajectory  = [40] # Number of trajectories
	num_data_test = 10
	m_config_aux = {'cfg' : 'dataconfig.yaml', 'output_file' : 'data/brusselator_dt{}'.format(args.dt), 'n_rollout' : 1, 
						'seed' : 201, 'time_step' : args.dt, 'trajectory_length' : int(float(traj_duration) / args.dt), 
						'num_data_train' : num_trajectory, 'num_data_test' : num_data_test}
	main_fn(m_config_aux['cfg'], m_config_aux, coeff_dur_test=1) # coeff_dur_test : The test data is expanded on coeff_dur_test * traj_duration