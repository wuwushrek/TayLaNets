from collections import namedtuple

# Haiku for Neural networks
import haiku as hk
import jax
import jax.numpy as jnp

import numpy as np

# Optax for the optimization scheme
import optax

# Yaml and pickle import
import pickle
import yaml

from tqdm.auto import tqdm

suffix = '.pickle'

_INITIALIZER_MAPPING = \
{
	'Constant' : hk.initializers.Constant,
	'RandomNormal' : hk.initializers.RandomNormal,
	'RandomUniform' : hk.initializers.RandomUniform,
	'TruncatedNormal' : hk.initializers.TruncatedNormal,
	'VarianceScaling' : hk.initializers.VarianceScaling,
	'UniformScaling' : hk.initializers.UniformScaling
}

_ACTIVATION_FN = \
{
	'relu' : jax.nn.relu,
	'sigmoid' : jax.nn.sigmoid,
	'softplus' : jax.nn.softplus,
	'hard_tanh' : jax.nn.hard_tanh,
	'selu' : jax.nn.selu,
	'tanh' : jax.numpy.tanh
}

_OPTIMIZER_FN = \
{
	'adam' : optax.scale_by_adam,
	'adabelief' : optax.scale_by_belief,
	# 'adagrad' : optax.adagrad,
	# 'fromage' : optax.fromage,
	# 'noisy_sgd' : optax.noisy_sgd,
	# 'sgd' : optax.sgd,
	# 'adamw' : optax.adamw,
	# 'lamb' : optax.lamb,
	'yogi' : optax.scale_by_yogi,
	# 'rmsprop' : optax.scale_by_
}

SampleLog = namedtuple("SampleLog", 
						"xTrainList xNextTrainList xTrainExtraList uTrainList xu_train_lb xu_train_ub "+\
						"xTestList xNextTestList xTestExtraList uTestList xu_test_lb xu_test_ub "+\
						"coloc_points num_traj_data trajectory_length env_name env_extra_args seed_number "+\
						"nstate ncontrol time_step n_rollout data_set_file others")

# Save information related to the training and hyper parameters of the NN
HyperParamsNN = namedtuple("HyperParamsNN", "model_name nstate ncontrol time_step nn_params baseline_params optimizer "+\
							"batch_size pen_constr num_gradient_iterations freq_accuracy freq_save patience normalize")

# Save information rekated to the learninf steps
LearningLog = namedtuple("LearningLog", "sampleLog nn_hyperparams loss_evol learned_weights seed_list")



def build_params(m_params):
	""" Build the parameters of the neural network from a dictionary having a string representation of the neural network
		SPecifically, parse the dictionary with the parameters and use specified output_size and input_index to specifies
		what are the set of indexes that are used as input of the neural network and the number of output
		:param m_params: A dictionary specifying the number of hidden layers and inputs  and extra parameters to construct the neural netowrk
	"""
	assert ('input_index' in m_params and 'output' in m_params) or ('input_index' not in m_params and 'output' not in m_params)

	# Store the resulting neural network
	nn_params = dict()

	# Include input_index if already given
	if 'input_index' in m_params:
		nn_params['input_index'] =  m_params['input_index']

	# Check of the size of the output layer is given -> If yes this is not the nn for the midpoint/remainder value
	if 'output' in m_params:
		nn_params['output_sizes'] = (*m_params['output_sizes'], m_params['output'])
	else:
		nn_params['output_sizes'] = m_params['output_sizes']

	# Activation function
	nn_params['activation'] = _ACTIVATION_FN[m_params['activation']]

	# Initialization of the biais values
	b_init_dict = m_params['b_init']
	nn_params['b_init'] = _INITIALIZER_MAPPING[b_init_dict['initializer']](**b_init_dict['params'])

	# Initialization of the weight values
	w_init_dict = m_params['w_init']
	nn_params['w_init'] = _INITIALIZER_MAPPING[w_init_dict['initializer']](**w_init_dict['params'])

	return nn_params

def load_config_file(path_config_file, extra_args={}):
	""" Load the yaml configuration file giving the neural network params in addition to the directory
		containing the data set to be used for training and testing.
		:param path_config_file : Path to the adequate yaml file
		:param extra_args : dictionary that rovides command line arguments to use in-place of parameters inside path_config_file
	"""
	# Load the configuration file
	yml_file = open(path_config_file).read()
	m_config = yaml.load(yml_file, yaml.SafeLoader)
	m_config = {**m_config, **extra_args}

	# Extract and load the file containing the training and testing samples
	data_set_file = m_config['train_data_file']
	mFile = open(data_set_file, 'rb')
	mSampleLog = pickle.load(mFile)
	mFile.close()

	# Get the path of the file where to write the training outputs
	out_file = m_config['output_file']

	# Get the model name
	model_name = m_config['model_name']

	# Get the optimizer information
	opt_info = m_config['optimizer']

	# Get the parameters for enforcing constraints on colocation points
	pen_constr = m_config['pen_constr']

	# Get the list of seed
	seed_list = m_config['seed']

	# Get the batch_size to be used in the training process
	batch_size = m_config['batch_size']

	# Total number of gradient iterations
	num_gradient_iterations = m_config['num_gradient_iterations']

	# Frequency at which to evaluate the loss on the testing data
	freq_accuracy = m_config['freq_accuracy']

	# Get the patience criteria for early stopping
	patience = m_config.get('patience', -1)

	# Frequence at which to save and print the data in the output file
	freq_save = m_config['freq_save']

	# Specify if the loss function is divided by the number of states (in addition to the batch size)
	normalize = m_config.get('normalize', False)

	# Get the baseline_params (method and neural network of the midpoint/remainder)
	baseline_params = m_config['baseline_params']
	if 'baseline_name' in extra_args:
		baseline_params['name'] = extra_args['baseline_name']
	if 'baseline_order' in extra_args:
		baseline_params['order'] = extra_args['baseline_order']

	# Get the parameterization of the unknown part of the vector field
	nn_params = m_config.get('nn_params', {})


	# Build the hyper params NN structure
	hyperParams = HyperParamsNN(model_name, mSampleLog.nstate, mSampleLog.ncontrol, mSampleLog.time_step, 
							nn_params, baseline_params, opt_info, batch_size, pen_constr, 
							num_gradient_iterations, freq_accuracy, freq_save, patience, normalize)

	print('################# Configuration file #################')
	print(m_config)
	print('######################################################')
	
	return mSampleLog, hyperParams, (out_file, seed_list)


def load_data_yaml(path_config_file, extra_args={}):
	""" Load the yaml configuration file giving the training/testing information
		:param path_config_file : Path to the adequate yaml file
		:param extra_args : Extra argument from the command line
	"""
	# Load the configuration file and append command line arguments if given
	yml_file = open(path_config_file).read()
	m_config = yaml.load(yml_file, yaml.SafeLoader)
	m_config = {**m_config, **extra_args}
	print('################# Configuration file #################')
	print(m_config)
	print('######################################################')

	# Parse the environment name
	env_name = m_config['env_name'] # Environment name

	# Extra information on the environment
	env_extra_args = m_config.get('env_extra_args', {})

	# File to store the training/testing data set
	output_file = m_config['output_file']

	# Get the time step for the integration scheme
	time_step = m_config['time_step']

	# Seed number for random initialization of the state
	seed_number = m_config.get('seed', 1)

	# Extract the number of rollout 
	n_rollout = m_config.get('n_rollout', 1)

	# Extract the maximum episode length
	trajectory_length = m_config['trajectory_length']

	# Extract the number of trainig data -> A list of number of data
	num_traj_data = m_config.get('num_data_train', [2000])

	# Extract the number of testing data
	num_data_test = m_config.get('num_data_test', 500)

	# Get colocation related information
	num_data_colocation = m_config.get('num_data_colocation', 0)
	extra_noise_colocation = m_config.get('extra_noise_coloc', 0)

	# Bound on the control applied when training and testing
	lowU_train_val = m_config.get('utrain_lb', None)
	highU_train_val = m_config.get('utrain_ub', None)
	lowU_test_val = m_config.get('utest_lb', None)
	highU_test_val = m_config.get('utest_ub', None)

	lowX_train_val = m_config.get('xtrain_lb', None)
	highX_train_val = m_config.get('xtrain_ub', None)
	lowX_test_val = m_config.get('xtest_lb', None)
	highX_test_val = m_config.get('xtest_ub', None)

	# Other user specific attributes
	others = m_config.get('others', None)

	# Return parameters
	return SampleLog(None, None, None, None, (lowX_train_val,lowU_train_val), (highX_train_val,highU_train_val), 
		None, None, None, None, (lowX_test_val,lowU_test_val), (highX_test_val,highU_test_val), (None,None,None), 
		num_traj_data, trajectory_length, env_name, env_extra_args, seed_number, None, None, time_step, n_rollout, 
		output_file+'.pkl', others), (num_data_test, num_data_colocation, extra_noise_colocation)


# Hack function to merge a list of different shapes
def combine_list(mList):
	""" Merge a list consisting of several other list of different shape by
		broadcasting each sub-list to the largest length sublist
	"""
	max_size = np.max(np.array([ len(elem) for elem in mList]))
	m_new_list = list()
	for elem in mList:
		if len(elem) >= max_size:
			m_new_list.append(elem)
			continue
		elem.extend([elem[-1] for i in range(max_size-len(elem))])
		m_new_list.append(elem)
	return m_new_list

# Running average
def running_mean(x, N):
	""" Compute the running average over a window of size N
		:param x :	The value to plot for which the average is done on the last component
		:param N : 	The window average to consider
	"""
	cumsum = np.cumsum(np.insert(x, 0, 0, axis=1), axis=1)
	return (cumsum[..., N:] - cumsum[..., :-N]) / float(N)


# Generate the data required to plot the evolution of the loss function throughout the trainign process
def generate_loss(loss_evol_1traj, window=1):
	""" Generate the data to plot the loss evolution of a fixed trajectory and for its different seed
	"""
	# Load the training and testing data total loss function
	loss_tr = np.array(combine_list([ dict_val['total_loss_train'] for seed, dict_val in  sorted(loss_evol_1traj.items())]))
	loss_tr = running_mean(loss_tr, window)
	loss_te = np.array(combine_list([ dict_val['total_loss_test'] for seed, dict_val in sorted(loss_evol_1traj.items()) ]))
	loss_te = running_mean(loss_te, window)

	# Load the specific data on the rollout and constraints
	spec_data_tr = np.array(combine_list([ dict_val['rollout_err_train'] for seed, dict_val in sorted(loss_evol_1traj.items()) ]))
	spec_data_te = np.array(combine_list([ dict_val['rollout_err_test'] for seed, dict_val in sorted(loss_evol_1traj.items()) ]))
	# coloc_data_tr = np.array([ dict_val['coloc_err_train'] for seed, dict_val in sorted(loss_evol_1traj.items()) ])
	coloc_data_te = np.array(combine_list([ dict_val['coloc_err_test'] for seed, dict_val in sorted(loss_evol_1traj.items()) ]))

	# Compute the mean and standard deviation of the total loss
	total_loss_tr_mean, total_loss_tr_std = np.mean(loss_tr, axis=0), np.std(loss_tr, axis=0)
	total_loss_tr_max, total_loss_tr_min = np.max(loss_tr, axis=0), np.min(loss_tr, axis=0)
	total_loss_te_mean, total_loss_te_std = np.mean(loss_te, axis=0), np.std(loss_te, axis=0)
	total_loss_te_max, total_loss_te_min = np.max(loss_te, axis=0), np.min(loss_te, axis=0)

	# Compute the mean square loss without the penalization and l2 and constraints
	ms_error_tr_roll = running_mean(np.mean(spec_data_tr[:,:,:,1], axis=2), window)
	ms_error_te_roll = running_mean(np.mean(spec_data_te[:,:,:,1], axis=2), window)

	ms_error_tr_mean, ms_error_tr_std = np.mean(ms_error_tr_roll, axis=0), np.std(ms_error_tr_roll, axis=0)
	ms_error_tr_max, ms_error_tr_min = np.max(ms_error_tr_roll, axis=0), np.min(ms_error_tr_roll, axis=0)
	ms_error_te_mean, ms_error_te_std = np.mean(ms_error_te_roll, axis=0), np.std(ms_error_te_roll, axis=0)
	ms_error_te_max, ms_error_te_min = np.max(ms_error_te_roll, axis=0), np.min(ms_error_te_roll, axis=0)

	coloc_error_te_roll = running_mean(coloc_data_te[:,:,0], window)
	coloc_error_te_roll_mean, coloc_error_te_roll_std = np.mean(coloc_error_te_roll, axis=0), np.std(coloc_error_te_roll, axis=0)
	coloc_error_te_roll_max, coloc_error_te_roll_min = np.max(coloc_error_te_roll, axis=0), np.min(coloc_error_te_roll, axis=0)


	# Compute the value of the constraints without the penalization term 
	# constr_error_tr_roll = running_mean(np.mean(spec_data_tr[:,:,:,2], axis=2) + np.mean(spec_data_tr[:,:,:,3], axis=2) +  coloc_data_te[:,:,1] + coloc_data_te[:,:,2], window)
	# constr_error_te_roll = running_mean(np.mean(spec_data_te[:,:,:,2], axis=2) +  np.mean(spec_data_te[:,:,:,3], axis=2) +  coloc_data_te[:,:,1] + coloc_data_te[:,:,2], window)
	constr_error_tr_roll = running_mean(np.mean(spec_data_tr[:,:,:,2], axis=2)  + coloc_data_te[:,:,1], window)
	constr_error_te_roll = running_mean(np.mean(spec_data_te[:,:,:,2], axis=2)  + coloc_data_te[:,:,1] , window)
	constr_error_tr_mean, constr_error_tr_std = np.mean(constr_error_tr_roll, axis=0), np.std(constr_error_tr_roll, axis=0)
	constr_error_tr_max, constr_error_tr_min = np.max(constr_error_tr_roll, axis=0), np.min(constr_error_tr_roll, axis=0)
	constr_error_te_mean, constr_error_te_std = np.mean(constr_error_te_roll, axis=0), np.std(constr_error_te_roll, axis=0)
	constr_error_te_max, constr_error_te_min = np.max(constr_error_te_roll, axis=0), np.min(constr_error_te_roll, axis=0)

	return (total_loss_tr_mean, total_loss_tr_mean-total_loss_tr_std,  total_loss_tr_mean+total_loss_tr_std), \
			(total_loss_te_mean, total_loss_te_mean-total_loss_te_std, total_loss_te_mean+total_loss_te_std),\
			(ms_error_tr_mean, np.maximum(ms_error_tr_min,ms_error_tr_mean-ms_error_tr_std), np.minimum(ms_error_tr_max,ms_error_tr_mean+ms_error_tr_std) ),\
			(ms_error_te_mean, np.maximum(ms_error_te_min,ms_error_te_mean-ms_error_te_std), np.minimum(ms_error_te_max,ms_error_te_mean+ms_error_te_std) ),\
			(constr_error_tr_mean, np.maximum(constr_error_tr_min,constr_error_tr_mean-constr_error_tr_std), np.minimum(constr_error_tr_max,constr_error_tr_mean+constr_error_tr_std)),\
			(constr_error_te_mean, np.maximum(constr_error_te_min,constr_error_te_mean-constr_error_te_std), np.minimum(constr_error_te_max,constr_error_te_mean+constr_error_te_std)),\
			(coloc_error_te_roll_mean, np.maximum(coloc_error_te_roll_min,coloc_error_te_roll_mean-coloc_error_te_roll_std), np.minimum(coloc_error_te_roll_max,coloc_error_te_roll_mean+coloc_error_te_roll_std))


# Geometric mean function
def expanding_gmean_log(s):
	""" Geometric mean of a 2D array along its last axis
	"""
	return np.transpose(np.exp(np.transpose(np.log(s).cumsum(axis=0)) / (np.arange(s.shape[0])+1)))


# Evaluate the accuracy of the learned predictor
def generate_rel_error(util_fun, params, xtrue):
	""" Generate the relative error and geometric mean of the error for some given trajectories of the pendulum

		Probably need to check that the trajectories aren't too different for 
		the standard deviation plot to not be too wide
	"""
	# Extract the functions to compute the loss and predict next state
	pred_xnext, loss_fun =  util_fun

	# A list to save the time evolution of the error per each initialization 
	res_value = [ list() for i in range(1, xtrue.shape[1])]
	opt_geom_mean = 2
	f_traj_evol = None

	for seed, params_val in sorted(params.items()):
		# The initial point are all initial point in the given trajectories
		init_value = xtrue[:, 0, :]
		trajectory_evolution = [init_value]
		# Then we estimate the future time step
		for i, l_res_value in tqdm(zip(range(1, xtrue.shape[1]), res_value), total=len(res_value), leave=False):
			init_value, (vField, mPoint, unknown_terms) = pred_xnext(params_val, init_value)
			trajectory_evolution.append(init_value)
			# Compute the relative error
			curr_relative_error = np.linalg.norm(init_value-xtrue[:,i,:], axis=1)/(np.linalg.norm(xtrue[:,i,:], axis=1)+ np.linalg.norm(init_value, axis=1))
			l_res_value.extend(curr_relative_error)
		val_geom_mean = np.mean(curr_relative_error)
		if val_geom_mean < opt_geom_mean:
			opt_geom_mean = val_geom_mean
			f_traj_evol = trajectory_evolution
		tqdm.write('Seed [{}]\t\t\t:\tEnd point geom mean : {}'.format(seed, val_geom_mean))
	# print(len(f_traj_evol), f_traj_evol[0].shape)
	# Make to an array the relative error as a function of initialization 
	res_value = np.array(res_value)
	res_value = expanding_gmean_log(res_value)

	# Compute the mean value and standard deviation
	mean_err = np.mean(res_value, axis=1)
	# tqdm.write('{}'.format(mean_err))
	std_err = np.std(res_value, axis=1)
	# Compute the maximum and minim value for proper confidence bound
	max_err_value = np.max(res_value, axis=1)
	min_err_value = np.min(res_value, axis=1)
	maximum_val = np.minimum(max_err_value, mean_err+std_err)
	minimum_val = np.maximum(min_err_value, mean_err-std_err)
	return (mean_err, minimum_val , maximum_val), \
			(mean_err[-1], minimum_val[-1], maximum_val[-1]),\
			np.array(f_traj_evol).transpose((1,0,2))