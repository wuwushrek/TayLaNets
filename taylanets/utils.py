from collections import namedtuple

# Haiku for Neural networks
import haiku as hk
import jax
import jax.numpy as jnp

# Optax for the optimization scheme
import optax

# Yaml and pickle import
import pickle
import yaml

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

	print('################# Configuration file #################')
	print(m_config)
	print('######################################################')

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
							nn_params, baseline_params, optimizer, batch_size, pen_constr, 
							num_gradient_iterations, freq_accuracy, freq_save, patience, normalize)

	return mSampleLog, hyperParams, (out_file, seed_list)