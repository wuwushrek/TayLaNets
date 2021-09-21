# from jax.config import config
# config.update("jax_enable_x64", True)

import pickle

import jax
import jax.numpy as jnp
import numpy as np

import optax

from tqdm.auto import tqdm

from taylanets.taylanets import build_taylanets
from taylanets.utils import SampleLog, HyperParamsNN, LearningLog, build_params
from taylanets.utils import _INITIALIZER_MAPPING, _ACTIVATION_FN, _OPTIMIZER_FN


def train(sampleLog, hyperParams, seeds_list, out_file=None, known_dynamics=None):
	""" Train a model given the functions to compute the loss, the functions to update the parameters of the neural network,
		the initial parameters of each neural network, the training and testing trajectories, the colocation points,
		and more information related to the training process
		:param sampleLog 			: A dictionary containing data from the training, the testing data set and the environments of interest
		:param hyperParams 			: A dictionary containing the hyper parameters used to construct the learner / build each neural networks
		:seeds_list 				: The different seed for the initial weight parameters of the networks
		:out_file 					: Destination file to write the logs of the training process
		:known_dynamics 			: The known part of the dynamics of the system
	"""

	# Copy the sampleLog without the data 
	reducedSampleLog = SampleLog(None, None, None, None, sampleLog.xu_train_lb, sampleLog.xu_train_ub, 
									None, None, None, None, sampleLog.xu_test_lb, sampleLog.xu_test_ub, 
									(None,None,None), sampleLog.num_traj_data, sampleLog.trajectory_length, 
									sampleLog.env_name, sampleLog.env_extra_args, 
									sampleLog.seed_number, sampleLog.nstate, 
									sampleLog.ncontrol, sampleLog.time_step, 
									sampleLog.n_rollout, sampleLog.data_set_file, 
									sampleLog.others
								)
	print(reducedSampleLog)
	
	######################### Parse the data set used for training and validation
	# Parse the data input files for the training set
	xTrainList = np.asarray(sampleLog.xTrainList)
	xNextTrainList = np.asarray(sampleLog.xNextTrainList)
	uTrainList = np.asarray(sampleLog.uTrainList) if sampleLog.uTrainList is not None else None
	xTrainExtraList = np.asarray(sampleLog.xTrainExtraList) if sampleLog.xTrainExtraList is not None else None

	# Parse the data input for the testing set --> Directly store it in the device memory as it will be fully used for testing the loss
	xTestList = jnp.asarray(sampleLog.xTestList)
	xNextTestList = jnp.asarray(sampleLog.xNextTestList)
	uTestList = jnp.asarray(sampleLog.uTestList) if sampleLog.uTestList is not None else None
	xTestExtraList = jnp.asarray(sampleLog.xTestExtraList) if sampleLog.xTestExtraList is not None else None

	# Obtain and parse the colocation set if it is given by the user
	coloc_set, u_coloc_set, coloc_set_extra = (np.asarray(sampleLog.coloc_points[0]) if sampleLog.coloc_points[0] is not None else None),\
												(np.asarray(sampleLog.coloc_points[1]) if sampleLog.coloc_points[1] is not None else None),\
												(np.asarray(sampleLog.coloc_points[2]) if sampleLog.coloc_points[2] is not None else None)

	# For stopping early, the algorithm evaluate on a subset of colocation points, we take the last data from the coloc set --> this assume size coloc >= test
	assert coloc_set is None or xTestList.shape[0] <= coloc_set.shape[0]
	coloc_early_stopping, u_coloc_early_stopping, extra_coloc_early_stopping = None, None, None
	if coloc_set is not None:
		coloc_rng = jax.random.PRNGKey(seeds_list[-1])
		rand_keys = jax.random.randint(coloc_rng, shape=(xTestList.shape[0],), minval=0, maxval=coloc_set.shape[0])
		coloc_early_stopping = jnp.asarray(coloc_set[rand_keys, :]) if coloc_set is not None else None
		u_coloc_early_stopping = jnp.asarray(u_coloc_set[rand_keys, :]) if u_coloc_set is not None else None
		extra_coloc_early_stopping = jnp.asarray(coloc_set_extra[rand_keys, :]) if coloc_set_extra is not None else None

	########################## Build the optimization function
	# The optimizer for gradient descent over the loss function -> See yaml for more details
	lr = hyperParams.optimizer['learning_rate_init'] 		# Optimizer initial learning rate
	lr_end = hyperParams.optimizer['learning_rate_end'] 	# Optimizer end learning rate

	# Customize the gradient descent algorithm
	chain_list = [_OPTIMIZER_FN[hyperParams.optimizer['name']](**hyperParams.optimizer.get('params', {}))]

	# Add weight decay if enable
	decay_weight = hyperParams.optimizer.get('weight_decay', 0.0)
	if decay_weight > 0.0:
		chain_list.append(optax.add_decayed_weights(decay_weight))

	# Add scheduler for learning rate decrease --> Linear decrease given bt learning_rate_init and learning_rate_end
	m_schedule = optax.linear_schedule(-lr, -lr_end, hyperParams.num_gradient_iterations)
	chain_list.append(optax.scale_by_schedule(m_schedule))

	# Add gradient clipping if enable
	grad_clip_coeff = hyperParams.optimizer.get('grad_clip', 0.0)
	if grad_clip_coeff > 0.0:
		chain_list.append(optax.adaptive_grad_clip(clipping=grad_clip_coeff))

	# Build the solver finally
	opt = optax.chain(*chain_list)

	# Some logging execution --> Hyper parameters
	patience = hyperParams.patience # Period over which no improvement yields the best solution
	# Get the frequency at which to evaluate on the testing
	high_freq_record_rg = int(hyperParams.freq_accuracy[0]*hyperParams.num_gradient_iterations)
	high_freq_val = hyperParams.freq_accuracy[1]
	low_freq_val = hyperParams.freq_accuracy[2]
	update_freq = jnp.array([ (j % high_freq_val)==0 if j <= high_freq_record_rg else ((j % low_freq_val)==0 if j < hyperParams.num_gradient_iterations-1 else True) \
										for j in range(hyperParams.num_gradient_iterations)])
	hyperParams.pen_constr['coloc_set_size'] = 0 if coloc_set is None else (coloc_set.shape[0]+ xTestList.shape[0])
	# Some more printing
	print("\n###### Hyper Parameters: ######")
	print(hyperParams)
	print('Early stopping criteria = {}\n'.format(patience > 0))

	# Construct the neural network parameters and baseline parameters
	nn_params = { key : build_params(val) for key, val in hyperParams.nn_params.items()}
	baseline_params = { key : (build_params(val) if (key == 'remainder' or key == 'midpoint') else val ) for key, val in hyperParams.baseline_params.items()}

	############################ Initialize NN and crete update, pred + loss functions
	# Build the initial neural networks parameters for the different seeds used
	m_init_params_seed = dict()
	print ('######### Start Setting Seed #########')
	for seed_val in seeds_list:
		print ('=========> Setting seed : {}'.format(seed_val))
		m_rng = jax.random.PRNGKey(seed=seed_val)
		(params_init, m_pen_ineq_k, m_lagr_ineq_k) , pred_xnext, loss_fun, update, update_lagrange =\
				build_taylanets(m_rng, hyperParams.nstate, hyperParams.ncontrol, hyperParams.time_step, baseline_params, nn_params, opt,
						hyperParams.model_name, known_dynamics, hyperParams.pen_constr, hyperParams.batch_size,
						xTrainExtraList.shape[-1] if xTrainExtraList is not None else None, hyperParams.normalize)
		m_init_params_seed[seed_val] = (m_rng, (params_init, m_pen_ineq_k, m_lagr_ineq_k), (jax.jit(pred_xnext), jax.jit(loss_fun), jax.jit(update), jax.jit(update_lagrange)))
		print('Initial --> penalty coefficient : {} | Lagrangian coeff : {}'.format(m_pen_ineq_k, (m_lagr_ineq_k.shape if m_lagr_ineq_k is not None else None)))
	print ('######### End Setting Seed #########\n')

	# Dictionary for logging training details
	final_log_dict = {i : {} for i in range(len(sampleLog.num_traj_data))}
	final_params_dict = {i : {} for i in range(len(sampleLog.num_traj_data))}

	# Iterate over the different trajectory sizes
	for i, n_train in tqdm(enumerate(sampleLog.num_traj_data),  total=len(sampleLog.num_traj_data)):
		# The total number of point in this trajectory
		total_traj_size = n_train * sampleLog.trajectory_length

		# Dictionary to save the loss and the optimal params per radom seed 
		dict_loss_per_seed = {seed_val : {} for seed_val in m_init_params_seed}
		dict_params_per_seed = {seed_val : None for seed_val in m_init_params_seed}

		# Iterate for the current trajectory through the number of seed
		for seed_val in tqdm(m_init_params_seed, total=len(dict_params_per_seed), leave=False):

			# Obtain the neural network, the update, loss function and paramters of the neural network structure
			rng_gen, (params, m_pen_ineq_k, m_lagr_ineq_k), (pred_xnext, loss_fun, update, update_lagrange) = \
				m_init_params_seed[seed_val]

			# Initialize the optimizer with the initial parameters
			opt_state = opt.init(params)

			# Dictionary to save the loss, the rollout error, and the colocation accuracy
			dict_loss = {'total_loss_train' : list(), 'rollout_err_train' : list(), 'total_loss_test' : list(), 'rollout_err_test' : list(), 'coloc_err_train' : list(), 'coloc_err_test' : list()}

			# Store the best parameters attained by the learning process so far
			best_params = None 				# Weights returning best loss over the training set
			best_test_loss = None 			# Best test loss over the training set
			best_train_loss = None 			# Best training loss
			best_constr_val = None 			# Constraint attained by the best params
			best_test_noconstr = None 		# Best mean squared error without any constraints
			iter_since_best_param = None 	# Iteration since the last best weights values (iteration in terms of the loss evaluation period)
			number_lagrange_update = 0 		# Current number of lagrangian and penalty terms updates
			number_inner_iteration = 0 		# Current number of iteration since the latest update

			# Iterate to learn the weight of the neural network
			for step in tqdm(range(hyperParams.num_gradient_iterations), leave=False):

				# Random key generator for batch data
				rng_gen, subkey = jax.random.split(rng_gen)
				idx_train = jax.random.randint(subkey, shape=(hyperParams.batch_size,), minval=0, maxval=total_traj_size)

				# Sample and put into the GPU memory the batch-size data
				xTrain, xTrainNext = jnp.asarray(xTrainList[idx_train,:]), jnp.asarray(xNextTrainList[:,idx_train,:])
				xTrainExtra = jnp.asarray(xTrainExtraList[idx_train,:]) if xTrainExtraList is not None else None
				uTrain = jnp.asarray(uTrainList[:,idx_train,:]) if uTrainList is not None else None

				# Sample batches of the colocation data sets -> Get random samples around test/ and coloc set
				if coloc_set is not None:
					# # Get the indexes in the trainign and testing set
					# rng_gen, subkey = jax.random.split(rng_gen)
					# idxtrain_coloc = jax.random.randint(subkey, shape=(mParamsNN.pen_constr['batch_size_train'],), minval=0, maxval=xTrainList.shape[0])
					rng_gen, subkey = jax.random.split(rng_gen)
					idxtest_coloc = jax.random.randint(subkey, shape=(hyperParams.pen_constr['batch_size_test'],), minval=0, maxval=xTestList.shape[0])
					rng_gen, subkey = jax.random.split(rng_gen)
					idxcoloc_coloc = jax.random.randint(subkey, shape=(hyperParams.pen_constr['batch_size_coloc'],), minval=0, maxval=coloc_set.shape[0])
					# Get the correct set of lagragian multipliers
					# print(m_lagr_eq_k.shape)
					lag_ineq_temp = jnp.vstack((m_lagr_ineq_k[idxtest_coloc], m_lagr_ineq_k[xTestList.shape[0]+idxcoloc_coloc]))
					# Get the proper colocation points
					x_coloc_temp = jnp.vstack((xTestList[idxtest_coloc,:], coloc_set[idxcoloc_coloc,:]))
					u_coloc_temp = None if uTestList is None else jnp.vstack((uTestList[0,idxtest_coloc,:], u_coloc_set[idxcoloc_coloc,:]))
					x_coloc_temp_extra = None if xTestExtraList is None else jnp.vstack((xTestExtraList[idxtest_coloc,:], coloc_set_extra[idxcoloc_coloc,:]))
				else:
					# Set all the parameters to None
					m_pen_ineq_k = None
					lag_ineq_temp = None
					x_coloc_temp = coloc_set
					u_coloc_temp = u_coloc_set
					x_coloc_temp_extra = coloc_set_extra

				# Update the parameters of the NN and the state of the optimizer
				params, opt_state, (spec_data_tr, coloc_ctr) = update( params, opt_state, xTrainNext, xTrain, uTrain, extra_args=xTrainExtra, 
											pen_ineq_k=m_pen_ineq_k, lagr_ineq_k=lag_ineq_temp, 
											coloc_points=(x_coloc_temp, u_coloc_temp, x_coloc_temp_extra) 
										)

				# If it is time to evaluate the models do it
				if update_freq[number_inner_iteration]:
					_, (spec_data_te, coloc_cte) = loss_fun( params, xNextTestList, xTestList, uTestList, extra_args=xTestExtraList, 
																	pen_ineq_k=m_pen_ineq_k, 
																	lagr_ineq_k = m_lagr_ineq_k[xTestList.shape[0]+rand_keys] if m_lagr_ineq_k is not None else m_lagr_ineq_k, 
																	coloc_points = (coloc_early_stopping, u_coloc_early_stopping, extra_coloc_early_stopping))
					
					# Gather some information
					loss_tr = coloc_ctr[0]
					loss_te = coloc_cte[0]
					coloc_ctr = coloc_ctr[1:]
					coloc_cte = coloc_cte[1:]

					# Log the value obtained by evaluating the current model
					dict_loss['total_loss_train'].append(float(loss_tr))
					dict_loss['total_loss_test'].append(float(loss_te))
					dict_loss['rollout_err_train'].append(spec_data_tr)
					dict_loss['rollout_err_test'].append(spec_data_te)
					dict_loss['coloc_err_train'].append(coloc_ctr)
					dict_loss['coloc_err_test'].append(coloc_cte)

					# Initialize the parameters for the best model so far
					if number_inner_iteration == 0:
						best_params = params
						best_test_noconstr = jnp.mean(spec_data_te[:,1]) # Rollout mean
						best_test_loss = loss_te
						best_train_loss = loss_tr
						best_constr_val = coloc_cte
						iter_since_best_param = 0
						
					# Check if the validation metrics has improved
					if loss_te < best_test_loss:
						best_params = params
						best_test_loss = loss_te
						best_test_noconstr = jnp.mean(spec_data_te[:,1])
						best_constr_val = coloc_cte
						best_train_loss = loss_tr if loss_tr < best_train_loss else best_train_loss
						iter_since_best_param = 0
					else:
						best_train_loss = loss_tr if loss_tr < best_train_loss else best_train_loss
						iter_since_best_param += 1


				# Period at which we save the models
				if number_inner_iteration % hyperParams.freq_save == 0: # Period at which we save the models

					# Update for the current seed what is the lost and the best params found so far
					dict_loss_per_seed[seed_val] = dict_loss
					final_log_dict[i] = dict_loss_per_seed

					# Update the best params found so far
					dict_params_per_seed[seed_val] = best_params
					final_params_dict[i] = dict_params_per_seed

					# Create the log file
					if out_file is not None:
						mLog = LearningLog(sampleLog=reducedSampleLog, nn_hyperparams= hyperParams, 
											loss_evol=final_log_dict, learned_weights=final_params_dict, 
											seed_list=seeds_list)

						# Save the current information in the file and close the file
						mFile = open(out_file+'.pkl', 'wb') 
						pickle.dump(mLog, mFile)
						mFile.close()

					# Debug message to bring on the consolde
					dbg_msg = '[Iter[{}][{}], N={}]\t train : {:.2e} | Loss test : {:.2e}\n'.format(step, number_lagrange_update, n_train, loss_tr, loss_te)
					dbg_msg += '\t\tBest Loss Function : Train = {:.2e} | Test = {:.2e}, {:.2e} | Constr = {}\n'.format(best_train_loss, best_test_loss, best_test_noconstr, best_constr_val)
					dbg_msg += '\t\tPer rollout loss train : {} | Loss test : {}\n'.format(spec_data_tr[:,1], spec_data_te[:,1])
					dbg_msg += '\t\tPer rollout INEQ Constraint Train : {} | Test : {}\n'.format(spec_data_tr[:,2], spec_data_te[:,2])
					dbg_msg += '\t\tColocotion loss : {}\n'.format(coloc_cte)
					tqdm.write(dbg_msg)

				# Update the number of iteration the latest update of the lagrangian terms
				number_inner_iteration += 1
			
				# If the patience periode has been violated, try to break
				if patience > 0 and iter_since_best_param > patience:

					# Debug message
					tqdm.write('####################### EARLY STOPPING [{}][{}] #######################'.format(step, number_lagrange_update))
					tqdm.write('Best Loss Function : Train = {:.2e} | Test = {:.2e}, {:.2e}'.format(best_train_loss, best_test_loss, best_test_noconstr))

					# If there is no constraints then break
					if m_pen_ineq_k is None or m_lagr_ineq_k is None or coloc_set is None:
						break

					# Check if the constraints threshold has been attained
					if best_constr_val[1] < hyperParams.pen_constr['tol_constraint_ineq']:
						tqdm.write('Constraints satisfied: [ineq = {}]\n'.format(best_constr_val[1]))
						tqdm.write('##############################################################################\n')
						break

					# If the constraints threshold hasn't been violated, update the lagrangian and penalty coefficients
					m_pen_ineq_k, m_lagr_ineq_k = \
						update_lagrange(	best_params, 
											pen_ineq_k=m_pen_ineq_k, 
											lagr_ineq_k = m_lagr_ineq_k, 
											coloc_points=(jnp.vstack((xTestList, coloc_set)), 
															None if uTestList is None else jnp.vstack((uTestList[0], u_coloc_set)), 
															None if xTestExtraList is None else jnp.vstack((xTestExtraList, coloc_set_extra))
														)
										)

					# Update the new params to be the best params
					params = best_params

					# UPdate the number of inner iteration since the last lagrangian terms update
					number_inner_iteration = 0

					# Update the number of lagrange update so far
					number_lagrange_update += 1

					# Reinitialize the optimizer
					opt_state = opt.init(params)

					# Some printing for debug
					tqdm.write('Update Penalty : [ineq = {:.2e}, lag_ineq = {:.2e}]'.format(m_pen_ineq_k, jnp.sum(m_lagr_ineq_k)))

			# Once the algorithm has converged, collect and save the data and params
			dict_loss_per_seed[seed_val] = dict_loss
			final_log_dict[i] = dict_loss_per_seed
			dict_params_per_seed[seed_val] = best_params
			final_params_dict[i] = dict_params_per_seed

			# Create the log file
			if out_file is not None:
				mLog = LearningLog(sampleLog=reducedSampleLog, nn_hyperparams= hyperParams, 
									loss_evol=final_log_dict, learned_weights=final_params_dict, 
									seed_list=seeds_list)

				# Save the current information in the file and close the file
				mFile = open(out_file+'.pkl', 'wb') 
				pickle.dump(mLog, mFile)
				mFile.close()



# if __name__ == "__main__":
# 	import time
# 	from scipy.integrate import odeint as scipy_ode
# 	import haiku as hk

# 	# Test the training algorithm on a very simple example --> Exponential dynamics
# 	def known_dyn(x, t=0, lam=-2):
# 		return lam * x

# 	# Create the data set
# 	def generate_sample(dyn_fn, seed, time_step, num_traj, traj_length, x_lb, x_ub, n_rollout, merge_traj=True):
# 		np.random.seed(seed)
# 		t_indexes = np.array([i * time_step for i in range(traj_length+n_rollout)])
# 		res_state = []
# 		res_rollout = [ [] for r in range(n_rollout)]
# 		for i in tqdm(range(num_traj)):
# 			x_init = np.random.uniform(low = x_lb, high=x_ub, size = x_lb.shape)
# 			x_traj = scipy_ode(dyn_fn, x_init, t_indexes, full_output=False)
# 			if merge_traj:
# 				res_state.extend(x_traj[:traj_length])
# 			else:
# 				res_state.append(x_traj[:traj_length])
# 			for j, r in enumerate(res_rollout):
# 				r.extend(x_traj[(j+1):(j+1+traj_length),:])
# 		return res_state, res_rollout

# 	seed = 10
# 	time_step = 0.1
# 	n_rollout = 4
# 	num_traj_train = [10]
# 	num_traj_test = 5
# 	traj_length = 50
# 	x_lb = np.array([-1, -1])
# 	x_ub = np.array([1, 1])
# 	x_lb_test = 2 * np.array([-1, -1])
# 	x_ub_test = 2 * np.array([1, 1])
# 	x_traj, xnext_traj = generate_sample(known_dyn, seed, time_step, num_traj_train[-1], traj_length, x_lb, x_ub, n_rollout)
# 	x_test, xnext_test = generate_sample(known_dyn, seed, time_step, num_traj_test, traj_length, x_lb_test, x_ub_test, n_rollout)
# 	# coloc_points = (np.array(x_traj)[:len(x_test)+20,:], None, None)
# 	coloc_points = (None, None, None)
# 	# print(x_traj)
# 	mSampleLog = SampleLog(x_traj, xnext_traj, None, None, (x_lb,None), (x_ub, None), \
# 							x_test, xnext_test, None, None, (x_lb_test,None), (x_ub_test,None), 
# 							coloc_points, num_traj_train, traj_length, 'test', {}, 
# 							seed, x_lb.shape[0], 0, time_step, n_rollout, data_set_file=None, others=None)

# 	nn_params = {}
# 	taylor_order = 2
# 	baseline_params = {'name' : 'tayla', 'order' : taylor_order, 'midpoint' : {'output_sizes' : [16, 16], 'activation' : 'tanh', 'b_init' : {'initializer' : 'Constant', 'params' : {'constant' : 0} }, 'w_init' : {'initializer' : 'RandomUniform', 'params' : {'minval' : -0.1, 'maxval' : 0.1} } } }

# 	pen_constr = {'batch_size_test' : 8, 'batch_size_coloc' : 8,  'pen_ineq_init' : 0.0001,  'beta_ineq' : 10, 'coloc_set_size' : 0, 'tol_constraint_ineq' : 0.001}
# 	if taylor_order == 1:
# 		optimizer = {'name' : 'adam', 'learning_rate_init' : 0.002, 'learning_rate_end' : 0.001, 'weight_decay' : 0.0001, 'grad_clip' : 0.01}
# 		num_gradient_iterations = 10000
# 	else:
# 		optimizer = {'name' : 'adam', 'learning_rate_init' : 0.01, 'learning_rate_end' : 0.01, 'weight_decay' : 0.0001, 'grad_clip' : 0.01}
# 		num_gradient_iterations = 20000
# 	batch_size = 64
# 	freq_accuracy = [1.0, 100, 100]
# 	freq_save = 200
# 	patience = -1
# 	normalize = False
# 	hyperParams = HyperParamsNN('model_test', x_lb.shape[0], 0, time_step, nn_params, baseline_params, 
# 								optimizer, batch_size, pen_constr, num_gradient_iterations, freq_accuracy, freq_save, patience, normalize)



# 	# print(hyperParams)
# 	# print(mSampleLog)
# 	train(mSampleLog, hyperParams, [1], 'test_output', known_dynamics=known_dyn)

# 	baseline_params['name'] = 'rk4'
# 	m_rng = jax.random.PRNGKey(seed=1)
# 	(params, _, _) , pred_xnext, loss_fun, _, _ =\
# 		build_taylanets(m_rng, hyperParams.nstate, hyperParams.ncontrol, hyperParams.time_step, hyperParams.baseline_params, hyperParams.nn_params, None,
# 			hyperParams.model_name, known_dyn, hyperParams.pen_constr, hyperParams.batch_size,
# 			None, hyperParams.normalize)
# 	cost, (m_res, extra) = jax.jit(loss_fun)(params, jnp.array(xnext_test), jnp.array(x_test), None , None , None, None, coloc_points=(None, None, None))
# 	print(cost)
# 	print(m_res)
# 	print(extra)