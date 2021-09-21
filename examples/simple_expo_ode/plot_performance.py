import numpy as np
import pickle

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from tqdm.auto import tqdm

from taylanets.utils import generate_loss, build_params, generate_rel_error
from taylanets.taylanets import build_taylanets

from generate_sample import system_ode, exact_solution

def load_files(data_to_show, integ_scheme=False):
	""" Load a series of file to be used in plotting the rewards and the 
	"""
	# The list to save the results of the log data
	m_learning_log = []

	# The list to save the functions for predicting the next state and computing the loss function
	m_util_fun = []

	m_util_integ = []

	for f_name in data_to_show:
		# Load the file
		m_file = open(f_name, 'rb')

		# Load the Log data
		m_llog = pickle.load(m_file)

		# Close the file
		m_file.close()

		# Append the data in the list of logs to analyze
		m_learning_log.append(m_llog)

		# Parse the file to obtain the function to predict the NN -> Just one element should be enough
		hyperParams = m_llog.nn_hyperparams
		seed_val = m_llog.seed_list[0]

		# Randome generator key for initialization of the NN
		m_rng = jax.random.PRNGKey(seed=seed_val)

		# Build the NN
		nn_params = { key : build_params(val) for key, val in hyperParams.nn_params.items()}
		baseline_params = { key : (build_params(val) if (key == 'remainder' or key == 'midpoint') else val ) for key, val in hyperParams.baseline_params.items()}

		# Set extra parameters to None for this specific example and build the function to predict next state and compute loss
		_ , pred_xnext, loss_fun, _, _ =\
				build_taylanets(m_rng, hyperParams.nstate, hyperParams.ncontrol, hyperParams.time_step, baseline_params, nn_params, None,
						hyperParams.model_name, jax.vmap(system_ode), hyperParams.pen_constr, hyperParams.batch_size,
						None, hyperParams.normalize)
		# Add the obtained function to the utils functions
		m_util_fun.append((jax.jit(pred_xnext), jax.jit(loss_fun)))

		# Print stuff for visualizing training params
		print(m_llog.nn_hyperparams)
		print(m_llog.sampleLog)

		if integ_scheme:
			baseline_params['name'] = 'taylor'
			_ , pred_xnext, loss_fun, _, _ =\
				build_taylanets(m_rng, hyperParams.nstate, hyperParams.ncontrol, hyperParams.time_step, baseline_params, nn_params, None,
						hyperParams.model_name, jax.vmap(system_ode), hyperParams.pen_constr, hyperParams.batch_size,
						None, hyperParams.normalize)
			m_util_integ.append( ( (jax.jit(pred_xnext), jax.jit(loss_fun)), 'Taylor {}'.format(baseline_params['order']) ) )			

	if integ_scheme:
		baseline_params['name'] = 'rk4'
		_ , pred_xnext_rk4, loss_fun_rk4, _, _ =\
			build_taylanets(m_rng, hyperParams.nstate, hyperParams.ncontrol, hyperParams.time_step, baseline_params, nn_params, None,
					hyperParams.model_name, jax.vmap(system_ode), hyperParams.pen_constr, hyperParams.batch_size,
					None, hyperParams.normalize)
		m_util_integ.append(((jax.jit(pred_xnext_rk4), jax.jit(loss_fun_rk4)), 'RK4'))

		baseline_params['name'] = 'odeint'
		_ , pred_xnext_ode, loss_fun_ode, _, _ = \
			build_taylanets(m_rng, hyperParams.nstate, hyperParams.ncontrol, hyperParams.time_step, baseline_params, nn_params, None,
					hyperParams.model_name, jax.vmap(system_ode), hyperParams.pen_constr, hyperParams.batch_size,
					None, hyperParams.normalize)
		m_util_integ.append( ( (jax.jit(pred_xnext_ode), jax.jit(loss_fun_ode)), 'ODEint') )

	return m_learning_log, m_util_fun, m_util_integ


if __name__ == "__main__":
	import time
	import argparse

	# example_command ='python generate_sample.py --cfg dataset_gen.yaml --output_file data/xxxx --time_step 0.01 --n_rollout 5'
	parser = argparse.ArgumentParser()
	parser.add_argument('--logdirs', nargs='+', required=True)
	parser.add_argument('--legends', nargs='+', required=True)
	parser.add_argument('--colors', nargs='+', required=True)
	parser.add_argument('--extra_colors', nargs='+', required=True)
	parser.add_argument('--window', '-w', type=int, default=1)
	parser.add_argument('--seed', type=int, default=701)
	parser.add_argument('--evaluate_on_test', action='store_true')
	parser.add_argument('--num_traj', type=int, default=100)
	parser.add_argument('--num_point_in_traj', type=int, default=100)
	parser.add_argument('--show_constraints', action='store_true')
	parser.add_argument('--noise', type=float, default=0.05)
	parser.add_argument('--indx_traj_test', type=int, default=0)
	parser.add_argument('--indx_traj', type=int, default=0)
	args = parser.parse_args()

	# Load the data
	m_logs, m_pred_loss, m_pred_loss_base = load_files(args.logdirs, integ_scheme=True)
	actual_dt = m_logs[0].sampleLog.time_step # Assume the step size is the same for every file
	n_state = m_logs[0].sampleLog.nstate

	##########################################################################################################################
	# Should define some parameters for the plot here
	alpha_std = 0.4
	alpha_std_test = 0.2
	linewidth = 1.5
	markerstyle = "*"
	markersize = 8
	linestyle_test = {'linestyle' : '--', 'dashes' : (10, 5)}
	# Generate the figure for the loss function evolution
	m_figs_loss = list()
	for _ in range(len(m_logs[0].sampleLog.num_traj_data)):
		ncols =  2 if args.show_constraints else 1
		nrows =  2 if args.show_constraints else 1
		m_fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 5*nrows), 
								sharex=True, sharey=False)
		m_figs_loss.append((m_fig, axs))
		# Set the title

	# Generate the plots for the loss function evolution on all the data
	for (log_data, legend, color) in tqdm(zip(m_logs, args.legends, args.colors), total=len(m_logs)):
		loss_evol_data = {i : val for i, val in log_data.loss_evol.items() if len(val) > 0}
		for traj_id, n_train, (m_fig, axs) in zip(loss_evol_data, log_data.sampleLog.num_traj_data, m_figs_loss):
			tl_tr, tl_te, ml_tr, ml_te, ctr_tr, ctr_te, coloc_err = generate_loss(loss_evol_data[traj_id], args.window)
			temp_nnparams = log_data.nn_hyperparams
			high_freq_record_rg = int(temp_nnparams.freq_accuracy[0]*temp_nnparams.num_gradient_iterations)
			high_freq_val = temp_nnparams.freq_accuracy[1]
			low_freq_val = temp_nnparams.freq_accuracy[2]
			update_freq = np.array([ (i % high_freq_val)==0 if i <= high_freq_record_rg else ((i % low_freq_val)==0 if i < temp_nnparams.num_gradient_iterations-1 else True) \
										for i in range(temp_nnparams.num_gradient_iterations)])
			gradient_step = np.array([i for i in range(temp_nnparams.num_gradient_iterations)])[update_freq]
			# print(axs)
			main_axs = axs.ravel()[0] if args.show_constraints else axs
			main_axs.plot(gradient_step[:tl_tr[0].shape[0]], tl_tr[0], color=color, linewidth=linewidth, label=legend)
			main_axs.fill_between(gradient_step[:tl_tr[0].shape[0]], tl_tr[1], tl_tr[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
			main_axs.plot(gradient_step[:tl_te[0].shape[0]], tl_te[0], color='dark'+color, linewidth=linewidth, label=legend, **linestyle_test)
			main_axs.fill_between(gradient_step[:tl_te[0].shape[0]], tl_te[1], tl_te[2], linewidth=linewidth, facecolor=color, alpha=alpha_std_test)
			main_axs.set_yscale('log')
			main_axs.set_xlabel(r'$\mathrm{Time \ steps}$')
			main_axs.set_ylabel(r'$\mathrm{Total \ loss}$ ($N_{\mathrm{train}}='+ str(n_train)+'$)')
			main_axs.grid()
			main_axs.legend(loc='best')
			if args.show_constraints:
				# Print the mean squared error without the l2 and constraint
				if np.sum(ml_tr[0]) > 1e-8:
					# axs.ravel()[1].plot(gradient_step[:ml_tr[0].shape[0]], ml_tr[0], color=color, linewidth=linewidth, label=legend)
					# axs.ravel()[1].fill_between(gradient_step[:ml_tr[1].shape[0]], ml_tr[1], ml_tr[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
					axs.ravel()[1].plot(gradient_step[:ml_te[0].shape[0]], ml_te[0], color='dark'+color, linewidth=linewidth, label=legend, **linestyle_test)
					axs.ravel()[1].fill_between(gradient_step[:ml_te[1].shape[0]], ml_te[1], ml_te[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
					axs.ravel()[1].set_yscale('log')
					axs.ravel()[1].set_xlabel(r'$\mathrm{Time \ steps}$')
					axs.ravel()[1].set_ylabel(r'$\mathrm{Mean \ squared \ loss}$ ($N_{\mathrm{train}}='+ str(n_train)+'$)')
					axs.ravel()[1].grid()
					axs.ravel()[1].legend(loc='best')
				# Plot the constraints
				if np.sum(ctr_tr[0]) > 1e-8:
					# axs.ravel()[2].plot(gradient_step[:ctr_tr[0].shape[0]], ctr_tr[0], color=color, linewidth=linewidth, label=legend)
					# axs.ravel()[2].fill_between(gradient_step[:ctr_tr[1].shape[0]], ctr_tr[1], ctr_tr[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
					axs.ravel()[2].plot(gradient_step[:ctr_te[0].shape[0]], ctr_te[0], color='dark'+color, linewidth=linewidth, label=legend, **linestyle_test)
					axs.ravel()[2].fill_between(gradient_step[:ctr_te[1].shape[0]], ctr_te[1], ctr_te[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
					axs.ravel()[2].set_yscale('log')
					axs.ravel()[2].set_xlabel(r'$\mathrm{Time \ steps}$')
					axs.ravel()[2].set_ylabel(r'$\mathrm{Constraints \ loss}$ ($N_{\mathrm{train}}='+ str(n_train)+'$)')
					axs.ravel()[2].grid()
					axs.ravel()[2].legend(loc='best')
				# Plot the colocation
				if np.sum(coloc_err[0]) > 1e-8:
					axs.ravel()[3].plot(gradient_step[:coloc_err[0].shape[0]], coloc_err[0], color=color, linewidth=linewidth, label=legend)
					axs.ravel()[3].fill_between(gradient_step[:coloc_err[1].shape[0]], coloc_err[1], coloc_err[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
					axs.ravel()[3].set_yscale('log')
					axs.ravel()[3].set_xlabel(r'$\mathrm{Time \ steps}$')
					axs.ravel()[3].set_ylabel(r'$\mathrm{Colocation \ Err.}$ ($N_{\mathrm{train}}='+ str(n_train)+'$)')
					axs.ravel()[3].grid()
					axs.ravel()[3].legend(loc='best')


	# Save this figure if required
	import tikzplotlib
	from pathlib import Path
	dir_save = str(Path(args.logdirs[0]).parent)
	for n_train, (m_fig, axs) in zip(log_data.sampleLog.num_traj_data, m_figs_loss):
		tikzplotlib.clean_figure(fig=m_fig)
		tikzplotlib.save(dir_save+'/loss_{}.tex'.format(n_train), figure=m_fig)
		m_fig.savefig(dir_save+'/loss_{}.png'.format(n_train), dpi=300)

	#########################################################################################################################################################################

	#########################################################################################################################
	m_rng  = jax.random.PRNGKey(args.seed)
	m_rng, subkey = jax.random.split(m_rng)

	# Check if the test should be evaluated on the training set
	if args.evaluate_on_test:
		x_init = jax.random.uniform(subkey, shape=(n_state,), 
										minval=jnp.array(m_logs[0].sampleLog.xu_test_lb[0]), maxval=jnp.array(m_logs[0].sampleLog.xu_test_ub[0]))
	else:
		x_init = jax.random.uniform(subkey, shape=(n_state,), 
										minval=jnp.array(m_logs[0].sampleLog.xu_train_lb[0]), maxval=jnp.array(m_logs[0].sampleLog.xu_train_ub[0]))

	# Generate a tube of trajectories around that initial point with the specified noise
	m_rng, subkey = jax.random.split(m_rng)
	x_init_lb = x_init - jax.random.uniform(subkey, shape=(n_state,), minval=0, maxval=args.noise)
	m_rng, subkey = jax.random.split(m_rng)
	x_init_ub = x_init + jax.random.uniform(subkey, shape=(n_state,), minval=0, maxval=args.noise)

	# Full evaluation set
	m_rng, subkey = jax.random.split(m_rng)
	x_eval_init = jax.random.uniform(subkey, (args.num_traj, n_state), minval = jnp.array(x_init_lb), maxval=jnp.array(x_init_ub))
	testTraj, _ = exact_solution(x_eval_init, actual_dt, args.num_point_in_traj, 1, merge_traj=False)
	time_index = [ actual_dt * i for i in range(1, args.num_point_in_traj)]

	# Generate the plots showing the relative error
	list_axes_rel_err =  list()
	for i in range(len(log_data.sampleLog.num_traj_data)):
		fig_rel_err =  plt.figure()
		ax_rel_err = plt.gca()
		ax_rel_err.set_xlabel(r'$\mathrm{Time \ (seconds)}$')
		ax_rel_err.set_yscale('log')
		ax_rel_err.grid()
		list_axes_rel_err.append((ax_rel_err,fig_rel_err))

	figure_gm_rerr = plt.figure()
	ax_gm_rerr = plt.gca()
	ax_gm_rerr.set_xlabel(r'$\mathrm{Number \ of \ training \ trajectories }$')
	ax_gm_rerr.set_ylabel(r'$\mathrm{Geometric \ mean \ of \ relative \ error }$')
	ax_gm_rerr.set_yscale('log')
	ax_gm_rerr.grid()

	# Generate the data with the accuracy of the learned model
	mStateEvol = dict()
	for pred_loss_fun, log_data, legend, color in tqdm(zip(m_pred_loss, m_logs, args.legends, args.colors), total=len(m_pred_loss)):
		training_list = list()
		list_gm_err_mean, list_gm_err_min, list_gm_err_max  = list(), list(), list()
		curr_learned_params = {i : val for i, val in log_data.learned_weights.items() if len(val) > 0}
		counterVal = 0
		for traj_id, n_train, (ax_rel_err,_) in tqdm(zip(curr_learned_params, log_data.sampleLog.num_traj_data,list_axes_rel_err), total=len(list_axes_rel_err), leave=False):
			rel_err, gm_rel_err, trajPred = generate_rel_error(pred_loss_fun, curr_learned_params[traj_id], testTraj)
			if counterVal == args.indx_traj:
				mStateEvol[legend] = trajPred
			list_gm_err_mean.append(float(gm_rel_err[0]))
			list_gm_err_min.append(float(gm_rel_err[1]))
			list_gm_err_max.append(float(gm_rel_err[2]))
			training_list.append(n_train)
			ax_rel_err.plot(time_index, rel_err[0], color=color, linewidth=linewidth, label=legend) # label=r'$N_{\mathrm{traj}} = '+ str(n_train) + '$'
			ax_rel_err.fill_between(time_index, rel_err[1], rel_err[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
			ax_rel_err.set_ylabel(r'$\mathrm{Relative \ error}$ ($N_{\mathrm{train}}='+ str(n_train)+'$)')
			ax_rel_err.legend(loc='best')
			tqdm.write('[N_train = {}, Legend = {}]\t : Geometric mean [Mean | Mean-Std | Mean+Std] -> {:.4f} | {:.4f} | {:.4f}'.format(n_train, legend, float(gm_rel_err[0]), float(gm_rel_err[1]), float(gm_rel_err[2])))
		ax_gm_rerr.plot(training_list, list_gm_err_mean, color=color, linewidth=linewidth, marker=markerstyle, markersize=markersize, label=legend)
		ax_gm_rerr.fill_between(training_list, list_gm_err_min, list_gm_err_max, linewidth=linewidth, facecolor=color, alpha=alpha_std)
	ax_gm_rerr.legend(loc='best')

	# Iterate over the fixed and adaptive integration to check the performance of the ode integration
	for (pred_loss_fun, legend), color in tqdm(zip(m_pred_loss_base, args.extra_colors), total=len(m_pred_loss_base)):
		training_list = list()
		list_gm_err_mean, list_gm_err_min, list_gm_err_max  = list(), list(), list()
		rel_err, gm_rel_err, trajPred = generate_rel_error(pred_loss_fun, {-1 : {}}, testTraj)
		mStateEvol[legend] = trajPred
		for n_train, (ax_rel_err,_) in tqdm(zip(m_logs[0].sampleLog.num_traj_data,list_axes_rel_err), total=len(list_axes_rel_err), leave=False):
			training_list.append(n_train)
			list_gm_err_mean.append(float(gm_rel_err[0]))
			list_gm_err_min.append(float(gm_rel_err[1]))
			list_gm_err_max.append(float(gm_rel_err[2]))
			ax_rel_err.plot(time_index, rel_err[0], color=color, linewidth=linewidth, label=legend) # label=r'$N_{\mathrm{traj}} = '+ str(n_train) + '$'
			ax_rel_err.fill_between(time_index, rel_err[1], rel_err[2], linewidth=linewidth, facecolor=color, alpha=alpha_std)
			ax_rel_err.set_ylabel(r'$\mathrm{Relative \ error}$ ($N_{\mathrm{train}}='+ str(n_train)+'$)')
			ax_rel_err.legend(loc='best')
			tqdm.write('[N_train = {}, Legend = {}]\t : Geometric mean [Mean | Mean-Std | Mean+Std] -> {:.2e} | {:.2e} | {:.2e}'.format(n_train, legend, float(gm_rel_err[0]), float(gm_rel_err[1]), float(gm_rel_err[2])))
		ax_gm_rerr.plot(training_list, list_gm_err_mean, color=color, linewidth=linewidth, marker=markerstyle, markersize=markersize, label=legend)
		ax_gm_rerr.fill_between(training_list, list_gm_err_min, list_gm_err_max, linewidth=linewidth, facecolor=color, alpha=alpha_std)
	ax_gm_rerr.legend(loc='best')


	# Show all the plots
	plt.show()