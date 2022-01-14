""" Code source to plot the results associated to
    learning the midpoint value
"""
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from generate_sample import exact_solution
from learn_midpoint import init_model as known_dyn_init
from learn_dynamics import init_model as unkn_dyn_init

import jax
import jax.numpy as jnp

import pickle

import tikzplotlib

import time

# Geometric mean function
def expanding_gmean_log(s):
    """ Geometric mean of a 2D array along its last axis
    """
    return np.transpose(np.exp(np.transpose(np.log(s).cumsum(axis=0)) / (np.arange(s.shape[0])+1)))

def prediction_error(params, pred_fn, trajx):
    """ Provide prediction error given a predictor function
    """
    # trajx is a list of 2D trajectories --> first dimension in time and second dimension state space
    num_time = trajx.shape[1]
    state_init = trajx[:,0,:]
    res_list = [state_init]
    rel_error_list = []
    time_evolution = []

    for i in tqdm(range(1,num_time), leave=False):
        curr_time = time.time()
        (next_state, nfe), _ = pred_fn(params, state_init)
        next_state.block_until_ready()
        comp_time = time.time() - curr_time
        state_init = next_state
        res_list.append(next_state)
        curr_relative_error = np.linalg.norm(state_init-trajx[:,i,:], axis=1)/(np.linalg.norm(trajx[:,i,:], axis=1)+ np.linalg.norm(state_init, axis=1))
        # curr_relative_error[curr_relative_error == np.nan] = 1.
        rel_error_list.append(curr_relative_error)
        time_evolution.append(comp_time)


    # Make to an array the relative error as a function of initialization 
    res_list = np.array(res_list)
    time_evolution = np.array(time_evolution)[1:-1] # Remove first compute due to jit
    # res_value = expanding_gmean_log(np.array(rel_error_list))
    res_value = np.cumsum(np.array(rel_error_list), axis=0) / np.array([[i+1] for i in range(num_time-1)])

    # Compute the mean value and standard deviation
    mean_err = np.mean(res_value, axis=1)
    # tqdm.write('{}'.format(mean_err))
    std_err = np.std(res_value, axis=1)
    # Compute the maximum and minim value for proper confidence bound
    max_err_value = np.max(res_value, axis=1)
    min_err_value = np.min(res_value, axis=1)

    maximum_val = np.minimum(max_err_value, mean_err+std_err)
    minimum_val = np.maximum(min_err_value, mean_err-std_err)

    meanTime = np.mean(time_evolution)
    stdTime = np.std(time_evolution)
    maxTime = np.max(time_evolution)
    maxTime = np.minimum(maxTime, meanTime+stdTime)
    minTime = np.min(time_evolution)
    minTime = np.maximum(minTime, meanTime-stdTime)

    return (mean_err, minimum_val , maximum_val), \
            (mean_err[0], minimum_val[0], maximum_val[0]),\
            np.array(res_list).transpose((1,0,2)),\
            (meanTime, minTime, maxTime)



def plot_prederror_dt(trajx, dt, files, labels, colors, show=True, seed=0, save=False, spacing=1, spacing_fill=1):
    fig = plt.figure()
    rng = jax.random.PRNGKey(seed)
    time_indexes = np.array([ i * dt for i in range(trajx.shape[1])])[1:]
    tqdm.write('----------------- dt = {} ------------------'.format(dt))
    final_data = dict()
    for info_data, label, color in tqdm(zip(files, labels, colors), total = len(labels)):
        tqdm.write('Method = {} | Order = {} | N step = {}'.format(info_data['method'], info_data['order'], info_data['n_step']))
        i_params, forward, *extra = info_data['init_module'](rng, ts=dt, batch_size=trajx.shape[0], **{key : value for key, value in info_data.items() if key != 'params' and key != 'init_module' and key != 'others'})
        m_params = info_data.get('params', i_params)
        (mean_err, min_val, max_val), (fmean_err, fmin_val, fmax_val), _, (meant, mint, maxt) = prediction_error(m_params, forward, trajx)
        plt.plot(time_indexes[::spacing], mean_err[::spacing], color=color, linewidth=2, linestyle='dashed', label=label)
        plt.fill_between(time_indexes[::spacing_fill], min_val[::spacing_fill], max_val[::spacing_fill], linewidth=2, facecolor=color, alpha=0.6)
        print_msg = 'End Time: Mean Err = {} | Min Err = {} | Max Err = {} - Mean Time = {} | Min Time = {} | Max Time = {}'.format(fmean_err, fmin_val, fmax_val, meant, mint, maxt)
        tqdm.write(print_msg)
        final_data[label] = (color, fmean_err, fmin_val, fmax_val, meant, mint, maxt)
        # final_data.append((label, color, fmean_err, fmin_val, fmax_val, meant, mint, maxt))

    plt.yscale('log')
    plt.xlabel(r'$\mathrm{Time(s)}$')
    plt.ylabel(r'$\mathrm{{Rel. \ Pred. \ Error[\Delta t = {}]}}$'.format(dt))
    plt.grid(True)
    plt.legend(loc='best')
    if save:
        tikzplotlib.clean_figure()
        tikzplotlib.save('data/prederror_dt{}.tex'.format(dt))
        plt.savefig('data/prederror_dt{}.png'.format(dt), dpi=300, transparent=True)
    if show:
        plt.show()
    return final_data


# Running average
def running_mean(x, N):
    """ Compute the running average over a window of size N
        :param x :  The value to plot for which the average is done on the last component
        :param N :  The window average to consider
    """
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    return (cumsum[..., N:] - cumsum[..., :-N]) / float(N)


def plot_log_dt(dt, files, labels, colors, show=True, save=False, spacing=1, window=1):
    fig = plt.figure()
    tqdm.write('----------------- dt = {} ------------------'.format(dt))
    final_data = dict()
    for info_data, label, color in tqdm(zip(files, labels, colors), total = len(labels)):
        tqdm.write('Method = {} | Order = {} | N step = {}'.format(info_data['method'], info_data['order'], info_data['n_step']))
        loss_evol = running_mean(np.array(info_data['others']['loss_evol_odeint']), window)
        true_loss_evol = running_mean(np.array(info_data['others']['loss_evol_test']), window)
        t_axis = np.arange(true_loss_evol.shape[0])
        spacing_val = int(len(true_loss_evol) / spacing) if len(true_loss_evol) > spacing else 1
        plt.plot(t_axis[::spacing_val], loss_evol[::spacing_val], color=color, linewidth=2, linestyle='dashed', label=label)
        plt.plot(t_axis[::spacing_val],true_loss_evol[::spacing_val], color=color, linewidth=2, linestyle='solid')

    plt.ylabel(r'$\mathrm{Loss \ Test}$')
    plt.xlabel(r'$\mathrm{Training \ steps}$')
    plt.grid(True)
    plt.legend(loc='best')
    if save:
        tikzplotlib.clean_figure()
        tikzplotlib.save('data/lossfun_dt{}.tex'.format(dt))
        plt.savefig('data/lossfun_dt{}.png'.format(dt), dpi=300, transparent=True)
    if show:
        plt.show()


def extract_params_logs(filename, prefix='data/'):
    ''' Load the dataset and save the information
    '''
    mfile = open(prefix+filename, 'rb')
    mdata = pickle.load(mfile)
    mfile.close()
    params = mdata['best_params']
    n_step = mdata['training_parameters'].get('n_steps', mdata['training_parameters'].get('num_steps', None))
    method = mdata['training_parameters']['method']
    order = mdata['training_parameters']['order']
    return {'params' : params, 'n_step' : n_step, 'method' : method, 'order' : order, 'others' : mdata}


def run_prederr(seed):
    # -- seed 201
    m_rng  = jax.random.PRNGKey(seed)
    m_rng, subkey = jax.random.split(m_rng)

    n_state=2

    # Different filenames to use
    naming_convention ='{}_stifflinear_dt{}_{}_o{}_s{}_res.pkl'
    maxpoints = 100
    duration = 1.5 # In seconds
    num_traj = 500
    savefig = True

    # Generate a set of trajectories for comparing the solutions
    init_states = jax.random.uniform(subkey, shape=(num_traj,n_state), minval= jnp.array([-1.,-1.]), maxval=jnp.array([-0.9,-0.9]))

    dt_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    results_per_dt = list()
    for dt_1 in dt_list:
        #### Relative error for fixed time step
        traj_length = int(duration/dt_1)
        # Integrate the trajectory for some time horizom
        testTraj, _, = exact_solution(init_states, dt_1, traj_length, 1, merge_traj=False)

        m_files = [{'method' : 'rk4', 'order' : 1, 'n_step' : 1, 'init_module' : known_dyn_init},
                    {'method' : 'taylor', 'order' : 2, 'n_step' : 1, 'init_module' : known_dyn_init},
                    {'init_module' : known_dyn_init, **extract_params_logs(naming_convention.format('mid', dt_1, 'hypersolver', 1, 1))},
                    {'init_module' : known_dyn_init, **extract_params_logs(naming_convention.format('mid', dt_1, 'tayla', 1, 1))},
                    {'init_module' : known_dyn_init, **extract_params_logs(naming_convention.format('mid', dt_1, 'tayla', 2, 1))},
                    {'method' : 'dopri5', 'order' : 1, 'n_step' : 1, 'atol' : 1.e-12, 'rtol' : 1.e-12, 'init_module' : known_dyn_init}]
        m_labels = ['RK4', 'Taylor, o=2', 'Hyper, o=1', 'Tayla, o=1', 'Tayla, o=2', 'Dopri5',]
        m_colors = ['blue', 'orange', 'green', 'magenta', 'cyan', 'gray']
        spacing_val = int(traj_length / maxpoints) if traj_length > maxpoints else 1
        dt_results  = plot_prederror_dt(testTraj, dt_1, m_files, m_labels, m_colors, show=False, seed=seed, save=savefig, spacing=spacing_val, spacing_fill=spacing_val)
        results_per_dt.append(dt_results)


    # Print the computation time per time step to illustrate the approach
    cTime_mean, cTime_min, cTime_max = dict(), dict(), dict()
    cTimeColor = dict()
    for res_dt in results_per_dt:
        for key, (color, fmean_err, fmin_val, fmax_val, meant, mint, maxt) in res_dt.items():
            cTimeColor[key] = color
            if key not in cTime_mean:
                cTime_mean[key] = []
            if key not in cTime_min:
                cTime_min[key] = []
            if key not in cTime_max:
                cTime_max[key] = []
            cTime_mean[key].append(meant)
            cTime_min[key].append(mint)
            cTime_max[key].append(maxt) 

    tFig = plt.figure()
    for key, values in cTime_mean.items():
        # print(dt_list, values)
        plt.plot(np.array(dt_list), np.array(values), linewidth=2, linestyle='dashed',  marker="*", color=cTimeColor[key], label=key)
        plt.fill_between(np.array(dt_list), np.array(cTime_min[key]), np.array(cTime_max[key]), linewidth=2, facecolor=cTimeColor[key], alpha=0.6)
    plt.yscale('log')
    plt.xlabel(r'$\mathrm{Time \ step \ \Delta t}$')
    plt.ylabel(r'$\mathrm{Compute \ Time \ [s]}$')
    plt.grid(True)
    plt.legend(loc='best')
    if savefig:
        try:
            tikzplotlib.clean_figure()
        except:
            pass
        tikzplotlib.save('data/cTime_dt.tex')
        plt.savefig('data/cTime_dt.png', dpi=300, transparent=True)

    # Print the accuracy per time step
    cTime_mean, cTime_min, cTime_max = dict(), dict(), dict()
    cTimeColor = dict()
    for res_dt in results_per_dt:
        for key, (color, fmean_err, fmin_val, fmax_val, meant, mint, maxt) in res_dt.items():
            cTimeColor[key] = color
            if key not in cTime_mean:
                cTime_mean[key] = []
            if key not in cTime_min:
                cTime_min[key] = []
            if key not in cTime_max:
                cTime_max[key] = []
            cTime_mean[key].append(fmean_err)
            cTime_min[key].append(fmin_val)
            cTime_max[key].append(fmax_val) 

    tFig = plt.figure()
    for key, values in cTime_mean.items():
        # print(dt_list, values)
        plt.plot(np.array(dt_list), np.array(values), linewidth=2, linestyle='dashed',  marker="*", color=cTimeColor[key], label=key)
        plt.fill_between(np.array(dt_list), np.array(cTime_min[key]), np.array(cTime_max[key]), linewidth=2, facecolor=cTimeColor[key], alpha=0.6)
    plt.yscale('log')
    plt.xlabel(r'$\mathrm{Time \ step \ \Delta t}$')
    plt.ylabel(r'$\mathrm{Integration.\ Error}$')
    plt.grid(True)
    plt.legend(loc='best')
    if savefig:
        try:
            tikzplotlib.clean_figure()
        except:
            pass
        tikzplotlib.save('data/accuracy_dt.tex')
        plt.savefig('data/accuracy_dt.png', dpi=300, transparent=True)
    plt.show()

def run_plotlog():
    dt_1 = 0.01
    maxpoints = 50
    naming_convention ='{}_stifflinear_dt{}_{}_o{}_s{}_res.pkl'
    m_files = [{'init_module' : known_dyn_init, **extract_params_logs(naming_convention.format('dyn', dt_1, 'rk4', 1, 1))},
                {'init_module' : known_dyn_init, **extract_params_logs(naming_convention.format('dyn', dt_1, 'taylor', 2, 1))},
                {'init_module' : known_dyn_init, **extract_params_logs(naming_convention.format('dyn', dt_1, 'tayla', 1, 1))},
                {'init_module' : known_dyn_init, **extract_params_logs(naming_convention.format('dyn', dt_1, 'dopri5', 1, 1))},]
    m_labels = ['RK4', 'Taylor, o=2','Tayla, o=1', 'Dopri5',]
    m_colors = ['blue', 'orange', 'magenta', 'gray']
    plot_log_dt(dt_1, m_files, m_labels, m_colors, show=False, save=True, spacing=maxpoints, window=20)
    plt.show()


if __name__ == "__main__":
    ''' Main function where we plot all the examples
    '''
    import time
    import argparse
    # Command line argument for setting parameters
    # python generate_sample.py --dt 0.04
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_plotlog()

    


