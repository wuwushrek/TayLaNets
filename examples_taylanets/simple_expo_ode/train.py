import jax
import jax.numpy as jnp
import numpy as np

from taylanets.utils import SampleLog, HyperParamsNN, LearningLog, load_config_file
from taylanets.train import train

# The known dynamics of the system
from generate_sample import system_ode

if __name__ == "__main__":
	import time
	import argparse
	# python generate_sample.py --cfg reacher_brax/dataset_gen.yaml --output_file reacher_brax/testdata --seed 101 --disable_substep 0 --save_video 1
	# Command line argument for setting parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', required=True, type=str, help='yaml configuration file for training/testing information: see reacher_cfg1.yaml for more information')
	parser.add_argument('--input_file', type=str, default='')
	parser.add_argument('--output_file', type=str, default='')
	parser.add_argument('--batch_size', type=int, default=0)
	parser.add_argument('--num_grad', type=int, default=0)
	parser.add_argument('--normalize', type=int, default=-1)
	parser.add_argument('--baseline_name', type=str, default='')
	parser.add_argument('--baseline_order', type=int, default=-1)
	args = parser.parse_args()

	m_config_aux = {'cfg' : args.cfg}

	if args.input_file != '':
		m_config_aux['train_data_file']  = args.input_file

	if args.output_file != '':
		m_config_aux['output_file']  = args.output_file

	if args.batch_size > 0:
		m_config_aux['batch_size']  = args.batch_size

	if args.num_grad > 0:
		m_config_aux['num_gradient_iterations']  = args.num_grad

	if args.normalize >= 0:
		m_config_aux['normalize']  = args.normalize == 1

	if args.baseline_name != '':
		m_config_aux['baseline_name'] = args.baseline_name

	if args.baseline_order >= 0:
		m_config_aux['baseline_order'] = args.baseline_order

	# Load the config file
	mSampleLog, hyperParams, (out_file, seed_list) = load_config_file(args.cfg, m_config_aux)

	# Train the model based on the information from the config files
	train(mSampleLog, hyperParams, seed_list, out_file, known_dynamics=jax.vmap(system_ode))