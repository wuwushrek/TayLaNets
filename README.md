# Taylor Lagrange Deep Neural Networks

This module seeks to provide training algorithms and deep neural networks architectures able to:

- Learn a cheap solver for Neural ordinary equations

- Learn both the underlying differential equation (and how to solve it) of an unknown dynamcal system from sampled trajectories


# Installation

This package requires [``jax``](https://github.com/google/jax) to be installed: The choice of CPU or GPU version depends on the user but the CPU is installed by default along with the package.
The package further requires [``dm-haiku``](https://github.com/deepmind/dm-haiku) for neural networks in jax and [``optax``](https://github.com/deepmind/optax) a gradient processing and optimization library for JAX. The following commands install everything that is required (except for the GPU version of JAX which must be installed manually):

```
pip install numpy matplotlib scipy tqdm pyyaml
cd TayLaNets/
python3 -m pip install -e . 
```

# Examples

## Stiff Dynamics example

To reproduce the experiments on the linear stiff dynamics, execute the set of command below.
```
# Generate the data set
cd TayLaNets/examples_taylanets/stiff_dynamics
python generate_sample.py --dt 0.001 # Do the same for the other time steps consider in the paper.

# Learn midpoint
cd TayLaNets/examples_taylanets/stiff_dynamics
python learn_midpoint.py --train_batch_size 500 --test_batch_size 1000 --lr_init 1e-5 --lr_end 1e-10 --test_freq 5000 --save_freq 10000 --n_steps 1 --nepochs 500 --w_decay 0 --grad_clip 0 --method tayla --order 1 --atol 1e-8 --rtol 1e-8 --trajfile data/stifflinear_dt0.001.pkl

# Learn the dynamics and the midpoint
python generate_sample.py --dt 0.01

python learn_dynamics.py --train_batch_size 500 --test_batch_size 250 --lr_init 1e-2 --lr_end 1e-4 --test_freq 1000 --save_freq 20000 --n_steps 1 --nepochs 100 --w_decay 0 --grad_clip 0 --method tayla --atol 1e-8 --rtol 1e-8 --trajfile data/stifflinear_dt0.01.pkl --mid_freq_update 10 --pen_remainder 1e-3 --mid_lr_init 1e-4 --mid_lr_end 1e-12 
```

## MNIST
Run the following command
```
python tayla_mnist.py --train_batch_size 500 --test_batch_size 500 --lr_init 1e-2 --lr_end 1e-2 --test_freq 400 --save_freq 10000 --n_steps 2 --order 1 --pen_remainder 1e1 --nepochs 500 --method tayla --atol 1.4e-8 --rtol 1.4e-8 --grad_clip 0 --mid_freq_update 500 --mid_lr_init 1e-4 --mid_lr_end 1e-8 --grad_clip 0.001 --dur_ending_sched 100 --ending_lr_init 1e-4 --ending_lr_end 1e-5
```

## TABULAR FFJORD MINIBOONE
Run the following command
```
python tayla_ffjord_tabular.py --train_batch_size 1000 --nepochs 400 --n_steps 8 --method tayla --lr_init 1e-3 --lr_end 1e-3 --w_decay 0.001 --grad_clip 0. --pen_remainder 5e1 --order 1 --mid_freq_update 100 --mid_lr_init 1e-4 --mid_lr_end 1e-7 --dur_ending_sched 150 --ending_lr_init 1e-5 --ending_lr_end 1e-5
```
