# Taylor Lagrange Deep Neural Networks

This module seeks to provide training algorithms and deep neural networks architectures able to:

- Solve with outstanding accuracy ODE from trajectories (when the ODEs are known)

- Learn both the underlying differential equation (and how to solve it) of an unknwon dynamcal system from sampled trajectories


# Installation

This package requires [``jax``](https://github.com/google/jax) to be installed: The choice of CPU or GPU version depends on the user but the CPU is installed by default along with the package.
The package further requires [``dm-haiku``](https://github.com/deepmind/dm-haiku) for neural networks in jax and [``optax``](https://github.com/deepmind/optax) a gradient processing and optimization library for JAX. The following commands install everything that is required (except for the GPU version of JAX which must be installed manually):

```
pip install numpy matplotlib scipy tqdm pyyaml
git clone https://github.com/wuwushrek/TayLaNets.git
cd TayLaNets/
python3 -m pip install -e . 
```

# Examples

