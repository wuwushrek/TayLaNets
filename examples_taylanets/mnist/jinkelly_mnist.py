"""
Neural ODEs on MNIST with no downsampling before ODE, implemented with Haiku.
"""
import argparse
import collections
import os
import pickle
import time
from math import prod

import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from jax import lax
from jax.config import config
from jax.experimental import optimizers
from jax.experimental.jet import jet
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten

# from lib_ode.ode import odeint, odeint_aux_one, odeint_sepaux, odeint_grid, odeint_grid_sepaux_one, odeint_grid_aux
from taylanets.ode import odeint, odeint_aux_one, odeint_sepaux, odeint_grid, odeint_grid_sepaux_one, odeint_grid_aux

from tqdm.auto import tqdm

float64 = False
config.update("jax_enable_x64", float64)

REGS = ["r2", "r3", "r4", "r5", "r6"]

parser = argparse.ArgumentParser('Neural ODE MNIST')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--lam_w', type=float, default=0)
parser.add_argument('--atol', type=float, default=1.4e-8)
parser.add_argument('--rtol', type=float, default=1.4e-8)
parser.add_argument('--vmap', action="store_true")
parser.add_argument('--reg', type=str, choices=['none'] + REGS, default='none')
parser.add_argument('--test_freq', type=int, default=3000)
parser.add_argument('--save_freq', type=int, default=3000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lam_fro', type=float, default=0)
parser.add_argument('--lam_kin', type=float, default=0)
parser.add_argument('--reg_type', type=str, choices=['our', 'fin'], default='our')
parser.add_argument('--num_steps', type=int, default=2)

# Custom input parameters
parser.add_argument('--no_validation_set', action="store_true")
parser.add_argument('--trajdir', type=str, default='data/')
parser.add_argument('--grid', action="store_true")
parser.add_argument('--no_compare_odeint',  action="store_true")

parse_args = parser.parse_args()


# if not os.path.exists(parse_args.dirname):
    # os.makedirs(parse_args.dirname)

# set up config

reg = parse_args.reg
lam = parse_args.lam
lam_fro = parse_args.lam_fro
lam_kin = parse_args.lam_kin
reg_type = parse_args.reg_type
lam_w = parse_args.lam_w
seed = parse_args.seed
rng = jax.random.PRNGKey(seed)
# dirname = parse_args.dirname
# count_nfe = not parse_args.no_count_nfe
vmap = parse_args.vmap
grid = parse_args.grid
odeint_compare = not parse_args.no_compare_odeint

if grid:
    all_odeint = odeint_grid
    odeint_aux1 = odeint_grid_aux           # finlay trick w/ 1 augmented state
    odeint_aux2 = odeint_grid_sepaux_one    # odeint_grid_sepaux_onefinlay trick w/ 2 augmented states
    ode_kwargs = {
        "step_size": 1 / parse_args.num_steps
    }
else:
    all_odeint = odeint
    odeint_aux1 = odeint_aux_one
    odeint_aux2 = odeint_sepaux
    ode_kwargs = {
        "atol": parse_args.atol,
        "rtol": parse_args.rtol
    }

# # This is used to compute the number of function evaluations
ode_comp = odeint
ode_comp_args = {"atol": parse_args.atol, "rtol": parse_args.rtol}

# some primitive functions
def sigmoid(z):
    """
    Numerically stable sigmoid.
    """
    return 1/(1 + jnp.exp(-z))


def softmax_cross_entropy(logits, labels):
    """
    Cross-entropy loss applied to softmax.
    """
    one_hot = hk.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def sol_recursive(f, z, t):
    """
    Recursively compute higher order derivatives of dynamics of ODE.
    """
    if reg == "none":
        return f(z, t), jnp.zeros_like(z)

    z_shape = z.shape
    z_t = jnp.concatenate((jnp.ravel(z), jnp.array([t])))

    def g(z_t):
        """
        Closure to expand z.
        """
        z, t = jnp.reshape(z_t[:-1], z_shape), z_t[-1]
        dz = jnp.ravel(f(z, t))
        dt = jnp.array([1.])
        dz_t = jnp.concatenate((dz, dt))
        return dz_t

    reg_ind = REGS.index(reg)

    (y0, [*yns]) = jet(g, (z_t, ), ((jnp.ones_like(z_t), ), ))
    for _ in range(reg_ind + 1):
        (y0, [*yns]) = jet(g, (z_t, ), ((y0, *yns), ))

    return (jnp.reshape(y0[:-1], z_shape), jnp.reshape(yns[-2][:-1], z_shape))


# set up modules
class Flatten(hk.Module):
    """
    Flatten all dimensions except batch dimension.
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def __call__(self, x):
        return jnp.reshape(x, (x.shape[0], -1))


class ConcatConv2D(hk.Module):
    """
    Convolution with extra channel and skip connection for time.
    """

    def __init__(self, **kwargs):
        super(ConcatConv2D, self).__init__()
        self._layer = hk.Conv2D(**kwargs)

    def __call__(self, x, t):
        tt = jnp.ones_like(x[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, x], axis=-1)
        return self._layer(ttx)


def get_epsilon(key, shape):
    """
    Sample epsilon from the desired distribution.
    """
    # rademacher
    if float64:
        return jax.random.randint(key, shape, minval=0, maxval=2).astype(jnp.float64) * 2 - 1
    else:
        return jax.random.randint(key, shape, minval=0, maxval=2).astype(jnp.float32) * 2 - 1


class PreODE(hk.Module):
    """
    Module applied before the ODE layer.
    """

    def __init__(self):
        super(PreODE, self).__init__()
        if float64:
            self.model = hk.Sequential([
                lambda x: x.astype(jnp.float64) / 255.,
                Flatten()
            ])
        else:
            self.model = hk.Sequential([
                lambda x: x.astype(jnp.float32) / 255.,
                Flatten()
            ])

    def __call__(self, x):
        return self.model(x)


class MLPDynamics(hk.Module):
    """
    Dynamics for ODE as an MLP.
    """

    def __init__(self, input_shape):
        super(MLPDynamics, self).__init__()
        self.input_shape = input_shape
        self.dim = prod(input_shape[1:])
        self.hidden_dim = 100
        self.lin1 = hk.Linear(self.hidden_dim)
        self.lin2 = hk.Linear(self.dim)

    def __call__(self, x, t):
        # vmapping means x will be a single batch element, so need to expand dims at 0
        x = jnp.reshape(x, (-1, self.dim))

        out = sigmoid(x)
        tt = jnp.ones_like(x[:, :1]) * t
        t_out = jnp.concatenate([tt, out], axis=-1)
        out = self.lin1(t_out)

        out = sigmoid(out)
        tt = jnp.ones_like(out[:, :1]) * t
        t_out = jnp.concatenate([tt, out], axis=-1)
        out = self.lin2(t_out)

        return out


class PostODE(hk.Module):
    """
    Module applied after the ODE layer.
    """

    def __init__(self):
        super(PostODE, self).__init__()
        self.model = hk.Sequential([
            sigmoid,
            hk.Linear(10)
        ])

    def __call__(self, x):
        return self.model(x)


def wrap_module(module, *module_args, **module_kwargs):
    """
    Wrap the module in a function to be transformed.
    """
    def wrap(*args, **kwargs):
        """
        Wrapping of module.
        """
        model = module(*module_args, **module_kwargs)
        return model(*args, **kwargs)
    return wrap


def initialization_data(input_shape, ode_shape):
    """
    Data for initializing the modules.
    """
    ode_shape = (1, ) + ode_shape[1:]
    ode_dim = prod(ode_shape)
    data = {
        "pre_ode": jnp.zeros(input_shape),
        "ode": (jnp.zeros(ode_dim), 0.),
        "post_ode": jnp.zeros(ode_dim)
    }
    return data


def init_model():
    """
    Instantiates transformed submodules of model and their parameters.
    """
    ts = jnp.array([0., 1.])

    input_shape = (1, 28, 28, 1)
    ode_shape = (-1, 28, 28, 1)

    initialization_data_ = initialization_data(input_shape, ode_shape)

    pre_ode = hk.without_apply_rng(hk.transform(wrap_module(PreODE)))
    pre_ode_params = pre_ode.init(rng, initialization_data_["pre_ode"])
    pre_ode_fn = pre_ode.apply

    dynamics = hk.without_apply_rng(hk.transform(wrap_module(MLPDynamics, ode_shape)))
    dynamics_params = dynamics.init(rng, *initialization_data_["ode"])
    dynamics_wrap = lambda x, t, params: dynamics.apply(params, x, t)

    def reg_dynamics(y, t, params):
        """
        Dynamics of regularization for ODE integration.
        """
        y0, r = sol_recursive(lambda _y, _t: dynamics_wrap(_y, _t, params), y, t)
        return y0, jnp.mean(jnp.square(r), axis=[axis_ for axis_ in range(1, r.ndim)])

    def fin_dynamics(y, t, eps, params):
        """
        Dynamics of finlay reg.
        """
        f = lambda y: dynamics_wrap(y, t, params)
        dy, eps_dy = jax.jvp(f, (y,), (eps,))
        return dy, eps_dy

    def aug_dynamics(yr, t, eps, params):
        """
        Dynamics augmented with regularization.
        """
        y, *_ = yr
        if reg_type == "our":
            return reg_dynamics(y, t, params)
        else:
            dy, eps_dy = fin_dynamics(y, t, eps, params)
            dfro = jnp.mean(jnp.square(eps_dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
            dkin = jnp.mean(jnp.square(dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
            return dy, dfro, dkin

    def all_aug_dynamics(yr, t, eps, params):
        """
        Dynamics augmented with all regularizations for tracking.
        """
        y, *_ = yr
        dy, eps_dy = fin_dynamics(y, t, eps, params)
        _, drdt = reg_dynamics(y, t, params)
        dfro = jnp.mean(jnp.square(eps_dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
        dkin = jnp.mean(jnp.square(dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
        return dy, drdt, dfro, dkin

    if reg_type == "our":
        _odeint = odeint_aux1
    else:
        _odeint = odeint_aux2
    nodeint_aux = lambda y0, ts, eps, params: \
        _odeint(lambda y, t, eps, params: dynamics_wrap(y, t, params),
                aug_dynamics, y0, ts, eps, params, **ode_kwargs)[0]
    all_nodeint = lambda y0, ts, eps, params: all_odeint(all_aug_dynamics,
                                                          y0, ts, eps, params, **ode_kwargs)[0]

    def ode(params, out_pre_ode, eps):
        """
        Apply the ODE block.
        """
        out_ode, *out_ode_rs = nodeint_aux(aug_init(out_pre_ode), ts, eps, params)
        return (out_ode[-1], *(out_ode_r[-1] for out_ode_r in out_ode_rs))

    def all_ode(params, out_pre_ode, eps):
        """
        Apply ODE block for all regularizations.
        """
        out_ode, *out_ode_rs = all_nodeint(all_aug_init(out_pre_ode), ts, eps, params)
        return (out_ode[-1], *(out_ode_r[-1] for out_ode_r in out_ode_rs))


    post_ode = hk.without_apply_rng(hk.transform(wrap_module(PostODE)))
    post_ode_params = post_ode.init(rng, initialization_data_["post_ode"])
    post_ode_fn = post_ode.apply

    # return a dictionary of the three components of the model
    model = {"model": {
        "pre_ode": pre_ode_fn,
        "ode": ode,
        "post_ode": post_ode_fn,
        "all_ode": all_ode
    }, "params": {
        "pre_ode": pre_ode_params,
        "ode": dynamics_params,
        "post_ode": post_ode_params
    }, "nfe": None
    }
    @jax.jit
    def forward(key, params, _images):
        """
        Forward pass of the model.
        """
        model_ = model["model"]

        out_pre_ode = model_["pre_ode"](params["pre_ode"], _images)
        out_ode, *regs = model_["ode"](params["ode"], out_pre_ode, get_epsilon(key, out_pre_ode.shape))
        logits = model_["post_ode"](params["post_ode"], out_ode)

        return (logits, *regs)

    def forward_all(key, params, _images):
        """
        Forward pass of the model.
        """
        model_ = model["model"]

        out_pre_ode = model_["pre_ode"](params["pre_ode"], _images)
        out_ode, *regs = model_["all_ode"](params["ode"], out_pre_ode, get_epsilon(key, out_pre_ode.shape))
        logits = model_["post_ode"](params["post_ode"], out_ode)

        return (logits, *regs)

    model["model"]["forward_all"] = forward_all
    model["model"]["forward"] = forward

    if odeint_compare:
        assert not vmap, 'vmap should be False'
        unreg_nodeint = lambda y0, t, params: ode_comp(dynamics_wrap, y0, t, params, **ode_comp_args)

        @jax.jit
        def nfe_fn(params, _images, _labels):
            """
            Function to return NFE.
            """
            in_ode = pre_ode_fn(params["pre_ode"], _images)
            out_ode, f_nfe = unreg_nodeint(in_ode, ts, params["ode"])
            logits = post_ode_fn(params["post_ode"], out_ode[-1])
            return out_ode[-1], logits, jnp.mean(f_nfe)

    else:
        nfe_fn = None

    # Save the number of function evaluations
    model["nfe"] = nfe_fn

    return forward, model


def aug_init(y, batch_size=-1):
    """
    Flatten the dynamics and append regularization dynamics.
    We need to flatten the dynamics first since they may be convolutional
    (has width, height, and channels).
    """
    if batch_size == -1:
        batch_size = y.shape[0]
    if reg_type == "our":
        return y, jnp.zeros(batch_size)
    else:
        return y, jnp.zeros(batch_size), jnp.zeros(batch_size)


def all_aug_init(y, batch_size=-1):
    """
    Flatten the dynamics and append regularization dynamics.
    We need to flatten the dynamics first since they may be convolutional
    (has width, height, and channels).
    """
    if batch_size == -1:
        batch_size = y.shape[0]
    return y, jnp.zeros(batch_size), jnp.zeros(batch_size), jnp.zeros(batch_size)

@jax.jit
def _acc_fn(logits, labels):
    """
    Classification accuracy of the model.
    """
    predicted_class = jnp.argmax(logits, axis=1)
    return jnp.mean(predicted_class == labels)

@jax.jit
def _loss_fn(logits, labels):
    return jnp.mean(softmax_cross_entropy(logits, labels))


def _reg_loss_fn(reg):
    return jnp.mean(reg)


def _weight_fn(params):
    flat_params, _ = ravel_pytree(params)
    return 0.5 * jnp.sum(jnp.square(flat_params))


def loss_fn(forward, params, images, labels, key):
    """
    The loss function for training.
    """
    if reg_type == "our":
        logits, regs = forward(key, params, images)
        loss_ = _loss_fn(logits, labels)
        reg_ = _reg_loss_fn(regs)
        weight_ = _weight_fn(params)
        return loss_ + lam * reg_ + lam_w * weight_
    else:
        logits, fro_regs, kin_regs = forward(key, params, images)
        loss_ = _loss_fn(logits, labels)
        fro_reg_ = _reg_loss_fn(fro_regs)
        kin_reg_ = _reg_loss_fn(kin_regs)
        weight_ = _weight_fn(params)
        return loss_ + lam_fro * fro_reg_ + lam_kin * kin_reg_ + lam_w * weight_


# Define a function to compute relative error
@jax.jit
def _rel_error(eststate, truestate):
    """ Compute the relative error between two quantities
    """
    return jnp.sum(jnp.abs(eststate - truestate)) / jnp.sum(jnp.abs(eststate) + jnp.abs(truestate))

### Custom function for uploading the data set
def init_data(train_batch_size, test_batch_size, seed_number=0, shuffle=1000, validation_set=True):
    """
    Initialize data from tensorflow dataset
    """
    # Import and cache the file in the current directory
    ds_data, ds_info = tfds.load('mnist',
                                     data_dir='data/tensorflow_data/',
                                     split=['train'] if not validation_set else ['train', 'test'],
                                     shuffle_files=True,
                                     as_supervised=True,
                                     with_info=True,
                                     read_config=tfds.ReadConfig(shuffle_seed=seed_number, try_autocache=False))

    if validation_set:
        ds_train, ds_test = ds_data
    else:
        ds_train, ds_test = ds_data[0], ds_data[0]

    num_train = ds_info.splits['train'].num_examples
    num_test = ds_info.splits['test'].num_examples if validation_set else num_train

    assert num_train % train_batch_size == 0
    num_train_batches = num_train // train_batch_size

    assert num_test % test_batch_size == 0
    num_test_batches = num_test // test_batch_size

    # Make the data set loopable mainly for testing loss evalutaion
    ds_train = ds_train.cache()
    ds_train = ds_train.repeat()
    ds_train = ds_train.shuffle(shuffle, seed=seed_number)

    # ds_test = ds_test.cache()
    # ds_test = ds_test.repeat()
    # ds_test = ds_test.shuffle(shuffle, seed=seed_number)

    ds_train, ds_test = ds_train.batch(train_batch_size), ds_test.batch(test_batch_size).repeat()
    ds_train, ds_test = tfds.as_numpy(ds_train), tfds.as_numpy(ds_test)

    meta = {
        "num_train_batches": num_train_batches,
        "num_test_batches": num_test_batches,
        "num_train" : num_train,
        "num_test" : num_test
    }

    # Return iter element on the training and testing set
    return iter(ds_train), iter(ds_test), meta


def run():
    """
    Run the experiment.
    """
    ds_train, ds_train_eval, meta = init_data(parse_args.batch_size, parse_args.test_batch_size, seed_number=parse_args.seed, validation_set= not parse_args.no_validation_set)
    num_batches = meta["num_train_batches"]
    num_test_batches = meta["num_test_batches"]

    forward, model = init_model()
    forward_all = model["model"]["forward_all"]
    grad_fn = jax.grad(lambda *args: loss_fn(forward, *args))

    def lr_schedule(train_itr):
        """
        The learning rate schedule.
        """
        _epoch = train_itr // num_batches
        id = lambda x: x
        return lax.cond(_epoch < 60, 1e-1, id, 0,
                        lambda _: lax.cond(_epoch < 100, 1e-2, id, 0,
                                           lambda _: lax.cond(_epoch < 140, 1e-3, id, 1e-4, id)))

    opt_init, opt_update, get_params = optimizers.momentum(step_size=lr_schedule, mass=0.9)

    init_params = model["params"]
    load_itr = 0

    opt_state = opt_init(init_params)

    @jax.jit
    def update(_itr, _opt_state, _key, _batch):
        """
        Update the params based on grad for current batch.
        """
        images, labels = _batch
        return opt_update(_itr, grad_fn(get_params(_opt_state), images, labels, _key), _opt_state)

    @jax.jit
    def sep_losses(_opt_state, _batch, key):
        """
        Convenience function for calculating losses separately.
        """
        params = get_params(_opt_state)
        images, labels = _batch
        logits, r2_regs, fro_regs, kin_regs = forward_all(key, params, images)
        loss_ = _loss_fn(logits, labels)
        r2_reg_ = _reg_loss_fn(r2_regs)
        fro_reg_ = _reg_loss_fn(fro_regs)
        kin_reg_ = _reg_loss_fn(kin_regs)
        total_loss_ = loss_ + lam * r2_reg_ + lam_fro * fro_reg_ + lam_kin * kin_reg_
        acc_ = _acc_fn(logits, labels)
        return acc_, total_loss_, loss_, r2_reg_, fro_reg_, kin_reg_

    def evaluate_loss(opt_state, _key, ds_train_eval, m_batches, fwd_odeint=None):
        """
        Convenience function for evaluating loss over train set in smaller batches.
        """
        sep_acc_, sep_loss_aug_, sep_loss_, \
        sep_loss_r2_reg_, sep_loss_fro_reg_, sep_loss_kin_reg_, predtime = [], [], [], [], [], [], []
        nfe_odeint, odeint_loss, odeint_acc, odeint_relerr, odeint_time = [], [], [], [], []

        for test_batch_num in tqdm(range(m_batches), leave=False):
            test_batch = next(ds_train_eval)
            _key, = jax.random.split(_key, num=1)

            curr_time = time.time()
            l, *_ = forward(_key, get_params(opt_state), test_batch[0])
            l.block_until_ready()
            m_predtime = time.time() - curr_time

            test_batch_acc_, test_batch_loss_aug_, test_batch_loss_, \
            test_batch_loss_r2_reg_, test_batch_loss_fro_reg_, test_batch_loss_kin_reg_ = \
                sep_losses(opt_state, test_batch, _key)

            if fwd_odeint is not None and odeint_compare:
                curr_time = time.time()
                out_ode, logits, nfe_est = model["nfe"](get_params(opt_state), *test_batch)
                logits.block_until_ready()
                m_odetime = time.time() - curr_time
                loss_ = _loss_fn(logits, test_batch[1])
                acc_ = _acc_fn(logits, test_batch[1])
                nfe_odeint.append(nfe_est)
                odeint_loss.append(loss_)
                odeint_acc.append(acc_)
                odeint_relerr.append(_rel_error(l, logits))
                odeint_time.append(m_odetime)
            else:
                nfe_odeint.append(-1)
                odeint_loss.append(-1) 
                odeint_acc.append(-1)
                odeint_relerr.append(-1)
                odeint_time.append(-1)

            sep_acc_.append(test_batch_acc_)
            sep_loss_aug_.append(test_batch_loss_aug_)
            sep_loss_.append(test_batch_loss_)
            sep_loss_r2_reg_.append(test_batch_loss_r2_reg_)
            sep_loss_fro_reg_.append(test_batch_loss_fro_reg_)
            sep_loss_kin_reg_.append(test_batch_loss_kin_reg_)
            predtime.append(m_predtime)

        sep_acc_ = jnp.array(sep_acc_)
        sep_loss_aug_ = jnp.array(sep_loss_aug_)
        sep_loss_ = jnp.array(sep_loss_)
        sep_loss_r2_reg_ = jnp.array(sep_loss_r2_reg_)
        sep_loss_fro_reg_ = jnp.array(sep_loss_fro_reg_)
        sep_loss_kin_reg_ = jnp.array(sep_loss_kin_reg_)
        # nfe = jnp.array(nfe)

        nfe_odeint = jnp.array(nfe_odeint)
        odeint_loss = jnp.array(odeint_loss)
        odeint_acc = jnp.array(odeint_acc)
        odeint_relerr = jnp.array(odeint_relerr)
        odeint_time = jnp.array(odeint_time)

        return jnp.mean(sep_acc_), jnp.mean(sep_loss_aug_), jnp.mean(sep_loss_), \
               jnp.mean(sep_loss_r2_reg_), jnp.mean(sep_loss_fro_reg_), jnp.mean(sep_loss_kin_reg_), jnp.mean(jnp.array(predtime)),\
               (jnp.mean(nfe_odeint), jnp.mean(odeint_loss), jnp.mean(odeint_acc), jnp.mean(odeint_relerr), jnp.mean(odeint_time))

    itr = 0
    info = collections.defaultdict(dict)

    key = rng

    # Save the optimal loss_value obtained
    opt_params_dict = None
    opt_loss_train = None
    opt_accuracy_train = None

    opt_loss_test = None
    opt_loss_odeint_test = None
    opt_accuracy_test = None
    opt_accuracy_odeint_test = None

    opt_relerr_odeint_test = None
    opt_predtime_test = None
    opt_predtime_test_odeint = None

    total_compute_time = 0.0
    loss_evol_train = list()
    loss_evol_test = list()
    loss_evol_odeint_test = list()
    train_accuracy = list()
    test_accuracy = list()
    test_accuracy_odeint = list()
    predtime_evol_train = list()
    predtime_evol_test = list()
    predtime_evol_odeint_test = list()
    compute_time_update = list()
    nfe_evol_test = list()
    err_evol_odeint = list()

    m_parameters_dict = vars(parse_args)
    trajdir = parse_args.trajdir
    out_data_file = trajdir +'dyn_mnist_{}_s{}_R{}_G{}'.format(parse_args.reg_type, parse_args.num_steps, parse_args.reg, int(grid))

    # Open the info file to save the command line print
    outfile = open(out_data_file+'_info.txt', 'w')
    outfile.write('Training parameters: {} \n'.format(m_parameters_dict))
    outfile.write('////// Command line messages \n\n\n')
    outfile.close()

    for epoch in tqdm(range(parse_args.nepochs)):
        for i in tqdm(range(num_batches), leave=False):
            batch = next(ds_train)

            key, = jax.random.split(key, num=1)

            itr += 1

            update_start = time.time()
            opt_state = update(itr, opt_state, key, batch)
            tree_flatten(opt_state)[0][0].block_until_ready()
            update_end = time.time() - update_start

            # [LOG] Update the compute time data
            compute_time_update.append(update_end)

            # [LOG] Total elapsed compute time for update only
            if itr >= 5: # Remove the first few steps due to jit compilation
                total_compute_time += update_end

            if itr % parse_args.test_freq == 0:
                print_str_test = '--------------------------------- Eval on Test Data [epoch={} | num_batch = {}] ---------------------------------\n'.format(epoch, i)
                tqdm.write(print_str_test)

                acc_, loss_aug_, loss_, loss_r2_reg_, loss_fro_reg_, loss_kin_reg_, predtime_, _ = \
                    evaluate_loss(opt_state, key, ds_train, meta['num_train_batches'], fwd_odeint=None)

                # Compute the loss on the testing set if it is different from the training set
                if not parse_args.no_validation_set:
                    acc_test_, loss_aug_test_, loss_test_, loss_r2_reg_test_, loss_fro_reg_test_, loss_kin_reg_test_, predtime_test_,\
                    (nfe_test_, odeint_loss, odeint_acc, odeint_relerr, odeint_time) = evaluate_loss(opt_state, key, ds_train_eval, meta['num_test_batches'], fwd_odeint=True)
                else:
                    acc_test_, loss_aug_test_, loss_test_, loss_r2_reg_test_, loss_fro_reg_test_, loss_kin_reg_test_, nfe_test_, predtime_test_,  = acc_, loss_aug_, loss_, loss_r2_reg_, loss_fro_reg_, loss_kin_reg_, -1., predtime_
                    odeint_loss, odeint_acc, odeint_relerr, odeint_time = [-1] * 4

                # First time we have a value for the loss function
                if opt_loss_train is None or opt_loss_test is None or (opt_accuracy_test < acc_test_):
                    opt_loss_test = loss_test_
                    opt_loss_odeint_test = odeint_loss
                    opt_loss_train = loss_
                    opt_accuracy_test = acc_test_
                    opt_accuracy_odeint_test = odeint_acc
                    opt_accuracy_train = acc_

                    opt_relerr_odeint_test = odeint_relerr
                    opt_nfe_test = nfe_test_
                    opt_predtime_test = predtime_test_
                    opt_predtime_test_odeint = odeint_time

                    opt_params_dict = get_params(opt_state)

                # Do some printing for result visualization
                print_str = 'Iter {:05d} | Total Update Time {:.2e} | Update time {:.2e}\n'.format(itr, total_compute_time, update_end)
                print_str += '[    Train     ] Loss = {:.2e} | Accuracy    = {:.3e} | Pred. Time. = {:.2e} | NFE = {:.3e} |  Fro. reg. = {:.3e} | Kin. reg. = {:.3e} | R2. reg. = {:.3e}\n'.format(loss_, acc_, predtime_, -1., loss_fro_reg_,loss_kin_reg_, loss_r2_reg_)
                print_str += '[    Test      ] Loss = {:.2e} | Accuracy    = {:.3e} | Pred. Time. = {:.2e} | NFE = {:.3e} |  Fro. reg. = {:.3e} | Kin. reg. = {:.3e} | R2. reg. = {:.3e}\n'.format(loss_test_, acc_test_, predtime_test_,-1.,loss_fro_reg_test_,loss_kin_reg_test_,loss_r2_reg_test_)
                print_str += '[  ODEINT Test ] Loss = {:.2e} | Accuracy    = {:.3e} | Pred. Time. = {:.2e} | NFE = {:.3e} | Diff. Pred.  = {:.2e}\n'.format(odeint_loss, odeint_acc, odeint_time, nfe_test_, odeint_relerr)
                print_str += '[  OPT. Value. ] Loss Train = {:.2e} | Loss Test = {:.2e} | Acc Train = {:.3e} | Acc Test = {:.3e} | Acc Test ODEINT = {:.3e}\n'.format(
                                opt_loss_train, opt_loss_test, opt_accuracy_train, opt_accuracy_test, opt_accuracy_odeint_test)
                print_str += '                 Loss ODE   = {:.2e} | NFE ODEINT = {:.3e} | Diff. Pred.  = {:.2e} | Pred. Time = {:.2e} | Pred. Time. ODEINT = {:.2e}\n'.format(opt_loss_odeint_test, opt_nfe_test, opt_relerr_odeint_test, opt_predtime_test, opt_predtime_test_odeint)

                tqdm.write(print_str)

                # Save all the obtained data
                loss_evol_train.append(loss_)
                loss_evol_test.append(loss_test_)
                loss_evol_odeint_test.append(odeint_loss)

                train_accuracy.append(acc_)
                test_accuracy.append(acc_test_)
                test_accuracy_odeint.append(odeint_acc)

                predtime_evol_train.append(predtime_)
                predtime_evol_test.append(predtime_test_)
                predtime_evol_odeint_test.append(odeint_time)

                err_evol_odeint.append(odeint_relerr)

                nfe_evol_test.append(nfe_test_)

                # Save these info in a file
                outfile = open(out_data_file+'_info.txt', 'a')
                outfile.write(print_str_test)
                outfile.write(print_str)
                outfile.close()

            if itr % parse_args.save_freq == 0 or (epoch == parse_args.nepochs-1 and i == meta['num_train_batches']-1):
                m_dict_res = {'best_params' : opt_params_dict, 'total_update_time' : total_compute_time, 'updatetime_evol' : compute_time_update,
                                'opt_loss_train' : opt_loss_train, 'opt_loss_test' : opt_loss_test, 
                                'opt_accuracy_test' : opt_accuracy_test, 'opt_accuracy_train' : opt_accuracy_train, 'opt_accuracy_odeint_test' : opt_accuracy_odeint_test,
                                'opt_nfe_test' : opt_nfe_test, 'opt_diff_test' : opt_relerr_odeint_test,  'opt_loss_odeint_test' : opt_loss_odeint_test,
                                'opt_predtime_test' : opt_predtime_test, 'opt_predtime_test_odeint' : opt_predtime_test_odeint,

                                'loss_evol_train' : loss_evol_train, 'loss_evol_test' : loss_evol_test,
                                'accuracy_evol_train' : train_accuracy, 'accuracy_evol_test' : test_accuracy, 'accuracy_evol_odeint' : test_accuracy_odeint,

                                'predtime_evol_train' : predtime_evol_train, 'predtime_evol_test' : predtime_evol_test, 'predtime_evol_odeint_test' : predtime_evol_odeint_test,
                                'nfe_evol_odeint' : nfe_evol_test, 'loss_evol_odeint' : loss_evol_odeint_test, 'err_evol_odeint' : err_evol_odeint,
                                'training_parameters' : m_parameters_dict}
                outfile = open(out_data_file+'_res.pkl', "wb")
                pickle.dump(m_dict_res, outfile)
                outfile.close()


if __name__ == "__main__":
    run()