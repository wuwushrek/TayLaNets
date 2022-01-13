"""
FFJORD on Tabular, implemented with Haiku.
"""
import argparse
import collections
import os
import pickle
import sys
import time

import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax
from jax.config import config
from jax.experimental import optimizers
from jax.experimental.jet import jet
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten

import datasets
from taylanets.ode import odeint, odeint_sepaux, odeint_grid, odeint_grid_sepaux, odeint_grid_sepaux_one

from tqdm.auto import tqdm

float_64 = False

config.update("jax_enable_x64", float_64)

parser = argparse.ArgumentParser('FFJORD Tabular MNIST')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--nepochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--lam_w', type=float, default=1e-6)
parser.add_argument('--atol', type=float, default=1.4e-8)
parser.add_argument('--rtol', type=float, default=1.4e-8)
parser.add_argument('--vmap', action="store_true")
parser.add_argument('--reg', type=str, choices=['none', 'r2', 'r3', 'r4'], default='none')
parser.add_argument('--test_freq', type=int, default=300)
parser.add_argument('--save_freq', type=int, default=300)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lam_fro', type=float, default=0)
parser.add_argument('--lam_kin', type=float, default=0)
parser.add_argument('--reg_type', type=str, choices=['our', 'fin'], default='our')
parser.add_argument('--num_steps', type=int, default=2)

# Custom input parameters
parser.add_argument('--trajdir', type=str, default='data/')
parser.add_argument('--grid', action="store_true")
parser.add_argument('--no_compare_odeint',  action="store_true")


parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hdim_factor', type=int, default=20)
parser.add_argument('--nonlinearity', type=str, default="softplus")

parse_args = parser.parse_args()

reg = parse_args.reg
lam = parse_args.lam
lam_fro = parse_args.lam_fro
lam_kin = parse_args.lam_kin
reg_type = parse_args.reg_type
lam_w = parse_args.lam_w
seed = parse_args.seed
rng = jax.random.PRNGKey(seed)
vmap = parse_args.vmap
grid = parse_args.grid
odeint_compare = not parse_args.no_compare_odeint

if grid:
    _odeint = odeint_grid
    _odeint_aux2 = odeint_grid_sepaux_one   # finlay trick w/ 2 augmented states
    _odeint_aux3 = odeint_grid_sepaux       # finlay trick w/ 3 augmented states
    ode_kwargs = {
        "step_size": 1 / parse_args.num_steps
    }
else:
    _odeint = odeint
    _odeint_aux2 = odeint_sepaux
    _odeint_aux3 = odeint_sepaux            # TODO: this will break for fin, but we shouldn't use it anyway
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

def _logaddexp(x1, x2):
    """
    Logad    dexp while ignoring the custom_jvp rule.
    """
    amax = lax.max(x1, x2)
    delta = lax.sub(x1, x2)
    return lax.select(jnp.isnan(delta),
                    lax.add(x1, x2),  # NaNs or infinities of the same sign.
                    lax.add(amax, lax.log1p(lax.exp(-lax.abs(delta)))))


softplus = lambda x: _logaddexp(x, jnp.zeros_like(x))

nonlinearity = softplus if parse_args.nonlinearity == "softplus" else jnp.tanh


def sol_recursive(f, z, t):
    """
    Recursively compute higher order derivatives of dynamics of ODE.
    """
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

    (y0, [y1h]) = jet(g, (z_t, ), ((jnp.ones_like(z_t), ), ))
    (y0, [y1, y2h]) = jet(g, (z_t, ), ((y0, y1h,), ))

    return (jnp.reshape(y0[:-1], z_shape), [jnp.reshape(y1[:-1], z_shape)])

# set up modules
class ConcatSquashLinear(hk.Module):
    """
    ConcatSquash Linear layer.
    """
    def __init__(self, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = hk.Linear(dim_out)
        self._hyper_bias = hk.Linear(dim_out, with_bias=False)
        self._hyper_gate = hk.Linear(dim_out)

    def __call__(self, x, t):
        return self._layer(x) * sigmoid(self._hyper_gate(jnp.reshape(t, (1, 1)))) \
               + self._hyper_bias(jnp.reshape(t, (1, 1)))


def get_epsilon(key, shape):
    """
    Sample epsilon from the desired distribution.
    """
    # normal
    return jax.random.normal(key, shape)


class NN_Dynamics(hk.Module):
    """
    NN_Dynamics of the ODENet.
    """

    def __init__(self,
                 hidden_dims,
                 input_shape):
        super(NN_Dynamics, self).__init__()
        self.input_shape = input_shape
        layers = []
        activation_fns = []
        base_layer = ConcatSquashLinear

        for dim_out in hidden_dims + (input_shape[-1], ):
            layer = base_layer(dim_out)
            layers.append(layer)
            activation_fns.append(nonlinearity)

        self.layers = layers
        self.activation_fns = activation_fns[:-1]

    def __call__(self, x, t):
        x = jnp.reshape(x, (-1, *self.input_shape))
        dx = x
        for l, layer in enumerate(self.layers):
            dx = layer(dx, t)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx


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


def initialization_data(input_shape):
    """
    Data for initializing the modules.
    """
    input_shape = (parse_args.test_batch_size, ) + input_shape[1:]
    data = {
        "ode": aug_init(jnp.zeros(input_shape))[:1] + (0., )  # (z, t)
    }
    return data


def init_model(n_dims):
    """
    Instantiates transformed submodules of model and their parameters.
    """
    ts = jnp.array([0., 1.])

    input_shape = (-1, n_dims)

    initialization_data_ = initialization_data(input_shape)

    dynamics = hk.without_apply_rng(hk.transform(wrap_module(NN_Dynamics,
                                                             input_shape=input_shape[1:],
                                                             hidden_dims=(n_dims * parse_args.hdim_factor, )
                                                                         * parse_args.num_layers)))
    dynamics_params = dynamics.init(rng, *initialization_data_["ode"])
    dynamics_wrap = lambda x, t, params: dynamics.apply(params, x, t)

    def reg_dynamics(y, t, params):
        """
        NN_Dynamics of regularization for ODE integration.
        """
        if reg == "none":
            y = jnp.reshape(y, input_shape)
            return jnp.zeros((y.shape[0], 1))
        else:
            # do r3 regularization
            y0, y_n = sol_recursive(lambda _y, _t: dynamics_wrap(_y, _t, params), y, t)
            r = y_n[-1]
            return jnp.mean(jnp.square(r), axis=[axis_ for axis_ in range(1, r.ndim)])

    def ffjord_dynamics(yp, t, eps, params):
        """
        Dynamics of augmented ffjord state.
        """
        y, p = yp
        f = lambda y: dynamics_wrap(y, t, params)
        dy, eps_dy = jax.jvp(f, (y,), (eps,))
        div = jnp.sum(jnp.reshape(eps_dy * eps, (y.shape[0], -1)), axis=1, keepdims=True)
        return dy, -div

    def ffjord2_dynamics(yp, t, eps, params):
        """
        Dynamics of augmented ffjord state.
        """
        y, p = yp
        f = lambda y: dynamics_wrap(y, t, params)
        dy, eps_dy = jax.jvp(f, (y,), (eps,))
        div = jnp.sum(jnp.reshape(eps_dy * eps, (y.shape[0], -1)), axis=1, keepdims=True)
        return dy, -div, eps_dy

    def aug_dynamics(ypr, t, eps, params):
        """
        NN_Dynamics augmented with logp and regularization.
        """
        y, p, *_ = ypr

        dy, dp, eps_dy = ffjord2_dynamics((y, p), t, eps, params)

        if reg_type == "our":
            dr = reg_dynamics(y, t, params)
            return dy, dp, dr
        else:
            dfro = jnp.mean(jnp.square(eps_dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
            dkin = jnp.mean(jnp.square(dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
            return dy, dp, dfro, dkin

    def all_aug_dynamics(ypr, t, eps, params):
        """
        NN_Dynamics augmented with logp and regularization.
        """
        y, p, *_ = ypr

        dy, dp, eps_dy = ffjord2_dynamics((y, p), t, eps, params)

        dr = reg_dynamics(y, t, params)
        dfro = jnp.mean(jnp.square(eps_dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
        dkin = jnp.mean(jnp.square(dy), axis=[axis_ for axis_ in range(1, dy.ndim)])
        return dy, dp, dr, dfro, dkin

    if reg_type == 'our':
        _odeint_aux = _odeint_aux2
    else:
        _odeint_aux = _odeint_aux3
    nodeint_aux = lambda y0, ts, eps, params: _odeint_aux(lambda y, t, eps, params: dynamics_wrap(y, t, params),
                                                          aug_dynamics, y0, ts, eps, params, **ode_kwargs)[0]
    all_nodeint = lambda y0, ts, eps, params: _odeint(all_aug_dynamics, y0, ts, eps, params, **ode_kwargs)[0]

    def ode_aux(params, y, delta_logp, eps):
        """
        Apply the ODE block.
        """
        ys, delta_logps, *rs = nodeint_aux(reg_init(y, delta_logp), ts, eps, params)
        return (ys[-1], delta_logps[-1], *(rs_[-1] for rs_ in rs))

    def all_ode(params, y, delta_logp, eps):
        """
        Apply the ODE block.
        """
        ys, delta_logps, *rs = all_nodeint(all_reg_init(y, delta_logp), ts, eps, params)
        return (ys[-1], delta_logps[-1], *(rs_[-1] for rs_ in rs))


    def forward_aux(key, params, _x):
        """
        Forward pass of the model.
        """
        eps = get_epsilon(key, _x.shape)

        return ode_aux(params["ode"], *aug_init(_x)[:2], eps)

    def forward_all(key, params, _x):
        """
        Forward pass of the model.
        """
        eps = get_epsilon(key, _x.shape)

        return all_ode(params["ode"], *aug_init(_x)[:2], eps)

    model = {
        "model": {
            "ode": all_ode
        },
        "params": {
            "ode": dynamics_params
        }, "nfe": None,
        "forward_all": forward_all
    }

    if odeint_compare:
        assert not vmap, 'vmap should be False'
        unreg_nodeint = lambda z, delta_logp, t, eps, params: ode_comp(ffjord_dynamics, (z, delta_logp), t, eps, params, **ode_comp_args)

        @jax.jit
        def nfe_fn(key, params, _x):
            """
            Function to return NFE.
            """
            eps = get_epsilon(key, _x.shape)
            (out_ode, dlogp), f_nfe = unreg_nodeint(*aug_init(_x)[:2], ts, eps, params["ode"])
            return out_ode[-1], dlogp[-1], jnp.mean(f_nfe)

    else:
        nfe_fn = None

    # Save the number of function evaluations
    model["nfe"] = nfe_fn

    return forward_aux, model


def aug_init(y):
    """
    Initialize dynamics with 0 for logpx and regs.
    """
    batch_size = y.shape[0]
    if reg_type == "our":
        return y, jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1))
    else:
        return y, jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1))


def reg_init(y, delta_logp):
    """
    Initialize dynamics with 0 for and regs.
    """
    batch_size = y.shape[0]
    if reg_type == "our":
        return y, delta_logp, jnp.zeros((batch_size, 1))
    else:
        return y, delta_logp, jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1))


def all_reg_init(y, delta_logp):
    """
    Initialize dynamics with 0 for and regs.
    """
    batch_size = y.shape[0]
    return y, delta_logp, jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1))

@jax.jit
def standard_normal_logprob(z):
    """
    Log probability of standard normal.
    """
    logz = -0.5 * jnp.log(2 * jnp.pi)
    return logz - jnp.square(z) / 2

@jax.jit
def _loss_fn(z, delta_logp):
    logpz = jnp.sum(jnp.reshape(standard_normal_logprob(z), (z.shape[0], -1)), axis=1, keepdims=True)  # logp(z)
    logpx = logpz - delta_logp

    return -jnp.mean(logpx)  # likelihood in nats

@jax.jit
def _reg_loss_fn(reg):
    return jnp.mean(reg)

@jax.jit
def _weight_fn(params):
    flat_params, _ = ravel_pytree(params)
    return 0.5 * jnp.sum(jnp.square(flat_params))


def loss_fn(forward, params, images, key):
    """
    The loss function for training.
    """
    if reg_type == "our":
        z, delta_logp, regs = forward(key, params, images)
        loss_ = _loss_fn(z, delta_logp)
        reg_ = _reg_loss_fn(regs)
        weight_ = _weight_fn(params)
        return loss_ + lam * reg_ + lam_w * weight_
    else:
        z, delta_logp, fro_regs, kin_regs = forward(key, params, images)
        loss_ = _loss_fn(z, delta_logp)
        fro_reg_ = _reg_loss_fn(fro_regs)
        kin_reg_ = _reg_loss_fn(kin_regs)
        weight_ = _weight_fn(params)
        return loss_ + lam_fro * fro_reg_ + lam_kin * kin_reg_ + lam_w * weight_

# Define a function to compute relative error
@jax.jit
def _rel_error(eststate, truestate):
    """ Compute the relative error between two quantities
    """
    eststate = jnp.concatenate(eststate, axis=-1)
    truestate = jnp.concatenate(truestate, axis=-1)
    return jnp.sum(jnp.abs(eststate - truestate)) / jnp.sum(jnp.abs(eststate) + jnp.abs(truestate))

def init_data():
    """
    Initialize data.
    """
    data = datasets.MINIBOONE()

    num_train = data.trn.N
    # num_test = data.trn.N
    num_test = data.val.N

    if float_64:
        convert = jnp.float64
    else:
        convert = jnp.float32

    data.trn.x = convert(data.trn.x)
    data.val.x = convert(data.val.x)
    data.tst.x = convert(data.tst.x)

    num_batches = num_train // parse_args.batch_size + 1 * (num_train % parse_args.batch_size != 0)
    num_test_batches = num_test // parse_args.test_batch_size + 1 * (num_train % parse_args.test_batch_size != 0)

    # make sure we always save the model on the last iteration
    assert num_batches * parse_args.nepochs % parse_args.save_freq == 0

    def gen_train_data():
        """
        Generator for train data.
        """
        key = rng
        inds = jnp.arange(num_train)

        while True:
            key, = jax.random.split(key, num=1)
            epoch_inds = jax.random.shuffle(key, inds)
            for i in range(num_batches):
                batch_inds = epoch_inds[i * parse_args.batch_size: min((i + 1) * parse_args.batch_size, num_train)]
                yield data.trn.x[batch_inds]

    def gen_val_data():
        """
        Generator for train data.
        """
        inds = jnp.arange(num_test)
        while True:
            for i in range(num_test_batches):
                batch_inds = inds[i * parse_args.test_batch_size: min((i + 1) * parse_args.test_batch_size, num_test)]
                yield data.val.x[batch_inds]

    def gen_test_data():
        """
        Generator for train data.
        """
        inds = jnp.arange(num_test)
        while True:
            for i in range(num_test_batches):
                batch_inds = inds[i * parse_args.test_batch_size: min((i + 1) * parse_args.test_batch_size, num_test)]
                yield data.tst.x[batch_inds]

    ds_train = gen_train_data()
    ds_test = gen_val_data()

    meta = {
        "dims": data.n_dims,
        "num_batches": num_batches,
        "num_test_batches": num_test_batches
    }

    return ds_train, ds_test, meta

def run():
    """
    Run the experiment.
    """

    # init the model first so that jax gets enough GPU memory before TFDS
    forward, model = init_model(43)  # how do you sleep at night
    grad_fn = jax.grad(lambda *args: loss_fn(forward, *args))
    # forward = jax.jit(forward)

    ds_train, ds_test_eval, meta = init_data()
    num_batches = meta["num_batches"]
    num_test_batches = meta["num_test_batches"]

    lr_schedule = optimizers.piecewise_constant(boundaries=[9000, 12750],  # 300 epochs, 425 epochs
                                                values=[1e-3, 1e-4, 1e-5])
    opt_init, opt_update, get_params = optimizers.adam(step_size=lr_schedule)

    unravel_opt = ravel_pytree(opt_init(model["params"]))[1]

    init_params = model["params"]
    opt_state = opt_init(init_params)

    @jax.jit
    def update(_itr, _opt_state, _key, _batch):
        """
        Update the params based on grad for current batch.
        """
        return opt_update(_itr, grad_fn(get_params(_opt_state), _batch, _key), _opt_state)

    @jax.jit
    def sep_losses(_opt_state, _batch, _key):
        """
        Convenience function for calculating losses separately.
        """
        z, delta_logp, r2_regs, fro_regs, kin_regs = model["forward_all"](_key, get_params(_opt_state), _batch)
        loss_ = _loss_fn(z, delta_logp)
        r2_reg_ = _reg_loss_fn(r2_regs)
        fro_reg_ = _reg_loss_fn(fro_regs)
        kin_reg_ = _reg_loss_fn(kin_regs)
        total_loss_ = loss_ + lam * r2_reg_ + lam_fro * fro_reg_ + lam_kin * kin_reg_
        return total_loss_, loss_, r2_reg_, fro_reg_, kin_reg_, (z, delta_logp)

    def evaluate_loss(opt_state, _key, ds_eval, m_batches):
        """
        Convenience function for evaluating loss over train set in smaller batches.
        """
        sep_loss_aug_, sep_loss_, sep_loss_r2_reg_, sep_loss_fro_reg_, sep_loss_kin_reg_, bs, predtime = \
            [], [], [], [], [], [], []
        nfe_odeint, odeint_loss, odeint_relerr, odeint_time = [], [], [], []

        for test_batch_num in range(m_batches):
            _key, = jax.random.split(_key, num=1)
            test_batch = next(ds_eval)

            curr_time = time.time()
            test_batch_loss_aug_, test_batch_loss_, \
            test_batch_loss_r2_reg_, test_batch_loss_fro_reg_, test_batch_loss_kin_reg_, (z, delta_logp) = \
                sep_losses(opt_state, test_batch, _key)
            z.block_until_ready()
            m_predtime = time.time() - curr_time

            if odeint_compare:
                curr_time = time.time()
                out_ode_z, dlogp, nfe_est = model["nfe"](_key, get_params(opt_state), test_batch)
                out_ode_z.block_until_ready()
                m_odetime = time.time() - curr_time
                loss_ = _loss_fn(out_ode_z, dlogp)
                nfe_odeint.append(nfe_est)
                odeint_loss.append(loss_)
                odeint_relerr.append(_rel_error((out_ode_z,dlogp), (z, delta_logp) ))
                odeint_time.append(m_odetime)
                # print('{} | {}'.format(jnp.sum(jnp.abs(out_ode_z-z)),  jnp.sum(jnp.abs(delta_logp-dlogp)) ))
            else:
                nfe_odeint.append(-1)
                odeint_loss.append(-1) 
                odeint_relerr.append(-1)
                odeint_time.append(-1)

            sep_loss_aug_.append(test_batch_loss_aug_)
            sep_loss_.append(test_batch_loss_)
            sep_loss_r2_reg_.append(test_batch_loss_r2_reg_)
            sep_loss_fro_reg_.append(test_batch_loss_fro_reg_)
            sep_loss_kin_reg_.append(test_batch_loss_kin_reg_)
            bs.append(len(test_batch))
            predtime.append(m_predtime)

        sep_loss_aug_ = jnp.array(sep_loss_aug_)
        sep_loss_ = jnp.array(sep_loss_)
        sep_loss_r2_reg_ = jnp.array(sep_loss_r2_reg_)
        sep_loss_fro_reg_ = jnp.array(sep_loss_fro_reg_)
        sep_loss_kin_reg_ = jnp.array(sep_loss_kin_reg_)
        bs = jnp.array(bs)

        nfe_odeint = jnp.array(nfe_odeint)
        odeint_loss = jnp.array(odeint_loss)
        odeint_relerr = jnp.array(odeint_relerr)
        odeint_time = jnp.array(odeint_time)

        return jnp.average(sep_loss_aug_, weights=bs), \
               jnp.average(sep_loss_, weights=bs), \
               jnp.average(sep_loss_r2_reg_, weights=bs), \
               jnp.average(sep_loss_fro_reg_, weights=bs), \
               jnp.average(sep_loss_kin_reg_, weights=bs), \
               jnp.average(jnp.array(predtime), weights=bs),\
               (jnp.average(nfe_odeint, weights=bs), jnp.average(odeint_loss, weights=bs),\
                    jnp.average(odeint_relerr, weights=bs), jnp.average(odeint_time, weights=bs))

    itr = 0
    info = collections.defaultdict(dict)

    key = rng

    # Save the optimal loss_value obtained
    opt_params_dict = None

    opt_loss_test = None
    opt_loss_odeint_test = None

    opt_relerr_odeint_test = None
    opt_predtime_test = None
    opt_predtime_test_odeint = None

    total_compute_time = 0.0
    loss_evol_test = list()
    loss_evol_odeint_test = list()
    predtime_evol_test = list()
    predtime_evol_odeint_test = list()
    compute_time_update = list()
    nfe_evol_test = list()
    err_evol_odeint = list()

    m_parameters_dict = vars(parse_args)
    trajdir = parse_args.trajdir
    out_data_file = trajdir +'dyn_ffjortab_{}_s{}_R{}_G{}'.format(parse_args.reg_type, parse_args.num_steps, parse_args.reg, int(grid))

    # Open the info file to save the command line print
    outfile = open(out_data_file+'_info.txt', 'w')
    outfile.write('Training parameters: {} \n'.format(m_parameters_dict))
    outfile.write('////// Command line messages \n\n\n')
    outfile.close()

    for epoch in tqdm(range(parse_args.nepochs)):
        for i in tqdm(range(num_batches), leave=False):
            key, = jax.random.split(key, num=1)
            batch = next(ds_train)

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

                loss_aug_test_, loss_test_, loss_r2_reg_test_, loss_fro_reg_test_, loss_kin_reg_test_, predtime_test_,\
                (nfe_test_, odeint_loss, odeint_relerr, odeint_time) = evaluate_loss(opt_state, key, ds_test_eval, meta['num_test_batches'])

                # First time we have a value for the loss function
                if opt_loss_test is None or (opt_loss_test < loss_test_):
                    opt_loss_test = loss_test_
                    opt_loss_odeint_test = odeint_loss

                    opt_relerr_odeint_test = odeint_relerr
                    opt_nfe_test = nfe_test_
                    opt_predtime_test = predtime_test_
                    opt_predtime_test_odeint = odeint_time

                    opt_params_dict = get_params(opt_state)

                # Do some printing for result visualization
                print_str = 'Iter {:05d} | Total Update Time {:.2e} | Update time {:.2e}\n'.format(itr, total_compute_time, update_end)
                print_str += '[    Test      ] Loss = {:.2e} | Pred. Time. = {:.2e} | NFE = {:.3e} |  Fro. reg. = {:.3e} | Kin. reg. = {:.3e} | R2. reg. = {:.3e}\n'.format(loss_test_, predtime_test_,-1.,loss_fro_reg_test_,loss_kin_reg_test_,loss_r2_reg_test_)
                print_str += '[  ODEINT Test ] Loss = {:.2e} | Pred. Time. = {:.2e} | NFE = {:.3e} | Diff. Pred.  = {:.2e}\n'.format(odeint_loss, odeint_time, nfe_test_, odeint_relerr)
                print_str += '[  OPT. Value. ] Loss Test = {:.2e} | Loss ODE   = {:.2e} | NFE ODEINT = {:.3e} | Diff. Pred.  = {:.2e} | Pred. Time = {:.2e} | Pred. Time. ODEINT = {:.2e}\n'.format(
                                opt_loss_test, opt_loss_odeint_test, opt_nfe_test, opt_relerr_odeint_test, opt_predtime_test, opt_predtime_test_odeint)
                tqdm.write(print_str)

                # Save all the obtained data
                loss_evol_test.append(loss_test_)
                loss_evol_odeint_test.append(odeint_loss)

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
                                'opt_loss_test' : opt_loss_test, 
                                'opt_nfe_test' : opt_nfe_test, 'opt_diff_test' : opt_relerr_odeint_test,  'opt_loss_odeint_test' : opt_loss_odeint_test,
                                'opt_predtime_test' : opt_predtime_test, 'opt_predtime_test_odeint' : opt_predtime_test_odeint,

                                'loss_evol_test' : loss_evol_test, 'predtime_evol_test' : predtime_evol_test, 'predtime_evol_odeint_test' : predtime_evol_odeint_test,
                                'nfe_evol_odeint' : nfe_evol_test, 'loss_evol_odeint' : loss_evol_odeint_test, 'err_evol_odeint' : err_evol_odeint,
                                'training_parameters' : m_parameters_dict}
                outfile = open(out_data_file+'_res.pkl', "wb")
                pickle.dump(m_dict_res, outfile)
                outfile.close()


if __name__ == "__main__":
    run()