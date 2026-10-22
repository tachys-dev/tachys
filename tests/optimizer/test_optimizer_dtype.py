"""The optimizers' ``dtype`` field: the precision of the network evaluations
inside an SR-family or TDVP update.

``dtype=None`` is the pre-existing code path, pinned by the regression values of
the other optimizer tests. These pin the new one:

* ``dtype=float64`` reproduces it: the casts and the full-precision matmuls do
  nothing in float64.
* ``dtype=float32`` stays close to it, and hands back updates and optimizer
  states in the parameters' own dtypes: a float32 leaf would retrace every
  jitted step, and ``apply_gradients`` would turn the parameters float32.
* The Jacobian shift before the contraction, on every path, is what keeps a
  float32 NTK accurate when the Jacobian has a large common part, which
  centering after a float32 contraction cancels only after rounding.
* The float32 VJP contracts the kernel's own float32 Jacobian, in float64:
  when a few parameters carry nearly all of the kernel, the SR solution
  cancels to a small step along them, and only the kernel's own Jacobian
  keeps that cancellation.
* Every float32 matmul runs at full precision. On GPUs JAX's default float32
  matmul is TF32, which a CPU run of the tests above cannot notice.
"""
from functools import partial

import jax
import jax.numpy as jnp
import pytest
from jax import shard_map
from jax.flatten_util import ravel_pytree
from jax.sharding import PartitionSpec as P

from testing_ansatz import SpinRBM, frozen_params
from testing_configs import frozen_config
from tachys.dynamics import TDVP
from tachys.lattice.lattice_database import square
from tachys.lattice.operator.local_estimator import local_estimator
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.spin_state import SpinState
from tachys.optimizer import MARCH, SPRING, SR
from tachys.optimizer._kernels import (_ntk_contraction, _shifted_jacobian, center_ntk, jacobian_vjp,
                                      linear_solver_cholesky)
from tachys.optimizer.optimizers import _build_ntk, _make_apply_fn
from tachys.parallel import mesh
from tachys.utils import _cast_floating_to
from tachys.wavefunction import WaveFunction

L = 4
N_mc = 16
DIAG_SHIFT = 1e-3
WEIGHTS = 1.0 + 0.5 * jnp.sin(jnp.arange(float(N_mc)))

OPTIMIZERS = [
    partial(SR, diag_shift=DIAG_SHIFT),
    partial(SPRING, diag_shift=DIAG_SHIFT, mu=0.9),
    partial(MARCH, diag_shift=DIAG_SHIFT, mu=0.95, beta=0.995),
]
OPTIMIZER_IDS = ["SR", "SPRING", "MARCH"]


def _setup(mode):
    H = heisenberg_square_pbc(L, J=1.0)
    state = SpinState(spins=frozen_config("square16_nmc16"), lattice=square(shape=(L, L)))
    model = SpinRBM(hidden_units=2, dtype=jnp.float64, complex=(mode == "complex"))
    # O(1) parameters, as in test_reweighted_sr: a near-uniform psi has a
    # degenerate Jacobian.
    params = jax.tree.map(lambda x: 0.4 * x, frozen_params(f"reweighted_square16_2hidden_{mode}"))
    return H, state, WaveFunction(params=params, apply_fn=model.apply)


def _local_energies(H, state, wf):
    return local_estimator(H, state, wf, wf.apply_fn(wf.params, state), optimize_mask=False)


def _two_steps(optimizer, H, state, wf, weights):
    """(updates, opt_state) of two consecutive steps, so that the SPRING and
    MARCH momentum terms act on a nonzero previous update."""
    opt_state = optimizer.init(wf.params)
    steps = []
    for _ in range(2):
        updates, opt_state = optimizer(_local_energies(H, state, wf), opt_state, state, wf,
                                       weights=weights)
        steps.append((updates, opt_state))
        wf = wf.apply_gradients(updates, 0.01)
    return steps


def _max_rel_err(tree, ref):
    x, r = ravel_pytree(tree)[0], ravel_pytree(ref)[0]
    return float(jnp.max(jnp.abs(x - r)) / jnp.max(jnp.abs(r)))


def _dtypes(tree):
    return jax.tree.map(jnp.result_type, tree)


@pytest.mark.parametrize("weights", [None, WEIGHTS], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("mode", ["real", "complex"])
@pytest.mark.parametrize("make_optimizer", OPTIMIZERS, ids=OPTIMIZER_IDS)
def test_float64_dtype_reproduces_the_default_update(make_optimizer, mode, weights):
    H, state, wf = _setup(mode)
    ref = _two_steps(make_optimizer(mode=mode), H, state, wf, weights)
    got = _two_steps(make_optimizer(mode=mode, dtype=jnp.float64), H, state, wf, weights)
    for (updates, _), (updates_ref, _) in zip(got, ref):
        assert _max_rel_err(updates, updates_ref) < 1e-9     # measured <= 1e-12


@pytest.mark.parametrize("weights", [None, WEIGHTS], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("mode", ["real", "complex"])
@pytest.mark.parametrize("make_optimizer", OPTIMIZERS, ids=OPTIMIZER_IDS)
def test_float32_dtype_is_close_and_keeps_the_parameter_dtypes(make_optimizer, mode, weights):
    H, state, wf = _setup(mode)
    optimizer = make_optimizer(mode=mode, dtype="float32")     # as read from a config
    assert optimizer.dtype == jnp.float32

    ref = _two_steps(make_optimizer(mode=mode), H, state, wf, weights)
    got = _two_steps(optimizer, H, state, wf, weights)
    for step_got, step_ref in zip(got, ref):
        # Measured <= 5.3e-4 (complex SPRING, weighted, second step), and machine-dependent:
        # the diag_shift=1e-3 solve amplifies float32 rounding.
        assert _max_rel_err(step_got[0], step_ref[0]) < 1e-2
        assert _dtypes(step_got) == _dtypes(step_ref)


@pytest.mark.parametrize("weights", [None, WEIGHTS], ids=["unweighted", "weighted"])
def test_tdvp_dtype(weights):
    H, state, wf = _setup("complex")
    E_L = _local_energies(H, state, wf)

    def velocity(dtype):
        tdvp = TDVP(mode="complex", dtype=dtype)
        return tdvp(E_L, tdvp.init(wf.params), state, wf, weights=weights)[0]

    ref = velocity(None)
    assert _max_rel_err(velocity(jnp.float64), ref) < 1e-9     # measured <= 1e-11
    f32 = velocity(jnp.float32)
    assert _max_rel_err(f32, ref) < 1e-3                       # measured 1.6e-4
    assert _dtypes(f32) == _dtypes(ref)


def _ntk(state, wf, dtype):
    build = partial(_build_ntk, mode="real", weights=None, nbatches=1, N_mc_local=N_mc,
                    V=None, dtype=dtype)
    return shard_map(build, mesh=mesh, in_specs=(P('i'), P()), out_specs=P(),
                     check_vma=False)(state, wf)


def test_float32_ntk_survives_a_jacobian_with_a_large_mean():
    """Spins shifted by 30 give every kernel derivative tanh(.) * (s_j + 30) a
    common part about 30 times its fluctuation. Centering after a float32
    contraction then loses (mean/std)^2 ~ 900 in relative precision; shifting
    the Jacobian first leaves only the float32 rounding of the Jacobian itself.
    """
    state = SpinState(spins=frozen_config("square16_nmc16") + 30, lattice=square(shape=(L, L)))
    model = SpinRBM(hidden_units=16, dtype=jnp.float64)
    params = jax.tree.map(lambda x: 0.05 * x, frozen_params("square16_16hidden"))
    wf = WaveFunction(params=params, apply_fn=model.apply)

    ref = _ntk(state, wf, None)

    def err(ntk):
        return float(jnp.max(jnp.abs(ntk - ref)) / jnp.max(jnp.abs(ref)))

    # The same float32 Jacobians, contracted without the shift.
    f = lambda p, s: jnp.squeeze(wf.apply_fn(p, s)).real
    J = jax.vmap(jax.jacobian(f), in_axes=(None, 0))(_cast_floating_to(params, jnp.float32), state)
    unshifted = center_ntk(_ntk_contraction(J, J, "real").astype(jnp.float64), None, state) / N_mc
    assert err(unshifted) > 2e-5                     # measured 8.2e-5 on CPU, 6.7e-4 on an A6000
    assert err(_ntk(state, wf, jnp.float32)) < 1e-5  # measured 1e-6 on both


def _wide_rbm():
    """square16_16hidden tiled to 128 hidden units."""
    base = frozen_params("square16_16hidden")["params"]["linear"]
    kernel = jnp.concatenate([base["kernel"] * (0.3 + 0.1 * k) for k in range(8)], axis=1)
    bias = jnp.concatenate([base["bias"] + 0.01 * k for k in range(8)])
    return SpinRBM(hidden_units=128, dtype=jnp.float64), {"params": {"linear": {"kernel": kernel, "bias": bias}}}


def _dominant_leaf():
    """log psi = 0.01 * RBM + 100 g sum_j cos(j) s_j: g carries the kernel's
    one large eigenvalue (6e4), the RBM its small ones (1e-3 to 6e-3)."""
    state = SpinState(spins=frozen_config("square16_nmc16"), lattice=square(shape=(L, L)))
    model, rbm = _wide_rbm()
    c = jnp.cos(jnp.arange(L * L))

    def apply_fn(p, s):
        return 0.01 * model.apply(p["rbm"], s) + 100.0 * p["g"][0] * (s.spins @ c.astype(p["g"].dtype))

    return state, WaveFunction(params={"g": jnp.zeros(1), "rbm": rbm}, apply_fn=apply_fn)


def test_float32_vjp_contracts_the_ntks_own_jacobian():
    """The SR solution x cancels to a small step along g, and J^T x keeps that
    cancellation only for the J the kernel was built from. A reverse-mode
    float32 VJP rounds g's sum differently, by eps * sum_i |J_ig x_i|."""
    state, wf = _dominant_leaf()
    x = linear_solver_cholesky(_ntk(state, wf, jnp.float32), jnp.cos(1.3 * jnp.arange(N_mc)), 1e-4, "real")
    x = (x - jnp.mean(x)) / N_mc ** 0.5
    params, state32 = _cast_floating_to((wf.params, state), jnp.float32)

    # Compiled, as in the optimizer: an eager mean rounds the shift differently.
    kernel_J = jax.jit(shard_map(lambda s: _shifted_jacobian(params, s, wf, "real", N_mc, jnp.float32)(s),
                                 mesh=mesh, in_specs=P('i'), out_specs=P('i'), check_vma=False))(state32)
    ref = jnp.dot(kernel_J["g"][:, 0].astype(jnp.float64), x)
    got = shard_map(lambda s, w, y: jacobian_vjp(s, w, "real", y, 1, jnp.float32), mesh=mesh,
                    in_specs=(P('i'), P(), P('i')), out_specs=P(), check_vma=False)(state, wf, x)["g"][0]
    with jax.default_matmul_precision("highest"):
        out, vjp = jax.vjp(lambda p: _make_apply_fn(wf.apply_fn, "real")(p, state32), params)
        reverse = vjp(x.astype(out.dtype))[0]["g"][0]
    assert abs(got - ref) < 1e-10 * abs(ref)         # measured 0
    assert abs(reverse - ref) > 1e-2 * abs(ref)      # measured 3.2e-1 on CPU


def _dot_precisions(jaxpr, found):
    """(input dtype, precision) of every dot_general in ``jaxpr``, sub-jaxprs included."""
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            found.append((eqn.invars[0].aval.dtype, eqn.params["precision"]))
        for param in eqn.params.values():
            for sub in param if isinstance(param, (tuple, list)) else (param,):
                sub = getattr(sub, "jaxpr", sub)    # ClosedJaxpr -> Jaxpr
                if hasattr(sub, "eqns"):
                    _dot_precisions(sub, found)
    return found


@pytest.mark.parametrize("mode", ["real", "complex"])
def test_float32_dtype_runs_every_float32_matmul_at_full_precision(mode):
    """MARCH traces all three network evaluations of an update: the NTK, the
    JVP correction and the VJP."""
    H, state, wf = _setup(mode)
    E_L = _local_energies(H, state, wf)

    def dots(dtype):
        optimizer = MARCH(diag_shift=DIAG_SHIFT, mode=mode, dtype=dtype)
        opt_state = optimizer.init(wf.params)
        jaxpr = jax.make_jaxpr(lambda E_L: optimizer(E_L, opt_state, state, wf))(E_L)
        return _dot_precisions(jaxpr.jaxpr, [])

    highest = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    float32_dots = [precision for dtype, precision in dots(jnp.float32) if dtype == jnp.float32]
    assert float32_dots and all(p == highest for p in float32_dots)
    assert all(p is None for _, p in dots(None))
