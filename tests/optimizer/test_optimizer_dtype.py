"""The optimizers' ``dtype`` field: the precision of the network evaluations
inside an SR-family or TDVP update.

``dtype=None`` is the pre-existing code path, pinned by the regression values of
the other optimizer tests. These pin the new one:

* ``dtype=float64`` reproduces it. The one change on that path is the constant
  shift of the Jacobian before the contraction, which the plain and weighted
  centering of the NTK cannot see (the per-system one is checked in
  test_foundation_centering.py, the multi-process one in tests/parallel).
* ``dtype=float32`` stays close to it, and hands back updates and optimizer
  states in the parameters' own dtypes: a float32 leaf would retrace every
  jitted step, and ``apply_gradients`` would turn the parameters float32.
* The shift is what keeps a float32 NTK accurate when the Jacobian has a large
  common part, which centering after a float32 contraction cancels only after
  rounding.
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
from tachys.optimizer.optimizers import _build_ntk
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
        assert _max_rel_err(step_got[0], step_ref[0]) < 1e-3    # measured <= 2.2e-4
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

    # The same float32 Jacobians without the shift: the default path on float32 parameters.
    unshifted = _ntk(state, wf.replace(params=_cast_floating_to(params, jnp.float32)), None)
    assert err(unshifted) > 5e-5                     # measured 2.4e-4 on CPU, 6.7e-4 on an A6000
    assert err(_ntk(state, wf, jnp.float32)) < 1e-5  # measured 1e-6 on both


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
