"""Runner for the CPU tests of the JAXMg kernel backend (_kernels_distributed).

JAXMg needs GPUs, so the two functions that call it, ``_potrs`` and
``_syevd``, are replaced here by jnp solves of the gathered system, inside the
same shard_map. All the rest of the backend -- the row-sharded NTK, its
centring, the device-major order, the diagonal shift, the solves inside the
optimizer step's shard_map -- runs as on GPUs, on several forced CPU devices
in one process. Every quantity is compared with the default backend
(_kernels), and the relative errors are written as JSON to the path given as
the first command-line argument.

Typical invocation from the project root:

    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_force_host_platform_device_count=4" \\
        python tests/optimizer/run_kernels_distributed.py /tmp/out.json
"""
import json
import sys
from functools import partial
from pathlib import Path

# Ensure the project root and tests/ (for the frozen test ansatze and
# configurations) are on sys.path regardless of working directory.
_ROOT = Path(__file__).resolve().parents[2]
_TESTS_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _TESTS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import shard_map
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cho_factor, cho_solve
from jax.sharding import NamedSharding, PartitionSpec as P

from testing_ansatz import FermionFoundationRBM, SpinRBM, frozen_params
from testing_configs import frozen_config
from tachys.lattice.fermions.hamiltonians.hubbard import hubbard_square_pbc
from tachys.lattice.foundation.foundation_state import FermionFoundationState
from tachys.lattice.foundation.operators import combine_systems, extract_system_couplings
from tachys.lattice.lattice_database import square
from tachys.lattice.operator.local_estimator import compute_expectation, local_estimator
from tachys.lattice.spins.hamiltonians.heisenberg import heisenberg_square_pbc
from tachys.lattice.spins.spin_state import SpinState
from tachys.dynamics import TDVP
from tachys.optimizer import MARCH, SPRING, SR
from tachys.optimizer import _kernels
from tachys.optimizer import _kernels_distributed as dist
from tachys.parallel import hard_shard, mesh, n_devices
from tachys.wavefunction import WaveFunction


# ─── JAXMg stand-ins: the same solves, by jnp on the gathered system ─────────

def _potrs_reference(a_rows, b_rows):
    """This device's rows of the solution, as JAXMg's potrs_shardmap_ctx
    returns them, from a Cholesky solve of the whole gathered system. A matrix
    that is not positive definite gives NaN, as it does in JAXMg."""
    a = jax.lax.all_gather(a_rows, 'i', tiled=True)
    b = jax.lax.all_gather(b_rows, 'i', tiled=True)
    return hard_shard(cho_solve(cho_factor(a, lower=True), b))


def _syevd_reference(a_rows):
    """Every eigenvalue and this device's rows of the eigenvectors, as JAXMg's
    syevd_shardmap_ctx returns them, from eigh of the whole gathered matrix.
    Like cuSOLVERMp, it reads one triangle and does not symmetrize."""
    a = jax.lax.all_gather(a_rows, 'i', tiled=True)
    lam, U = jnp.linalg.eigh(a, symmetrize_input=False)
    return lam, hard_shard(U)


dist._potrs = _potrs_reference
dist._syevd = _syevd_reference


# ─── Helpers ──────────────────────────────────────────────────────────────────

def rel_err(got, ref):
    got, ref = ravel_pytree(got)[0], ravel_pytree(ref)[0]
    return float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))


def to_default_order(G, mode):
    """A dense kernel in device-major order (rows and columns) to component-major."""
    if mode == "real":
        return G
    return rearrange(G, '(d a i) (e b j) -> (a d i) (b e j)', a=2, b=2, d=n_devices, e=n_devices)


def to_device_major(G, mode):
    if mode == "real":
        return G
    return rearrange(G, '(a d i) (b e j) -> (d a i) (e b j)', a=2, b=2, d=n_devices, e=n_devices)


def row_sharded(G):
    return jax.device_put(G, NamedSharding(mesh, dist.NTK_SPEC))


def sharded_solve(solver, G, eps, shift, mode):
    """A solver of dist on a row-sharded kernel, inside shard_map as in the
    optimizer step."""
    solve = shard_map(partial(solver, mode=mode), mesh=mesh,
                      in_specs=(dist.NTK_SPEC, P(), P()), out_specs=P(), check_vma=False)
    return jax.jit(solve)(G, eps, shift)


# ─── Systems ──────────────────────────────────────────────────────────────────

L = 4
DIAG_SHIFT = 1e-3
N_MC_SPIN = 16
WEIGHTS = 1.0 + 0.5 * jnp.sin(jnp.arange(float(N_MC_SPIN)))


def spin_setup(mode):
    """As in test_optimizer_dtype: SpinRBM on the 4x4 Heisenberg model, 16 chains."""
    H = heisenberg_square_pbc(L, J=1.0)
    state = SpinState(spins=frozen_config("square16_nmc16"), lattice=square(shape=(L, L)))
    model = SpinRBM(hidden_units=2, dtype=jnp.float64, complex=(mode == "complex"))
    params = jax.tree.map(lambda x: 0.4 * x, frozen_params(f"reweighted_square16_2hidden_{mode}"))
    return H, state, WaveFunction(params=params, apply_fn=model.apply)


def spin_local_energies(H, state, wf):
    return local_estimator(H, state, wf, wf.apply_fn(wf.params, state), optimize_mask=False)


def permute_chains(op, order):
    """The operator with its per-chain couplings, (n_terms, N_mc), reordered."""
    if hasattr(op, "operators"):
        return op.replace(operators=tuple(permute_chains(o, order) for o in op.operators))
    if jnp.ndim(op.coupling) == 2:
        return op.replace(coupling=op.coupling[:, order])
    return op


def foundation_setup():
    """As in test_foundation_centering, but with the chains of the four systems
    interleaved, so that every device holds chains of every system and the
    per-system means really combine the devices."""
    Us, n_per_system, Ne = [0, 2, 4, 8], 8, 10
    n_systems = len(Us)
    H = combine_systems([hubbard_square_pbc(L, U=U) for U in Us], n_per_system)
    couplings = extract_system_couplings(H)
    system_ids = jnp.repeat(jnp.arange(n_systems), n_per_system)
    occupations = frozen_config("foundation_square16_ne10_nmc32")

    # chain k of the interleaved batch is chain k // n_systems of system k % n_systems
    order = np.arange(n_systems * n_per_system).reshape(n_systems, n_per_system).T.reshape(-1)
    H = permute_chains(H, order)
    state = FermionFoundationState(
        occupations=occupations[order], lattice=square(shape=(L, L)), Ne=Ne,
        system_couplings=couplings[order], system_ids=system_ids[order], n_systems=n_systems,
    )
    model = FermionFoundationRBM(hidden_units=8)
    wf = WaveFunction(params=frozen_params("foundation_hidden8"), apply_fn=model.apply,
                      dtype=jnp.float64)
    return H, state, wf


# ─── 1. Layout: the device-major order and the diagonal shift ─────────────────

def check_layout():
    out = {}
    M = 8 * n_devices
    m = M // n_devices
    eps = jnp.arange(1.0, M + 1) + 1j * jnp.arange(M + 1.0, 2 * M + 1)
    # Device-major row d*2m + a*m + i is component-major row a*M + d*m + i.
    device_major = np.array([a * M + d * m + i for d in range(n_devices) for a in range(2) for i in range(m)])

    for mode, order in (("real", np.arange(M)), ("complex", device_major)):
        local_rhs = shard_map(partial(dist._local_rhs, mode=mode), mesh=mesh, in_specs=P(),
                              out_specs=P('i'), check_vma=False)
        b = _kernels._solver_rhs(eps, mode)                 # component-major
        out[f"local_rhs_{mode}"] = rel_err(jax.jit(local_rhs)(eps), b[order])

        gather = shard_map(partial(dist._gather_solution, mode=mode), mesh=mesh, in_specs=P('i'),
                           out_specs=P(), check_vma=False)
        out[f"gather_solution_{mode}"] = rel_err(jax.jit(gather)(b[order]), b)

    G = jax.random.normal(jax.random.key(0), (16, 16))
    shift = jnp.atleast_1d(0.25)
    add_shift = shard_map(dist._add_diag_shift, mesh=mesh, in_specs=(dist.NTK_SPEC, P()),
                          out_specs=dist.NTK_SPEC, check_vma=False)
    got = jax.jit(add_shift)(row_sharded(G), shift)
    out["diag_shift"] = rel_err(got, G + 0.25 * jnp.eye(16))
    return out


# ─── 2. The row-sharded kernel against _kernels.compute_ntk ───────────────────

def ntk_error(state, wf, mode, weights=None, V=None, nbatches=1, dtype=None):
    w_specs = () if weights is None else (P('i'),)
    w_args  = () if weights is None else (weights,)

    def default(state, wf, *w):
        ntk = _kernels.compute_ntk(state, wf, mode, weights=w[0] if w else None, V=V,
                                   nbatches=nbatches, dtype=dtype)
        return _kernels._ntk_to_dense(ntk, mode)

    def distributed(state, wf, *w):
        return dist.compute_ntk(state, wf, mode, weights=w[0] if w else None, V=V,
                                nbatches=nbatches, dtype=dtype)

    ref = shard_map(default, mesh=mesh, in_specs=(P('i'), P()) + w_specs,
                    out_specs=P(), check_vma=False)(state, wf, *w_args)
    got = shard_map(distributed, mesh=mesh, in_specs=(P('i'), P()) + w_specs,
                    out_specs=dist.NTK_SPEC, check_vma=False)(state, wf, *w_args)
    return rel_err(to_default_order(got, mode), ref)


def check_ntk():
    out = {}
    for mode in ("real", "complex"):
        _, state, wf = spin_setup(mode)
        V = jax.tree.map(lambda p: 1.0 + jnp.abs(p), wf.params)
        for nbatches in (1, 2):
            out[f"spin_{mode}_nb{nbatches}"] = ntk_error(state, wf, mode, nbatches=nbatches)
            out[f"spin_{mode}_nb{nbatches}_weighted"] = ntk_error(state, wf, mode, WEIGHTS,
                                                                   nbatches=nbatches)
        out[f"spin_{mode}_V"] = ntk_error(state, wf, mode, V=V)
        out[f"spin_{mode}_float64"] = ntk_error(state, wf, mode, nbatches=2, dtype=jnp.float64)

        _, fstate, fwf = foundation_setup()
        for nbatches in (1, 2):
            out[f"foundation_{mode}_nb{nbatches}"] = ntk_error(fstate, fwf, mode, nbatches=nbatches)
    return out


# ─── 3. The Cholesky and eigh solvers on a random kernel ──────────────────────

def random_ntk(M, P_, seed):
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(M, 2, P_))
    ntk = jnp.asarray(np.einsum('iap,jbp->ijab', J, J) / M)
    eps = jnp.asarray(rng.normal(size=M) + 1j * rng.normal(size=M))
    return ntk, eps


def check_solvers():
    out = {}
    M = 4 * n_devices
    shift = jnp.atleast_1d(DIAG_SHIFT)
    for mode in ("real", "complex"):
        ntk, eps = random_ntk(M, 40, seed=3)
        if mode == "real":
            ntk = ntk[..., 0, 0]
        G = row_sharded(to_device_major(_kernels._ntk_to_dense(ntk, mode), mode))
        ref = _kernels.linear_solver_cholesky(ntk, eps, shift, mode=mode)
        out[f"cholesky_{mode}"] = rel_err(sharded_solve(dist.linear_solver_cholesky, G, eps, shift, mode), ref)
        ref = _kernels.linear_solver_eigh(ntk, eps, shift, mode=mode)
        out[f"eigh_{mode}"] = rel_err(sharded_solve(dist.linear_solver_eigh, G, eps, shift, mode), ref)

    # Failed solves give zeros, as in the default solvers: Cholesky of a kernel
    # that is not positive definite, eigh of a kernel that is not finite.
    eps = jnp.ones(M, dtype=jnp.complex128)
    zero = jnp.atleast_1d(0.0)
    failed = {
        "cholesky_not_pd": sharded_solve(dist.linear_solver_cholesky, row_sharded(-jnp.eye(M)), eps, zero, "real"),
        "eigh_nan": sharded_solve(dist.linear_solver_eigh, row_sharded(jnp.full((M, M), jnp.nan)), eps, zero, "real"),
    }
    return out, {name: float(jnp.max(jnp.abs(sol))) for name, sol in failed.items()}


# ─── 4. Optimizer updates, default vs distributed backend ─────────────────────

def two_steps(optimizer, H, state, wf, weights, local_energies):
    """(updates, opt_state) of two consecutive steps, so that the SPRING and
    MARCH momentum terms act on a nonzero previous update."""
    opt_state = optimizer.init(wf.params)
    steps = []
    for _ in range(2):
        updates, opt_state = optimizer(local_energies(H, state, wf), opt_state, state, wf,
                                       weights=weights)
        steps.append((updates, opt_state))
        wf = wf.apply_gradients(updates, 0.01)
    return steps


def foundation_local_energies(H, state, wf):
    return compute_expectation(H, wf, state, wf.apply_fn(wf.params, state))[0]


def run_updates(kernels):
    """Every update case, with the optimizers' ``kernels`` set to ``kernels``."""
    results = {}
    for mode in ("real", "complex"):
        H, state, wf = spin_setup(mode)
        for name, make in (("SR", SR), ("SPRING", SPRING), ("MARCH", MARCH)):
            for wname, weights in (("", None), ("_weighted", WEIGHTS)):
                optimizer = make(diag_shift=DIAG_SHIFT, mode=mode, nbatches=2, kernels=kernels)
                results[f"{name}_{mode}{wname}"] = two_steps(optimizer, H, state, wf, weights,
                                                            spin_local_energies)
        fH, fstate, fwf = foundation_setup()
        results[f"SR_foundation_{mode}"] = two_steps(SR(diag_shift=DIAG_SHIFT, mode=mode, kernels=kernels),
                                                     fH, fstate, fwf, None, foundation_local_energies)

    H, state, wf = spin_setup("complex")
    for wname, weights in (("", None), ("_weighted", WEIGHTS)):
        tdvp = TDVP(mode="complex", nbatches=2, kernels=kernels)
        results[f"TDVP_complex{wname}"] = two_steps(tdvp, H, state, wf, weights, spin_local_energies)
    return results


def check_updates():
    ref = run_updates("default")
    got = run_updates("distributed")
    return {name: max(rel_err(g[0], r[0]) for g, r in zip(got[name], ref[name])) for name in ref}


if __name__ == "__main__":
    solvers, failed_max_abs = check_solvers()
    results = {
        "n_devices": n_devices,
        "layout": check_layout(),
        "ntk": check_ntk(),
        "solvers": solvers,
        "failed_max_abs": failed_max_abs,
        "updates": check_updates(),
    }
    with open(sys.argv[1], "w") as f:
        json.dump(results, f, indent=1)
