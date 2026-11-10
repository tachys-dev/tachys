"""The JAXMg kernel backend (tachys.optimizer._kernels_distributed), on CPU.

JAXMg itself needs GPUs, one per process, so the two functions that call it,
``_potrs`` and ``_syevd``, are not tested here. Everything around them is
checked against the default backend, with the module-scoped ``results``
fixture launching

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python tests/optimizer/run_kernels_distributed.py <tmpfile>

once: four devices in one process, with ``_potrs`` and ``_syevd`` replaced by
jnp solves of the gathered system.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import pytest

_RUNNER = Path(__file__).parent / "run_kernels_distributed.py"
_FORCED_DEVICE_COUNT = 4


@pytest.fixture(scope="module")
def results():
    fd, output_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={_FORCED_DEVICE_COUNT}"

        proc = subprocess.run(
            [sys.executable, str(_RUNNER), output_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            pytest.fail(
                f"runner exited with code {proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}"
            )
        with open(output_path) as f:
            return json.load(f)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_runner_used_forced_device_count(results):
    assert results["n_devices"] == _FORCED_DEVICE_COUNT


# Maximum relative errors against the default backend. The NTK is the same
# contractions, only reduced in another order; in "complex" mode the default
# Cholesky solves by Schur complement, the distributed one in one piece.
_TOLERANCES = {"layout": 0.0, "ntk": 1e-12, "solvers": 1e-12, "updates": 1e-10}


@pytest.mark.parametrize("group", sorted(_TOLERANCES))
def test_matches_the_default_backend(results, group):
    """layout : each device's rows of the right-hand side are rows
                d*2m + a*m + i of the device-major order, i.e. component-major
                rows a*M + d*m + i; gathering the local solutions gives the
                component-major [u, v] back; the diagonal shift lands on the
                global diagonal of the row-sharded matrix.
    ntk     : the row-sharded, row-centred kernel equals _kernels.compute_ntk
              (real/complex, nbatches 1/2, weights, MARCH's V, a dtype, and a
              FoundationState whose systems are spread over all the devices).
    solvers : the Cholesky and the eigh solves, inside shard_map, on a random
              kernel.
    updates : two steps of SR/SPRING/MARCH (real/complex, weighted or not, and
              a FoundationState) and of TDVP (weighted or not) through their
              own step, with kernels="default" and kernels="distributed" in
              one process.
    """
    tol = _TOLERANCES[group]
    worst = {name: err for name, err in results[group].items() if not err <= tol}
    assert not worst, f"{group}: relative errors above {tol:g}: {worst}"


@pytest.mark.parametrize("case", ["cholesky_not_pd", "eigh_nan"])
def test_failed_solve_gives_a_zero_step(results, case):
    """A failed solve returns NaN: the Cholesky factorization of a kernel that
    is not positive definite, the eigh of one that is not finite. The solvers
    must degrade to a zero step, as the default ones do."""
    assert results["failed_max_abs"][case] == 0.0


# ─── The kernels field ────────────────────────────────────────────────────────

def test_default_kernels():
    from tachys.optimizer import SR, _kernels
    optimizer = SR(diag_shift=1e-3, mode="real")
    assert optimizer.kernels == "default"
    assert optimizer._backend is _kernels


def test_distributed_falls_back_on_a_single_device(monkeypatch):
    from tachys.optimizer import SR, optimizers
    monkeypatch.setattr(optimizers, "n_devices", 1)
    with pytest.warns(UserWarning, match="kernels='distributed' needs several devices"):
        optimizer = SR(diag_shift=1e-3, mode="real", kernels="distributed")
    assert optimizer.kernels == "default"


def test_kernels_is_a_static_field(monkeypatch):
    """It is part of the jit cache key, so both backends compile side by side."""
    from tachys.optimizer import SR, _kernels_distributed, optimizers
    monkeypatch.setattr(optimizers, "n_devices", _FORCED_DEVICE_COUNT)
    optimizer = SR(diag_shift=1e-3, mode="real", kernels="distributed")
    assert optimizer._backend is _kernels_distributed
    assert jax.tree.structure(optimizer) != jax.tree.structure(optimizer.replace(kernels="default"))


def test_unknown_kernels_raise():
    from tachys.optimizer import SR
    with pytest.raises(ValueError, match="kernels='jaxmg'"):
        SR(diag_shift=1e-3, mode="real", kernels="jaxmg")


def test_tdvp_accepts_distributed(monkeypatch):
    from tachys.dynamics import TDVP
    from tachys.optimizer import _kernels_distributed, optimizers
    monkeypatch.setattr(optimizers, "n_devices", _FORCED_DEVICE_COUNT)
    tdvp = TDVP(mode="complex", kernels="distributed")
    assert tdvp._backend is _kernels_distributed


@pytest.mark.parametrize("version, ok", [("1.4.0", False), ("1.4.2", True), ("1.10.0", True)])
def test_jaxmg_version_gate(monkeypatch, version, ok):
    """The *_shardmap_ctx solvers take local blocks from JAXMg 1.4.2 on."""
    import types
    from tachys.optimizer._kernels_distributed import _jaxmg
    monkeypatch.setitem(sys.modules, "jaxmg", types.SimpleNamespace(__version__=version))
    if ok:
        assert _jaxmg().__version__ == version
    else:
        with pytest.raises(ImportError, match="needs JAXMg 1.4.2 or later"):
            _jaxmg()


# ─── Tile size ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_local, tile", [
    (4096, 1024),   # capped
    (3072, 1024),   # the largest power of two dividing it
    (384, 128),
    (8, 8),         # a whole shard in one tile
    (24, 16),       # no divisor >= 128: pad to a multiple of 16
    (1000, 512),
])
def test_tile_size(monkeypatch, n_local, tile):
    from tachys.optimizer._kernels_distributed import _tile_size
    monkeypatch.delenv("TACHYS_JAXMG_TILE_SIZE", raising=False)
    assert _tile_size(n_local) == tile


def test_tile_size_cap_from_the_environment(monkeypatch):
    from tachys.optimizer._kernels_distributed import _tile_size
    monkeypatch.setenv("TACHYS_JAXMG_TILE_SIZE", "256")
    assert _tile_size(4096) == 256
