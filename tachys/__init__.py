# -----------------------------------------------------------------------
# tachys — distributed JAX initialisation
#
# Supports three launch modes:
#   python3 main.py              (single process, no MPI)
#   mpirun -np N python3 main.py (OpenMPI → OMPI_COMM_WORLD_SIZE,
#                                  MPICH/PMI  → PMI_SIZE)
#   srun python3 main.py         (Slurm     → SLURM_NTASKS)
#
# jax.distributed.initialize() raises ValueError when called outside an
# MPI/Slurm context, so we skip it for single-process runs.
# -----------------------------------------------------------------------
import builtins
import os
import jax

_mpi_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                os.environ.get("PMI_SIZE",
                os.environ.get("SLURM_NTASKS", 1))))
if _mpi_size > 1:
    jax.distributed.initialize()

jax.config.update("jax_enable_x64", True)

from tachys.parallel import rank, n_devices, MASTER

_builtin_print = builtins.print

def _master_print(*args, **kwargs):
    if rank == MASTER:
        _builtin_print(*args, **kwargs)

# Redirect all print() calls globally so only the master process produces
# output — prevents N identical lines when running on N processes.
builtins.print = _master_print

# -----------------------------------------------------------------------
# Startup banner — printed once by the master process.
# Shows device platform/model, total device count, and node layout.
# -----------------------------------------------------------------------
import sys
import importlib.metadata
_kind      = jax.devices()[0].platform.upper()
_model     = jax.devices()[0].device_kind
_n_nodes   = int(os.environ.get("SLURM_NNODES", 1))
_dev_node  = (jax.process_count() // max(1, _n_nodes)) * jax.local_device_count()
_title     = "⚡ tachys"
_dev_str   = f"{n_devices} × {_kind}  ({_model})"
_node_str  = f"{_n_nodes} node{'s' if _n_nodes > 1 else ''}  ·  {_dev_node} {_kind}/node"

def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "n/a"

_jax_str  = f"jax {_pkg_version('jax')}  ·  flax {_pkg_version('flax')}"

_conda_env = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")
if _conda_env:
    _conda_env = os.path.basename(_conda_env)
_py_ver   = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
_env_str  = f"python {_py_ver}  {sys.executable}" + (f"  ({_conda_env})" if _conda_env else "")

_USE_COLOR = sys.stdout.isatty()
_C = "\033[36m" if _USE_COLOR else ""
_B = "\033[1m"  if _USE_COLOR else ""
_R = "\033[0m"  if _USE_COLOR else ""

_cw  = max(len(_title), len(_dev_str), len(_node_str), len(_jax_str), len(_env_str))
_w   = _cw + 24

def _center(s):
    lp = (_w - len(s)) // 2
    return " " * lp + s + " " * (_w - len(s) - lp)

_tlp = (_w - len(_title)) // 2
_trp = _w - len(_title) - _tlp

print(f"{_C}╔{'═' * _w}╗{_R}")
print(f"{_C}║{_R}{' ' * _tlp}{_B}{_title}{_R}{' ' * _trp}{_C}║{_R}")
print(f"{_C}╟{'─' * _w}╢{_R}")
print(f"{_C}║{_R}{_center(_dev_str)}{_C}║{_R}")
print(f"{_C}║{_R}{_center(_node_str)}{_C}║{_R}")
print(f"{_C}╟{'─' * _w}╢{_R}")
print(f"{_C}║{_R}{_center(_env_str)}{_C}║{_R}")
print(f"{_C}║{_R}{_center(_jax_str)}{_C}║{_R}")
print(f"{_C}╚{'═' * _w}╝{_R}", flush=True)

