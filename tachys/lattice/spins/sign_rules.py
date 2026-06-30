import jax.numpy as jnp

def _as_LxLy(L):
    """Normalize L into (Lx, Ly). Accepts an int (square) or a (Lx, Ly) pair."""
    if isinstance(L, (tuple, list)):
        Lx, Ly = L
        return int(Lx), int(Ly)
    return int(L), int(L)

def triangular_classical_log_phase(spins, L):
    """
    120-degree / three-sublattice classical sign rule as a log-phase.

    spins: array, shape (batch, Lx*Ly), S^z +1 (up) / -1 (down),
           row-major flattened (site index = x*Ly + y).
    L:     int (square) or (Lx, Ly) tuple.
    Returns: complex array, shape (batch,).
    """
    Lx, Ly = _as_LxLy(L)
    sublattice_idx_x = jnp.arange(Lx) % 3
    sublattice_idx_y = jnp.arange(Ly) % 3  #! add a - sign if column-major ordering
    sublattice_idx = sublattice_idx_x[:, None] + sublattice_idx_y[None, :]
    sublattice_idx = sublattice_idx.flatten() % 3          # (Lx*Ly,)

    # only down spins contribute; up spins give factor 1 -> log 0
    contrib = jnp.where(spins > 0, 0, sublattice_idx)      # broadcasts to (batch, Lx*Ly)
    total = jnp.sum(contrib, axis=-1)                      # (batch,)
    return (1j * (2 * jnp.pi / 3) * total).astype(jnp.complex128)

def MSR_log_phase_square(spins, L):
    """
    Marshall sign rule as a log-phase: log((-1)^{N_down^A}) = i*pi*(N_down^A mod 2).

    spins: array, shape (batch, Lx*Ly), S^z +1 (up) / -1 (down),
           row-major flattened (site index = x*Ly + y).
    L:     int (square) or (Lx, Ly) tuple.
    Returns: complex array, shape (batch,).
    """
    Lx, Ly = _as_LxLy(L)
    # A-sublattice = checkerboard sites with (x + y) % 2 == 0
    sublattice_parity = (jnp.arange(Lx)[:, None] + jnp.arange(Ly)[None, :]) % 2
    A_mask = (sublattice_parity == 0).astype(jnp.int32).flatten()   # (Lx*Ly,), 1 on A else 0

    # up spins -> 0, down spins -> A_mask (counts only down spins on A)
    contrib = jnp.where(spins > 0, 0, A_mask)      # broadcasts to (batch, Lx*Ly)
    N_down_A = jnp.sum(contrib, axis=-1)           # (batch,)

    return (1j * jnp.pi * (N_down_A % 2)).astype(jnp.complex128)

def MSR_log_phase_chain(spins, L):
    """
    Marshall sign rule as a log-phase: log((-1)^{N_down^A}) = i*pi*(N_down^A mod 2).

    spins: array, shape (batch, L), S^z +1 (up) / -1 (down),
           site index = x.
    L:     int (chain length).
    Returns: complex array, shape (batch,).
    """
    # A-sublattice = even sites x % 2 == 0
    A_mask = (jnp.arange(L) % 2 == 0).astype(jnp.int32)   # (L,), 1 on A else 0

    # up spins -> 0, down spins -> A_mask (counts only down spins on A)
    contrib = jnp.where(spins > 0, 0, A_mask)      # broadcasts to (batch, L)
    N_down_A = jnp.sum(contrib, axis=-1)           # (batch,)

    return (1j * jnp.pi * (N_down_A % 2)).astype(jnp.complex128)

def add_sign_rule(sign_fn, apply_fn, L):
    """
    Factory: wrap a wavefunction apply_fn so its log-amplitudes get an
    additive log-phase from `sign_fn`.

    sign_fn(spins, L) must return the FULL log-phase (i.e. i*theta), shape (batch,).
    apply_fn(params, lattice, *args, **kwargs) returns log-amplitudes.
    """
    def wrapped(params, state, *args, **kwargs):
        log_amps = apply_fn(params, state, *args, **kwargs)
        spins = jnp.atleast_2d(state.spins)
        return log_amps + sign_fn(spins, L)
    return wrapped