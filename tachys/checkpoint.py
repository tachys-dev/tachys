import logging
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

from tachys.parallel import all_unshard, mesh

_PATH_BUF_LEN = 1024  # fixed-size buffer so broadcast_one_to_all sees equal shapes on every host


class _DropTemporaryPathWarning(logging.Filter):
    """Orbax's checkpoint deleter (CheckpointManagerOptions.max_to_keep pruning)
    scans every directory in the checkpoint root to locate the step it's about to
    delete, calling is_path_temporary() on each entry along the way (see
    orbax.checkpoint._src.path.step.find_step_path). Every already-finalized
    checkpoint it encounters "fails" that check and gets logged at WARNING level,
    even though "not a temporary path" is the expected, correct outcome for a
    normal, successfully-committed checkpoint. Harmless, but fires on every
    pruning step, so it's dropped here rather than left to alarm every save.
    """

    def filter(self, record):
        return "could not be identified as a temporary checkpoint path" not in record.getMessage()


logging.getLogger("absl").addFilter(_DropTemporaryPathWarning())


def _broadcast_str(s, is_source):
    buf = np.zeros(_PATH_BUF_LEN, dtype=np.uint8)
    if is_source:
        b = s.encode("utf-8")
        buf[: len(b)] = np.frombuffer(b, dtype=np.uint8)
    buf = multihost_utils.broadcast_one_to_all(buf, is_source=is_source)
    return bytes(buf).rstrip(b"\x00").decode("utf-8")


def resolve_checkpoint_settings(wandb_run, N_steps, rank, MASTER):
    """Compute (dir, every, keep) on rank MASTER from wandb_run and broadcast to all ranks.

    Every process must call this collectively (it runs multi-host broadcasts),
    even ranks where wandb_run is None. Returns (None, None, None) if no rank
    has an active wandb_run.
    """
    is_source = rank == MASTER and wandb_run is not None

    enabled = bool(multihost_utils.broadcast_one_to_all(np.array(is_source), is_source=is_source))
    if not enabled:
        return None, None, None

    dir_ = os.path.join(wandb_run.dir, "checkpoints") if is_source else ""
    every = int(wandb_run.config.get("checkpoint_every", N_steps)) if is_source else 0
    keep = int(wandb_run.config.get("checkpoint_keep", 1)) if is_source else 0

    dir_ = _broadcast_str(dir_, is_source)
    every = int(multihost_utils.broadcast_one_to_all(np.array(every), is_source=is_source))
    keep = int(multihost_utils.broadcast_one_to_all(np.array(keep), is_source=is_source))
    return dir_, every, keep


def build_checkpoint_manager(directory, save_interval_steps, max_to_keep):
    # Orbax's default save_decision_policy always saves on the very first call
    # (its InitialSavePolicy) regardless of save_interval_steps, so a run barely
    # past step 0 would otherwise get a checkpoint. FixedIntervalPolicy alone
    # only saves every save_interval_steps steps; the explicit force=True on the
    # final training step (see ground_state_training.train) still always saves.
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=save_interval_steps,
        max_to_keep=max_to_keep,
        save_decision_policy=ocp.checkpoint_managers.FixedIntervalPolicy(save_interval_steps),
    )
    return ocp.CheckpointManager(directory, options=options)


def get_last_step(checkpoint_dir):
    manager = build_checkpoint_manager(checkpoint_dir, save_interval_steps=1, max_to_keep=1)
    step = manager.latest_step()
    manager.close()
    return step


def save_training_checkpoint(manager, step, key, state, params, opt_state, force=False):
    """Checkpoint params, key, and the mutable parts of state/opt_state as one atomic
    composite checkpoint.

    The whole ``state`` pytree is saved — foundation-model
    states (``SpinFoundationState``/``FermionFoundationState``) carry additional data
    fields alongside ``config`` (``system_couplings``, ``system_ids``) that must round-
    trip too; saving only ``.config`` would silently drop them on restore. Static
    fields (``pytree_node=False``, e.g. ``lattice``, ``N_mc``, ``n_systems``) aren't
    data to begin with, so there's nothing to serialize there regardless. ``state`` and
    ``opt_state`` are each wrapped in a single-entry dict: orbax's PyTreeCheckpointHandler
    errors on a bare top-level array (ambiguous truth-value check) and on an empty
    top-level pytree (e.g. SR's zero-field ``SRState()``) — wrapping avoids both.

    Must be called collectively by every process (no rank guard) so orbax can write
    each host's own shards of sharded arrays.

    ``key`` is saved as its raw bit representation (``jax.random.key_data``) through
    the same generic ``PyTreeSave`` path as everything else, not via orbax's
    ``JaxRandomKeySave``. Two reasons:
    - ``JaxRandomKeySave``'s handler hardcodes ``create_directories_asynchronously=
      False`` internally, which forces every save to block the main thread waiting
      on directory-creation signals (the "_SignalingThread.join() ... will slow
      down blocking save times" warning) instead of using the outer handler's async
      directory creation like the other items.
    - The raw key is also unsharded onto the global mesh first: unlike ``params``/
      ``state``/``opt_state`` (which pick up a proper multi-host
      ``NamedSharding`` by passing through jit+shard_map calls every step), ``key``
      only ever goes through plain ``jax.random.split`` and so stays on whatever
      single local device it started on. Orbax refuses to serialize such a "host
      local" array in a multi-host run, so ``all_unshard`` promotes it to a
      replicated global array first (every host holds the same key value, so this
      is just a sharding change, not a data change).
    """
    key_data = all_unshard(jax.random.key_data(key))
    args = ocp.args.Composite(
        params=ocp.args.PyTreeSave(params),
        opt_state=ocp.args.PyTreeSave({"opt_state": opt_state}),
        state=ocp.args.PyTreeSave({"state": state}),
        key=ocp.args.PyTreeSave({"key_data": key_data}),
    )
    return manager.save(step, args=args, force=force)


def load_checkpoint(checkpoint_dir, state_template, opt_state_template, params_template=None, step=None):
    """Restore params, opt_state, state and key from a checkpoint directory written by
    save_training_checkpoint.

    ``state_template`` and ``opt_state_template`` supply the pieces that aren't
    serialized (static fields like ``state.lattice``/``state.N_mc`` and the opt_state
    NamedTuple type respectively) — the restored state is built by using
    ``state_template`` as the structural template for orbax's PyTree restore, so its
    static fields are preserved while every data field (``config``, plus any
    foundation-model extras like ``system_couplings``/``system_ids``) is overwritten
    with the checkpointed values.
    ``params_template`` is optional: omit it to recover params as a plain dict
    inferred from the checkpoint's own metadata (fine here since wf.params is already
    a plain dict), at the cost of orbax falling back to the sharding recorded at save
    time for that item only (a UserWarning we suppress, since it's harmless for a
    same-topology resume). Everything else is restored directly onto the *current*
    global ``mesh`` (from ``tachys.parallel``, built from whatever devices are live
    right now) rather than the sharding recorded at save time, so those checkpoints
    are portable across topologies (e.g. resuming with a different device/host count
    than they were saved with) instead of being tied to the original mesh.

    ``params``/``opt_state``/``key`` are restored fully replicated (``P()``) — every
    device holds a full copy, matching how the training step's ``shard_map`` calls
    expect them. ``state`` (every per-walker data field, all sharing the same leading
    MC-batch axis) is restored partitioned along the mesh's ``'i'`` axis (``P('i')``)
    instead: unlike a fully-replicated target, which ``shard_map`` can pick up from a
    single device automatically, it requires an already-partitioned array for the
    batch axis and raises "Received incompatible devices" if handed one sitting on a
    single device — restoring straight onto ``P('i')`` avoids ever creating that
    intermediate, wrongly-sharded array.
    """
    manager = build_checkpoint_manager(checkpoint_dir, save_interval_steps=1, max_to_keep=1)
    replicated = NamedSharding(mesh, P())
    partitioned = NamedSharding(mesh, P('i'))
    restore_args = lambda template, sharding: jax.tree.map(lambda _: ocp.ArrayRestoreArgs(sharding=sharding), template)

    if params_template is None:
        params_args = ocp.args.PyTreeRestore(item=None)
    else:
        params_args = ocp.args.PyTreeRestore(item=params_template, restore_args=restore_args(params_template, replicated))

    opt_state_item = {"opt_state": opt_state_template}
    state_item = {"state": state_template}
    key_item = {"key_data": jnp.zeros((2,), dtype=jnp.uint32)}
    args = ocp.args.Composite(
        params=params_args,
        opt_state=ocp.args.PyTreeRestore(item=opt_state_item, restore_args=restore_args(opt_state_item, replicated)),
        state=ocp.args.PyTreeRestore(item=state_item, restore_args=restore_args(state_item, partitioned)),
        key=ocp.args.PyTreeRestore(item=key_item, restore_args=restore_args(key_item, replicated)),
    )
    with warnings.catch_warnings():
        if params_template is None:
            warnings.filterwarnings("ignore", category=UserWarning,
                                     message="Sharding info not provided when restoring.*")
        restored = manager.restore(step, args=args)
    manager.wait_until_finished()
    manager.close()

    state = restored.state["state"]
    key = jax.random.wrap_key_data(restored.key["key_data"])
    return restored.params, restored.opt_state["opt_state"], state, key
