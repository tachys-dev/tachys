import jax
jax.config.update("jax_enable_x64", True)

from tachys.parallel import rank, n_devices, MASTER

if rank == MASTER:
    import sys
    _kind      = jax.devices()[0].platform.upper()
    _model     = jax.devices()[0].device_kind
    _n_nodes   = jax.process_count()
    _dev_node  = jax.local_device_count()
    _title     = "⚡ tachys"
    _dev_str   = f"{n_devices} × {_kind}  ({_model})"
    _node_str  = f"{_n_nodes} node{'s' if _n_nodes > 1 else ''}  ·  {_dev_node} {_kind}/node"

    _USE_COLOR = sys.stdout.isatty()
    _C = "\033[36m" if _USE_COLOR else ""
    _B = "\033[1m"  if _USE_COLOR else ""
    _R = "\033[0m"  if _USE_COLOR else ""

    _cw  = max(len(_title), len(_dev_str), len(_node_str))
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
    print(f"{_C}╚{'═' * _w}╝{_R}")

