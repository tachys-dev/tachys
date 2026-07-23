from typing import Callable, Optional

import optax


def linear_decay(eta0: float, eta_final: float, N_steps: int) -> Callable:
    """Linear decay from ``eta0`` down to ``eta_final`` over ``N_steps``.

    Returns a function of the absolute step, so it decays correctly across
    resumes when ``step`` is offset by ``start_step``.
    """
    def schedule(step):
        frac = min(step / N_steps, 1.0)
        return eta0 + (eta_final - eta0) * frac

    return schedule


def shifted_cosine_decay(init_value: float, decay_steps: int, min_value: Optional[float] = None) -> Callable:
    if min_value is None:
        min_value = init_value / 10 #* reduce the lr by a factor 10

    cos_dec = optax.cosine_decay_schedule(init_value=init_value - min_value, decay_steps=decay_steps)
    lr_func = lambda t: cos_dec(t) + min_value # shift the minimum value
    return lr_func
