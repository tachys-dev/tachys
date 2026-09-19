"""Shared setup for the whole test suite.

Puts the test helpers (`testing_ansatz`, `testing_configs`) on `sys.path` so
every test module can import them regardless of which sub-directory it lives
in, and turns on 64-bit floats before any test module is imported.  `tachys`
enables x64 itself on import, but the helpers hand out float64 arrays and must
not depend on which module happened to be imported first.
"""
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

_TESTS_ROOT = str(Path(__file__).resolve().parent)
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)
