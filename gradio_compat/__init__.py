"""
gradio_compat — sys.modules shim that makes Gradio 6.x look like 3.41.2.

Activated by modules_forge/initialization.py before any other module imports gradio.
"""
import sys

_REAL_KEY = "_gradio_real"

if _REAL_KEY not in sys.modules:
    # Temporarily remove ourselves so the real gradio can be imported
    _was = sys.modules.pop("gradio", None)
    import gradio as _real_mod
    sys.modules[_REAL_KEY] = _real_mod
    if _was is not None:
        sys.modules["gradio"] = _was

_real = sys.modules[_REAL_KEY]

# Apply all compatibility patches IN PLACE on the real gradio objects
from gradio_compat._patches_blocks     import apply as _pb; _pb(_real)  # noqa: E702
from gradio_compat._patches_components import apply as _pc; _pc(_real)  # noqa: E702
from gradio_compat._patches_events     import apply as _pe; _pe(_real)  # noqa: E702
from gradio_compat._patches_themes     import apply as _pt; _pt(_real)  # noqa: E702
from gradio_compat._patches_utils      import apply as _pu; _pu(_real)  # noqa: E702
from gradio_compat._patches_processing import apply as _pp; _pp(_real)  # noqa: E702
from gradio_compat._patches_preload    import apply as _pl; _pl(_real)  # noqa: E702

# Re-export everything from the real gradio so this module IS gradio
from _gradio_real import *  # noqa: F401, F403

__version__ = _real.__version__

from gradio_compat import _submodules
_SUBMODULE_NAMES = _submodules.NAMES


def _submodule(name):
    return _submodules.get(name)


def __getattr__(name):
    # Try the real gradio's namespace first
    try:
        return getattr(_real, name)
    except AttributeError:
        pass
    # Fall back to submodule lookup. Must temporarily restore sys.modules['gradio']
    # to the real package so importlib resolves from the real filesystem path.
    sub_key = f'gradio.{name}'
    if sub_key in sys.modules:
        return sys.modules[sub_key]
    import importlib as _il
    _prev = sys.modules.get('gradio')
    sys.modules['gradio'] = _real
    try:
        mod = _il.import_module(sub_key)
        sys.modules[sub_key] = mod
        return mod
    except ImportError:
        raise AttributeError(name)
    finally:
        if _prev is not None:
            sys.modules['gradio'] = _prev
