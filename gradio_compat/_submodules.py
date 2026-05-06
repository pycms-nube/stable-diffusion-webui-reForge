import importlib
import sys
import types

NAMES = [
    'blocks',
    'components',
    'components.base',
    'utils',
    'routes',
    'themes',
    'themes.base',
    'events',
    'processing_utils',
    'deprecation',
    'layouts',
    'helpers',
    'data_classes',
    'http_server',
    'route_utils',
    'queueing',
    'server',
    'image_utils',
    'mcp',
    '_simple_templates',
    'templates',
]


def get(name):
    full = f'gradio.{name}'
    if full in sys.modules:
        return sys.modules[full]
    # Temporarily restore real gradio so importlib resolves from the real
    # filesystem path, not gradio_compat's directory.
    real = sys.modules.get('_gradio_real')
    compat = sys.modules.get('gradio')
    if real is not None:
        sys.modules['gradio'] = real
    try:
        mod = importlib.import_module(full)
        sys.modules[full] = mod
        return mod
    except ImportError:
        stub = types.ModuleType(full)
        if name == 'deprecation':
            stub.GradioDeprecationWarning = DeprecationWarning
        sys.modules[full] = stub
        return stub
    finally:
        if compat is not None:
            sys.modules['gradio'] = compat
        elif real is not None and sys.modules.get('gradio') is real:
            sys.modules.pop('gradio', None)
