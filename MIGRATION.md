# Gradio 3.41.2 → 6.x Migration Reference

## Overview

This document captures all research needed to implement a `gradio_compat/` compatibility shim that allows the project to run on Gradio 6.x without modifying the 91+ files that `import gradio as gr`.

**Current pin**: `gradio==3.41.2`, `gradio_client==0.5.0`  
**Target**: `gradio>=6.0,<7.0`  
**Strategy**: `sys.modules` injection — shim intercepts `import gradio`, passes stable APIs through via `__getattr__`, translates only broken APIs in-place.

---

## Codebase Gradio Usage Inventory

### Import patterns found

| Pattern | Files |
|---|---|
| `import gradio as gr` | 91 files across `modules/`, `extensions-builtin/`, `extensions/` |
| `from gradio import processing_utils` | `modules/gradio_extensons.py:2` |
| `import gradio.utils` | `modules/ui.py` |
| `import gradio.components` | `modules/ui_tempdir.py:6` |

### Core UI files (highest risk)

- `modules/ui.py` — main Gradio UI construction, 1000+ lines
- `modules/gradio_extensons.py` — monkey-patches Gradio internals (see below)
- `modules/ui_tempdir.py` — patches `IOComponent.pil_to_temp_file`
- `modules/ui_components.py` — custom components using `gr.components.Form`
- `modules/shared_gradio_themes.py` — theme loading via `gr.themes.ThemeClass`
- `webui.py:22` — `shared.demo.queue(64)` positional arg

---

## Monkey-Patch Sites in Codebase

These are the highest-risk locations because they reach into Gradio internals.

### `modules/gradio_extensons.py:144-148`

```python
original_IOComponent_init    = patches.patch(__name__, obj=gr.components.IOComponent,  field="__init__",        replacement=IOComponent_init)
original_Block_get_config    = patches.patch(__name__, obj=gr.blocks.Block,            field="get_config",      replacement=Block_get_config)
original_BlockContext_init   = patches.patch(__name__, obj=gr.blocks.BlockContext,     field="__init__",        replacement=BlockContext_init)
original_Blocks_get_config_file = patches.patch(__name__, obj=gr.blocks.Blocks,       field="get_config_file", replacement=Blocks_get_config_file)
original_Image_init          = patches.patch(__name__, obj=gr.components.Image,       field="__init__",        replacement=Image_init_extension)
```

Also at line 14: `comp.get_block_name()` — called on every component during init.

The `Image_init_extension` / `Image_custom_preprocess` functions reference these `gr.Image` internal attributes (all likely removed in 5.x+):
- `self.tool` (e.g. `"sketch"`)
- `self.source` (e.g. `"upload"`, `"webcam"`)
- `self.shape`
- `self.invert_colors`
- `self.mirror_webcam`
- `self.image_mode`
- `self._format_image(im)`

### `modules/ui_tempdir.py:61`

```python
gradio.components.IOComponent.pil_to_temp_file = save_pil_to_file
```

`pil_to_temp_file` was removed from `IOComponent` in Gradio 4.x. The temp file system was completely redesigned (`FileData`, `gradio.utils.save_to_cache`).

### `modules/ui_components.py:6`

```python
class FormComponent:
    def get_expected_parent(self):
        return gr.components.Form   # Form removed in 4.x
```

Custom components that inherit `FormComponent`:
- `ToolButton(FormComponent, gr.Button)`
- `FormRow(FormComponent, gr.Row)`
- `FormColumn(FormComponent, gr.Column)`
- `FormGroup(FormComponent, gr.Group)`
- `FormHTML(FormComponent, gr.HTML)`
- `FormColorPicker(FormComponent, gr.ColorPicker)`
- `DropdownMulti(FormComponent, gr.Dropdown)`
- `DropdownEditable(FormComponent, gr.Dropdown)`

These classes are defined at module import time, so the `gr.components.Form` lookup happens immediately.

### `modules/shared_gradio_themes.py`

```python
shared.gradio_theme = gr.themes.Default(font=[...], font_mono=[...])
shared.gradio_theme = gr.themes.ThemeClass.load(theme_cache_path)   # ThemeClass → Base in 6.x
shared.gradio_theme = gr.themes.ThemeClass.from_hub(theme_name)
shared.gradio_theme.dump(theme_cache_path)
shared.gradio_theme.sd_webui_modal_lightbox_toolbar_opacity = value  # dynamic attr
shared.gradio_theme.sd_webui_modal_lightbox_icon_opacity = value
```

### `webui.py` (via `modules/shared.py`)

```python
shared.demo.queue(64)   # positional concurrency_count — changed in 4.x
```

---

## Breaking Changes: Gradio 3.41.2 → 6.x

### Removed/moved classes

| Symbol | 3.41.2 location | 6.x status | Fix |
|---|---|---|---|
| `gr.components.IOComponent` | `gradio.components` | Moved to `gradio.components.base` | Add alias on `gradio.components` |
| `gr.blocks.BlockContext` | `gradio.blocks` | May have moved | Ensure accessible |
| `gr.components.Form` | `gradio.components` | Removed in 4.x | Provide `BlockContext` subclass stub |
| `gr.Box` | `gradio.layouts` | Removed in 4.x | Alias to `gr.Group` |
| `gr.deprecation.GradioDeprecationWarning` | `gradio.deprecation` | Module may not exist | Stub module with `= DeprecationWarning` |
| `Block.get_block_name()` | `gradio.blocks.Block` | Removed in 4.x | Add back as `type(self).__name__.lower()` |
| `Block.get_config_file()` on `Blocks` | `gradio.blocks.Blocks` | Renamed to `get_config` in 4.x | Add alias |
| `IOComponent.pil_to_temp_file` | `gradio.components.IOComponent` | Removed in 4.x | Stub slot to allow assignment |
| `Dropdown.update()` (class method) | `gradio.components.Dropdown` | Removed in 4.x | `cls.update = staticmethod(gr.update)` |

### Changed call signatures

| Call | 3.41.2 | 6.x | Fix |
|---|---|---|---|
| `demo.queue(64)` | First positional = `concurrency_count` | No positional args; use `max_size=` | Wrap positional → `max_size=` |
| `demo.launch(show_api=False)` | `show_api` param exists | Removed | Silently drop unknown kwargs |
| `event.click(_js="...")` | `_js=` aliased to `js=` internally | Alias removed | Wrap `__call__` to translate `_js` → `js` |
| `event.click(show_progress=False)` | Bool accepted | String only: `"hidden"`/`"full"` | Wrap to normalize bools |
| `gr.themes.ThemeClass` | Exists | Renamed to `gr.themes.Base` | Alias `ThemeClass = Base` |
| `gr.themes.ThemeClass.load(path)` | `.load()` / `.dump()` / `.from_hub()` | May have changed | Proxy through |

### Image component internals (5.x+)

The `gr.Image` component was redesigned in Gradio 5.x. These instance attributes no longer exist:

```
self.tool          # was: "sketch", "color-sketch", None
self.source        # was: "upload", "webcam", "canvas"
self.shape         # was: (w, h) tuple or None
self.invert_colors # was: bool
self.mirror_webcam # was: bool
self.image_mode    # was: "RGB", "L", etc.
self._format_image # was: method returning str path
```

The `preprocess()` mechanism also changed entirely. This is the deepest incompatibility — `Image_custom_preprocess` in `gradio_extensons.py` will need its own compat wrapper.

### `processing_utils` API

Still present in Gradio 6.x but some functions may have changed:

| Function | Status |
|---|---|
| `processing_utils.decode_base64_to_image(x)` | Likely still exists |
| `processing_utils.resize_and_crop(im, shape)` | Likely still exists |

Stub if missing.

### `gradio.utils` functions

Overridden in `modules/ui.py`:

```python
gradio.utils.version_check = lambda: None
gradio.utils.get_local_ip_address = lambda: '127.0.0.1'
```

Both may not exist in 6.x — add stubs before the assignment.

### `gr.update()` return type

`modules/infotext_utils.py` does:

```python
type_of_gr_update = type(gr.update())
```

Then uses `isinstance(x, type_of_gr_update)` throughout. In 3.41.2, `gr.update()` returns a plain `dict` with `__type__: "update"`. In 6.x it returns a different type. The shim must ensure `gr.update()` still returns something `isinstance`-compatible.

---

## Event Listener Patterns in Use

All of these keyword args appear across the 91 Gradio files:

```python
component.click(
    fn=callback,
    inputs=[...],
    outputs=[...],
    show_progress=False,       # bool → needs normalization in 6.x
    queue=False,               # still accepted
    _js="javascript_string"    # renamed to js= in 6.x
)

component.change(fn=..., inputs=..., outputs=..., show_progress=False, queue=False)
component.submit(fn=..., inputs=..., outputs=...)
component.release(fn=..., inputs=..., outputs=..., show_progress=False, _js="...")
component.select(fn=..., inputs=..., outputs=..., show_progress=False)
component.blur(...)
component.upload(fn=..., inputs=..., outputs=...)

# Chaining
button.click(...).then(
    fn=next_callback,
    inputs=[...],
    outputs=[...],
    show_progress="hidden",
    _js="..."
)

interface.load(fn=..., inputs=[], outputs=[...], queue=False, show_progress=False)
```

---

## Shim Architecture

### Activation point

`modules_forge/initialization.py` — imported at `webui.py:15`, before `initialize.imports()` at `webui.py:24` where `import gradio` first runs.

Insert at the **very top** of `modules_forge/initialization.py` (before existing `import os, sys`):

```python
import sys as _sys, os as _os
_compat_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _compat_root not in _sys.path:
    _sys.path.insert(0, _compat_root)
import gradio_compat as _gc
_sys.modules['gradio'] = _gc
for _sub in _gc._SUBMODULE_NAMES:
    _sys.modules[f'gradio.{_sub}'] = _gc._submodule(_sub)
del _gc, _sys, _os, _compat_root
```

### `sys.modules` submodule injection list

All of these must be injected so `import gradio.utils` / `from gradio.blocks import Block` style imports resolve correctly:

```
gradio.blocks
gradio.components
gradio.components.base
gradio.utils
gradio.routes
gradio.themes
gradio.themes.base
gradio.events
gradio.processing_utils
gradio.deprecation        ← stub if missing
gradio.layouts
gradio.helpers
gradio.data_classes
```

### Bootstrap — avoiding circular import

```python
# gradio_compat/__init__.py
import sys

_REAL_KEY = "_gradio_real"
if _REAL_KEY not in sys.modules:
    _was = sys.modules.pop("gradio", None)   # temporarily remove ourselves
    import gradio as _real                    # loads the real Gradio package
    sys.modules[_REAL_KEY] = _real
    if _was is not None:
        sys.modules["gradio"] = _was
_real = sys.modules[_REAL_KEY]

# Apply all patches IN PLACE on the real gradio objects
from gradio_compat._patches_blocks      import apply as _pb; _pb(_real)
from gradio_compat._patches_components  import apply as _pc; _pc(_real)
from gradio_compat._patches_events      import apply as _pe; _pe(_real)
from gradio_compat._patches_themes      import apply as _pt; _pt(_real)
from gradio_compat._patches_utils       import apply as _pu; _pu(_real)
from gradio_compat._patches_processing  import apply as _pp; _pp(_real)

from _gradio_real import *   # re-export everything

__version__ = _real.__version__

from gradio_compat import _submodules
_SUBMODULE_NAMES = _submodules.NAMES
def _submodule(name): return _submodules.get(name)

def __getattr__(name):
    return getattr(_real, name)   # zero-overhead pass-through
```

### File structure

```
gradio_compat/
    __init__.py              # proxy root
    _patches_blocks.py       # Blocks.queue(), Blocks.launch(), get_config_file alias
    _patches_components.py   # IOComponent, Form stub, gr.Box, cls.update(), get_block_name, Image attrs
    _patches_events.py       # _js→js, show_progress bool→str
    _patches_processing.py   # processing_utils stubs
    _patches_themes.py       # ThemeClass alias, permissive __setattr__
    _patches_utils.py        # gradio.utils stubs
    _submodules.py           # sys.modules injection helpers
```

### Patch implementations (key patterns)

**`_patches_blocks.py`**

```python
def apply(real_gr):
    _ver = int(real_gr.__version__.split('.')[0])

    # queue(64) positional → queue(max_size=64)
    orig_queue = real_gr.Blocks.queue
    @functools.wraps(orig_queue)
    def patched_queue(self, concurrency_count_or_max_size=None, **kw):
        if _ver >= 4 and concurrency_count_or_max_size is not None:
            kw.setdefault('max_size', concurrency_count_or_max_size)
        elif concurrency_count_or_max_size is not None:
            kw['concurrency_count'] = concurrency_count_or_max_size
        if _ver >= 5:
            kw['default_concurrency_limit'] = kw.pop('concurrency_count', kw.get('default_concurrency_limit', 1))
        return orig_queue(self, **kw)
    real_gr.Blocks.queue = patched_queue

    # launch() — silently drop removed kwargs (show_api, encrypt, file_directories)
    orig_launch = real_gr.Blocks.launch
    @functools.wraps(orig_launch)
    def patched_launch(self, **kw):
        if _ver >= 4:
            for dead in ('show_api', 'encrypt', 'file_directories'):
                kw.pop(dead, None)
        return orig_launch(self, **kw)
    real_gr.Blocks.launch = patched_launch

    # get_config_file alias
    if not hasattr(real_gr.Blocks, 'get_config_file') and hasattr(real_gr.Blocks, 'get_config'):
        real_gr.Blocks.get_config_file = real_gr.Blocks.get_config
```

**`_patches_components.py`**

```python
def apply(real_gr):
    _ensure_ioc_accessible()
    _ensure_form_class()
    _ensure_get_block_name()
    _ensure_class_update_methods(real_gr)
    _ensure_box_alias(real_gr)
    _ensure_pil_to_temp_file_slot()
    _stub_image_attrs(real_gr)

def _ensure_ioc_accessible():
    import gradio.components as comp
    if not hasattr(comp, 'IOComponent'):
        try:
            from gradio.components.base import IOComponent
            comp.IOComponent = IOComponent
        except ImportError:
            pass

def _ensure_form_class():
    import gradio.components as comp
    if not hasattr(comp, 'Form'):
        try:
            from gradio.blocks import BlockContext
            class Form(BlockContext):
                pass
            comp.Form = Form
        except ImportError:
            pass

def _ensure_get_block_name():
    try:
        from gradio.blocks import Block
        if not hasattr(Block, 'get_block_name'):
            Block.get_block_name = lambda self: type(self).__name__.lower()
    except ImportError:
        pass

def _ensure_class_update_methods(real_gr):
    targets = [real_gr.Dropdown, real_gr.File, real_gr.Image, real_gr.Gallery,
               real_gr.Textbox, real_gr.Slider, real_gr.Button, real_gr.HTML,
               real_gr.Audio, real_gr.Video, real_gr.Radio, real_gr.Checkbox]
    for cls in targets:
        if not callable(getattr(cls, 'update', None)):
            cls.update = staticmethod(real_gr.update)

def _ensure_box_alias(real_gr):
    if not hasattr(real_gr, 'Box'):
        real_gr.Box = real_gr.Group

def _ensure_pil_to_temp_file_slot():
    import gradio.components as comp
    if hasattr(comp, 'IOComponent') and not hasattr(comp.IOComponent, 'pil_to_temp_file'):
        comp.IOComponent.pil_to_temp_file = None

def _stub_image_attrs(real_gr):
    orig_init = real_gr.Image.__init__
    @functools.wraps(orig_init)
    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # Inject 3.x attrs if missing
        for attr, default in [('tool', None), ('source', 'upload'), ('shape', None),
                               ('invert_colors', False), ('mirror_webcam', False), ('image_mode', 'RGB')]:
            if not hasattr(self, attr):
                setattr(self, attr, default)
        if not hasattr(self, '_format_image'):
            self._format_image = lambda im: im
    real_gr.Image.__init__ = patched_init
```

**`_patches_events.py`**

```python
def apply(real_gr):
    _ver = int(real_gr.__version__.split('.')[0])
    if _ver < 4:
        return  # 3.41.2 handles _js= natively
    try:
        from gradio.events import EventListenerMethod
    except ImportError:
        return
    orig = EventListenerMethod.__call__
    @functools.wraps(orig)
    def patched(self, fn=None, inputs=None, outputs=None, **kw):
        if '_js' in kw:
            kw.setdefault('js', kw.pop('_js'))
        if 'show_progress' in kw and isinstance(kw['show_progress'], bool):
            kw['show_progress'] = 'full' if kw['show_progress'] else 'hidden'
        return orig(self, fn, inputs, outputs, **kw)
    EventListenerMethod.__call__ = patched
```

**`_patches_themes.py`**

```python
def apply(real_gr):
    import gradio.themes as t
    if not hasattr(t, 'ThemeClass'):
        t.ThemeClass = getattr(t, 'Base', None) or getattr(t, 'Default', None)
    if not hasattr(t, 'Default') and hasattr(t, 'Base'):
        t.Default = t.Base
    # Allow arbitrary dynamic attrs on theme instances
    base_cls = getattr(t, 'ThemeClass', None) or getattr(t, 'Base', None)
    if base_cls:
        orig_set = base_cls.__setattr__
        def permissive_set(self, name, value):
            try:
                orig_set(self, name, value)
            except (AttributeError, TypeError):
                object.__setattr__(self, name, value)
        base_cls.__setattr__ = permissive_set
```

**`_patches_utils.py`**

```python
def apply(real_gr):
    import gradio.utils as u
    if not hasattr(u, 'version_check'):
        u.version_check = lambda: None
    if not hasattr(u, 'get_local_ip_address'):
        u.get_local_ip_address = lambda: '127.0.0.1'
```

**`_patches_processing.py`**

```python
def apply(real_gr):
    try:
        import gradio.processing_utils as pu
    except ImportError:
        return
    # Stub any missing functions used by gradio_extensons.py
    if not hasattr(pu, 'decode_base64_to_image'):
        import base64, io
        from PIL import Image
        def decode_base64_to_image(s):
            s = s.split(',', 1)[-1]
            return Image.open(io.BytesIO(base64.b64decode(s)))
        pu.decode_base64_to_image = decode_base64_to_image
    if not hasattr(pu, 'resize_and_crop'):
        from PIL import Image as _PILImage
        def resize_and_crop(img, size):
            return img.resize(size, _PILImage.LANCZOS)
        pu.resize_and_crop = resize_and_crop
```

**`_submodules.py`**

```python
import importlib, types

NAMES = [
    'blocks', 'components', 'components.base', 'utils', 'routes',
    'themes', 'themes.base', 'events', 'processing_utils',
    'deprecation', 'layouts', 'helpers', 'data_classes',
]

def get(name):
    try:
        return importlib.import_module(f'gradio.{name}')
    except ImportError:
        stub = types.ModuleType(f'gradio.{name}')
        if name == 'deprecation':
            stub.GradioDeprecationWarning = DeprecationWarning
        return stub
```

---

## Files to Create / Modify

### Create

| File | Purpose |
|---|---|
| `gradio_compat/__init__.py` | Proxy root, bootstrap, `__getattr__` |
| `gradio_compat/_patches_blocks.py` | `queue()`, `launch()`, `get_config_file` |
| `gradio_compat/_patches_components.py` | `IOComponent`, `Form`, `gr.Box`, `cls.update()`, `get_block_name`, Image attrs |
| `gradio_compat/_patches_events.py` | `_js→js`, `show_progress` bool→str |
| `gradio_compat/_patches_processing.py` | `processing_utils` stubs |
| `gradio_compat/_patches_themes.py` | `ThemeClass` alias, permissive `__setattr__` |
| `gradio_compat/_patches_utils.py` | `gradio.utils` stubs |
| `gradio_compat/_submodules.py` | `sys.modules` injection helpers |

### Modify

| File | Change |
|---|---|
| `modules_forge/initialization.py` | 8-line shim activation block at very top of file |
| `requirements.txt` | `gradio==3.41.2` → `gradio>=6.0,<7.0` |

---

## Known Hard Risks

| Risk | Impact | Status |
|---|---|---|
| `gr.Image` internal attrs removed in 5.x | `Image_custom_preprocess` in `gradio_extensons.py` will fail | Mitigate: stub attrs in `Image.__init__` wrapper |
| `pil_to_temp_file` removed | PNG metadata silently lost; temp file serving may break | Stub slot; real fix requires `FileData` compat layer (Phase 2) |
| `Blocks.get_config_file` structure changed | `Blocks_get_config_file` patch iterates `config["components"]` — structure may differ | Defensive iteration with `.get()` checks |
| `gr.components.Form` in MRO | `TypeError` at class definition time (before any runtime) | Provide `BlockContext` subclass stub — must be in place before `ui_components.py` imports |
| `type(gr.update())` in `infotext_utils.py` | All `isinstance(x, type_of_gr_update)` checks break | Ensure `gr.update()` returns a `dict` subclass |
| `gradio.routes.templates` replacement | `ui_gradio_extensions.py` replaces Starlette template handler | May need its own investigation; Starlette API changed significantly in Gradio 4+ |
| `errors.check_versions()` | Prints warning if `gradio.__version__ != "3.41.2"` | Document as expected; suppress via `--skip-version-check` |

---

## Verification Checklist

```bash
# 1. Import smoke test
python -c "import sys; sys.path.insert(0, '.'); import gradio_compat; import sys; sys.modules['gradio'] = gradio_compat; import gradio as gr; print(gr.__version__)"

# 2. Launch server (CPU mode, no models)
python launch.py --skip-prepare-environment --skip-torch-cuda-test \
  --no-half --do-not-download-clip --always-cpu \
  --skip-version-check

# 3. API tests
python -m pytest -vv test/

# Check for:
# - No ImportError or AttributeError at startup
# - Gradio UI renders at port 7860 without JS console errors
# - gr.components.Form accessible
# - gr.Box accessible (alias to gr.Group)
# - gr.themes.ThemeClass accessible
# - gradio.deprecation.GradioDeprecationWarning accessible
# - queue(64) accepted without TypeError
```
