import functools


def apply(real_gr):
    _patch_component_meta(real_gr)
    _ensure_ioc_accessible()
    _ensure_form_class()
    _ensure_get_block_name()
    _ensure_class_update_methods(real_gr)
    _ensure_box_alias(real_gr)
    _ensure_pil_to_temp_file_slot()
    _stub_image_attrs(real_gr)
    _ensure_gr_update_dict_compat(real_gr)


_STRIPPED_KWARGS = frozenset({
    'tooltip', 'allow_preview', 'keep_aspect_ratio', 'brush_radius',
    'brush_color', 'mask_opacity', 'selectable',
})


def _patch_component_meta(real_gr):
    """Intercept all component instantiation to strip removed 3.x kwargs."""
    try:
        from gradio.component_meta import ComponentMeta
    except ImportError:
        return

    orig_call = ComponentMeta.__call__

    def patched_call(cls, *args, **kwargs):
        # Strip kwargs removed in 4.x+
        for kw in _STRIPPED_KWARGS:
            kwargs.pop(kw, None)
        # gr.Button(label=...) → gr.Button(value=...) when value not given
        if 'label' in kwargs:
            import inspect
            try:
                sig = inspect.signature(cls.__init__)
                params = sig.parameters
                if 'value' in params and 'label' not in params:
                    kwargs.setdefault('value', kwargs.pop('label'))
                else:
                    kwargs.pop('label', None) if 'label' not in params else None
            except (ValueError, TypeError):
                pass
        return orig_call(cls, *args, **kwargs)

    ComponentMeta.__call__ = patched_call


def _ensure_ioc_accessible():
    try:
        import gradio.components as comp
        if not hasattr(comp, 'IOComponent'):
            try:
                from gradio.components.base import IOComponent
                comp.IOComponent = IOComponent
            except ImportError:
                # Last resort: alias the base Component class
                try:
                    from gradio.components.base import Component
                    comp.IOComponent = Component
                except ImportError:
                    pass
    except ImportError:
        pass


def _ensure_form_class():
    try:
        import gradio.components as comp
        if not hasattr(comp, 'Form'):
            try:
                from gradio.blocks import BlockContext
                class Form(BlockContext):
                    pass
                comp.Form = Form
            except ImportError:
                # Stub: a plain object that can be used as a parent class
                class Form:
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
    component_names = [
        'Dropdown', 'File', 'Image', 'Gallery', 'Textbox', 'Slider',
        'Button', 'HTML', 'Audio', 'Video', 'Radio', 'Checkbox',
        'CheckboxGroup', 'Number', 'Label', 'Dataframe', 'ColorPicker',
    ]
    for name in component_names:
        cls = getattr(real_gr, name, None)
        if cls is None:
            continue
        if not callable(getattr(cls, 'update', None)):
            cls.update = staticmethod(real_gr.update)


def _ensure_box_alias(real_gr):
    if not hasattr(real_gr, 'Box'):
        real_gr.Box = getattr(real_gr, 'Group', None)


def _ensure_pil_to_temp_file_slot():
    try:
        import gradio.components as comp
        ioc = getattr(comp, 'IOComponent', None)
        if ioc is not None and not hasattr(ioc, 'pil_to_temp_file'):
            # Provide a slot so assignments don't AttributeError
            ioc.pil_to_temp_file = None
    except ImportError:
        pass


def _stub_image_attrs(real_gr):
    img_cls = getattr(real_gr, 'Image', None)
    if img_cls is None:
        return

    orig_init = img_cls.__init__

    @functools.wraps(orig_init)
    def patched_init(self, *args, **kwargs):
        # Strip 3.x-only kwargs that 6.x rejects
        for dead_kw in ('tool', 'source', 'shape', 'invert_colors',
                         'mirror_webcam', 'image_mode'):
            kwargs.pop(dead_kw, None)
        orig_init(self, *args, **kwargs)
        # Inject 3.x attrs if missing so gradio_extensons.py can read them
        for attr, default in [
            ('tool', None),
            ('source', 'upload'),
            ('shape', None),
            ('invert_colors', False),
            ('mirror_webcam', False),
            ('image_mode', 'RGB'),
        ]:
            if not hasattr(self, attr):
                object.__setattr__(self, attr, default)
        if not hasattr(self, '_format_image'):
            object.__setattr__(self, '_format_image', lambda im: im)

    real_gr.Image.__init__ = patched_init


def _ensure_gr_update_dict_compat(real_gr):
    """Ensure gr.update() returns a dict subclass so isinstance checks work."""
    orig_update = real_gr.update

    # If it already returns a plain dict, nothing to do — isinstance(x, dict) works.
    # If it returns a non-dict type, wrap it so the result IS a dict.
    sentinel = orig_update()
    if isinstance(sentinel, dict):
        return

    class _GrUpdateDict(dict):
        pass

    def patched_update(**kwargs):
        result = orig_update(**kwargs)
        d = _GrUpdateDict(result) if hasattr(result, '__iter__') else _GrUpdateDict()
        return d

    real_gr.update = patched_update
