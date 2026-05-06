import functools


def apply(real_gr):
    _ver = int(real_gr.__version__.split('.')[0])
    if _ver < 4:
        return  # 3.41.2 handles _js= natively

    try:
        import gradio.component_meta as cm
    except ImportError:
        return

    # Wrap every event_trigger function stored on component classes so that
    # _js= is translated to js= and show_progress bool is normalized to str.
    def _make_compat_trigger(orig):
        if getattr(orig, '_compat_wrapped', False):
            return orig

        @functools.wraps(orig)
        def wrapped(block, fn=None, inputs=None, outputs=None, **kw):
            if '_js' in kw:
                kw.setdefault('js', kw.pop('_js'))
            if isinstance(kw.get('show_progress'), bool):
                kw['show_progress'] = 'full' if kw['show_progress'] else 'hidden'
            return orig(block, fn, inputs, outputs, **kw)

        wrapped._compat_wrapped = True
        return wrapped

    def _patch_class(cls):
        for ev in getattr(cls, 'EVENTS', []):
            name = str(ev)
            if name in cls.__dict__:
                orig = cls.__dict__[name]
                if callable(orig) and not getattr(orig, '_compat_wrapped', False):
                    setattr(cls, name, _make_compat_trigger(orig))

    # Retroactively patch all already-defined Block subclasses (covers both
    # Component subclasses and layout blocks like Tab, Tabs, Accordion, etc.)
    try:
        from gradio.blocks import Block
        seen = set()

        def _walk(c):
            if c in seen:
                return
            seen.add(c)
            _patch_class(c)
            for sub in c.__subclasses__():
                _walk(sub)

        _walk(Block)
    except ImportError:
        pass

    # Hook ComponentMeta.__new__ so extension-defined components get the same fix
    orig_meta_new = cm.ComponentMeta.__new__

    def patched_meta_new(mcs, name, bases, attrs):
        result = orig_meta_new(mcs, name, bases, attrs)
        _patch_class(result)
        return result

    cm.ComponentMeta.__new__ = patched_meta_new

    # Dependency.then/success/failure are partials created per-instance at
    # Dependency.__init__ time — not class attributes, so the above loop misses
    # them. Patch Dependency.__init__ to wrap each partial after construction.
    try:
        from gradio.events import Dependency
    except ImportError:
        return

    orig_dep_init = Dependency.__init__

    def patched_dep_init(self, *args, **kwargs):
        orig_dep_init(self, *args, **kwargs)
        for attr in ('then', 'success', 'failure'):
            orig_partial = getattr(self, attr, None)
            if orig_partial is not None and callable(orig_partial):
                setattr(self, attr, _make_compat_partial(orig_partial))

    Dependency.__init__ = patched_dep_init


def _make_compat_partial(orig_partial):
    @functools.wraps(orig_partial)
    def wrapped(fn=None, inputs=None, outputs=None, **kw):
        if '_js' in kw:
            kw.setdefault('js', kw.pop('_js'))
        if isinstance(kw.get('show_progress'), bool):
            kw['show_progress'] = 'full' if kw['show_progress'] else 'hidden'
        return orig_partial(fn, inputs, outputs, **kw)
    return wrapped
