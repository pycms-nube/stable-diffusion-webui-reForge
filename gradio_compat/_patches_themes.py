def apply(real_gr):
    import gradio.themes as t

    # ThemeClass → Base in 4.x+
    if not hasattr(t, 'ThemeClass'):
        t.ThemeClass = getattr(t, 'Base', None) or getattr(t, 'Default', None)

    if not hasattr(t, 'Default') and hasattr(t, 'Base'):
        t.Default = t.Base

    # Allow arbitrary dynamic attrs on theme instances (sd_webui_modal_lightbox_* etc.)
    base_cls = getattr(t, 'ThemeClass', None) or getattr(t, 'Base', None)
    if base_cls is None:
        return

    orig_set = base_cls.__setattr__

    def permissive_set(self, name, value):
        try:
            orig_set(self, name, value)
        except (AttributeError, TypeError, ValueError):
            object.__setattr__(self, name, value)

    base_cls.__setattr__ = permissive_set
