def apply(real_gr):
    import gradio.utils as u
    if not hasattr(u, 'version_check'):
        u.version_check = lambda: None
    if not hasattr(u, 'get_local_ip_address'):
        u.get_local_ip_address = lambda: '127.0.0.1'
