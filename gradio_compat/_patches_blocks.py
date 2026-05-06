import functools


def apply(real_gr):
    _ver = int(real_gr.__version__.split('.')[0])

    # queue(64) positional arg → queue(max_size=64) in 4.x+
    orig_queue = real_gr.Blocks.queue

    @functools.wraps(orig_queue)
    def patched_queue(self, concurrency_count_or_max_size=None, **kw):
        if _ver >= 4 and concurrency_count_or_max_size is not None:
            kw.setdefault('max_size', concurrency_count_or_max_size)
        elif concurrency_count_or_max_size is not None:
            kw['concurrency_count'] = concurrency_count_or_max_size
        if _ver >= 5:
            cc = kw.pop('concurrency_count', None)
            if cc is not None:
                kw.setdefault('default_concurrency_limit', cc)
        return orig_queue(self, **kw)

    real_gr.Blocks.queue = patched_queue

    # launch() — silently drop kwargs removed across 4.x/5.x/6.x
    orig_launch = real_gr.Blocks.launch

    @functools.wraps(orig_launch)
    def patched_launch(self, **kw):
        if _ver >= 4:
            for dead in ('show_api', 'encrypt', 'file_directories'):
                kw.pop(dead, None)
        return orig_launch(self, **kw)

    real_gr.Blocks.launch = patched_launch

    # get_config_file was renamed to get_config in 4.x
    if not hasattr(real_gr.Blocks, 'get_config_file') and hasattr(real_gr.Blocks, 'get_config'):
        real_gr.Blocks.get_config_file = real_gr.Blocks.get_config
