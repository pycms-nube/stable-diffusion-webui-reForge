def apply(real_gr):
    try:
        import gradio.processing_utils as pu
    except ImportError:
        return

    if not hasattr(pu, 'decode_base64_to_image'):
        import base64
        import io
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
