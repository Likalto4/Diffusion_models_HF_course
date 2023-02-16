from PIL import Image

def make_grid(images:list, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing

    Args:
        images (list): list of PIL images
        size (int, optional): size of the images (square). Defaults to 64.

    Returns:
        _type_: _description_
    """
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im