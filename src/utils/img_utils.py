import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_img_shape(img_path, channel="last"):
    """
    References
    ----------
    https://stackoverflow.com/questions/52962969/number-of-channels-in-pil-pillow-image
    """
    img = Image.open(img_path)
    if channel == "last":
        return (*img.size, len(img.getbands()))
    elif channel == "first":
        return (len(img.getbands()), *img.size)
