import numpy as np
import torch

from skimage.color import rgb2hsv
from torch import Tensor

from image_types import FloatImage
from image_types import Uint8Image
from image_types import Uint16Image


def convert_image_to_float(image: np.ndarray) -> np.ndarray:
    """Converts image to [0.0, 1.0] np.float type.

    Args:
        image (np.ndarray): RGB or grayscale image.

    Returns:
        np.ndarray: Converted image.
    """

    converted_image = FloatImage()(image)

    return converted_image


def convert_image_to_int(image: np.ndarray, image_type: np.dtype = np.uint8) -> np.ndarray:
    """Casts data to integer format specified in image_type.

    Args:
        image (np.ndarray): RGB or grayscale image.
        image_type (np.dtype): Either uint8 or uint16, default: np.uint8.

    Returns:
        np.ndarray: Converted image.
    """

    if image_type == np.uint8:
        converted_image = Uint8Image()(image)
    elif image_type == np.uint16:
        converted_image = Uint16Image()(image)

    return converted_image


def convert_image_to_tensor(image: np.ndarray) -> Tensor:
    """Converts image from numpy array to Tensor.

    Args:
        image (np.ndarray): Image numpy array.

    Returns:
        Tensor: Image Tensor.
    """

    image_channel_first = np.moveaxis(image, -1, 0)
    image_tensor = torch.from_numpy(image_channel_first)

    return image_tensor


def convert_tensor_to_image(image_tensor: Tensor) -> np.ndarray:
    """Converts image from Tensor to numpy array.

    Args:
        image_tensor (Tensor: Image Tensor.

    Returns:
        np.ndarray: Image numpy array.
    """

    image = image_tensor.numpy()
    image_channel_last = np.moveaxis(image, 0, -1)

    return image_channel_last
