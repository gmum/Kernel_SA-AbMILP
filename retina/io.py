from typing import Union
from typing import Tuple

import numpy as np
import torch

from skimage.io import imread as skimread
from skimage.io import imsave as skimsave
from torch.optim.optimizer import Optimizer
from torchvision.transforms import ToTensor

from conversion import convert_image_to_int
from conversion import convert_image_to_float


def imread(
    filepath: str,
    return_type: Union[np.dtype, str] = np.uint8,
    convert_to_tensor: bool = False,
) -> np.ndarray:
    """Read image data from file into the specified format.

    This function wraps scikit-image imread functionality and applies required
    conversions.

    Args:
        filepath: string containing path to the image data; .npy files are also
            supported.
        return_type: data format of the returned image, can be np.dtype or
            a string. Supported formats are: float, uint8, uint16.
            Default is uint8.
        convert_to_tensor: flag to convert output image to torch.Tensor.
            Defauls is False.

    Returns:
        Image in selected format."""

    str_to_type = {'float': np.float,
                   'uint8': np.uint8,
                   'uint16': np.uint16,
                   }

    if hasattr(return_type, 'lower'):
        return_type = return_type.lower()

    if isinstance(return_type, str):
        return_type = str_to_type[return_type]

    # load file
    if filepath.endswith('.npy'):
        image = np.load(filepath)
    else:
        image = skimread(filepath)

    # apply conversions
    if not np.issubdtype(image.dtype, return_type):
        if return_type is np.float:
            image = convert_image_to_float(image)
        else:
            image = convert_image_to_int(image, image_type=return_type)

    if convert_to_tensor:
        image = ToTensor()(image)

    return image


def imsave(
    filepath: str,
    image: Union[np.ndarray, torch.Tensor],
):
    """Save image into a file.

    Args:
        filepath: string containing path to output file.
        image: NumPy array or a torch.Tensor to save."""

    if not isinstance(image, torch.Tensor):
        skimsave(filepath, image)
    else:
        if image.ndim == 3:
            image_to_save = image.permute(1, 2, 0).cpu().numpy().squeeze()
            skimsave(filepath, image_to_save)
        else:
            skimsave(filepath, image.cpu().numpy().squeeze())
