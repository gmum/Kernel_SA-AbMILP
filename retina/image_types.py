from abc import ABC
from abc import abstractmethod

import numpy as np


class Image(ABC):
    """Interface for casting images to given data type."""

    MAX_UINT8 = 255.0
    MAX_UINT16 = 65535.0
    UINT16_TO_UINT8 = MAX_UINT16 / MAX_UINT8
    MAX_FLOAT = 1.0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.cast_to_type(image)

    @abstractmethod
    def cast_to_type(self, image: np.ndarray) -> np.ndarray:
        """Converts the image to particular type."""
        pass


class FloatImage(Image):
    """Callable casting image to np.float type."""

    def cast_to_type(self, image: np.ndarray) -> np.ndarray:
        """Converts the image to [0.0, 1.0] np.float type.

        Args:
            image (np.ndarray): numpy array storing the image

        Returns:
            converted_image (np.ndarray): image converted to float type
        """
        converted_image = np.copy(image).astype(np.float)

        if np.issubdtype(image.dtype, np.uint8):
            converted_image = converted_image / self.MAX_UINT8

        elif np.issubdtype(image.dtype, np.uint16):
            converted_image = converted_image / self.MAX_UINT16

        # scale values if range exceeds 1.0
        if np.max(converted_image) > self.MAX_FLOAT:
            converted_image = converted_image / np.max(converted_image)

        return converted_image


class Uint8Image(Image):
    """Callable casting image to np.uint8 type."""

    def cast_to_type(self, image: np.ndarray) -> np.ndarray:
        """Converts the image to [0, 255] np.uint8 type.

        Args:
            image (np.ndarray): numpy array storing the image

        Returns:
            converted_image (np.ndarray): image converted to int type
        """
        converted_image = np.copy(image)

        if np.issubdtype(converted_image.dtype, np.float):
            if np.max(converted_image) > self.MAX_FLOAT:
                converted_image = converted_image / np.max(converted_image)
            converted_image = converted_image * self.MAX_UINT8

        elif np.issubdtype(converted_image.dtype, np.uint16):
            converted_image = converted_image / self.UINT16_TO_UINT8

        converted_image = np.round(converted_image)
        converted_image = converted_image.clip(0, self.MAX_UINT8).astype(np.uint8)
        return converted_image


class Uint16Image(Image):
    """Callable casting image to np.uint16 type."""

    def cast_to_type(self, image: np.ndarray):
        """Converts the image to [0, 65535] np.uint16 type.

        Args:
            image (np.ndarray): numpy array storing the image

        Returns:
            converted_image (np.ndarray): image converted to int type
        """
        converted_image = np.copy(image)

        if np.issubdtype(converted_image.dtype, np.float):
            if np.max(converted_image) > self.MAX_FLOAT:
                converted_image = converted_image / np.max(converted_image)
            converted_image = converted_image * self.MAX_UINT16

        elif np.issubdtype(converted_image.dtype, np.uint8):
            converted_image = converted_image * self.UINT16_TO_UINT8

        converted_image = np.round(converted_image)
        converted_image = converted_image.clip(0, self.MAX_UINT16).astype(np.uint16)
        return converted_image
