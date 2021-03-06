import os

from typing import Callable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import Tuple

import itertools

import numpy as np
import pandas as pd

from io import imsave


class Patch(NamedTuple):
    """
    Patch object:
    image_array: np array with the image,
    size: width and height of the patch,
    x: x position,
    y: y position.
    """
    image_array: np.ndarray
    size: int
    x: int
    y: int

    def __repr__(self) -> str:
        return "Patch<{}x{} @ {}x{}>".format(self.size, self.size, self.x, self.y)


class PatchGenerator:
    """Patches Generator"""

    def __init__(self, image: np.array, size: int, stride: int):
        """Class for generating, filtering, and saving patches.
        Args:
            image: image to be cut into patches,
            size: size of one patch, height and width are equal,
            stride: stride of the patch"""
        self.image = image
        self.shape = self.image.shape[:2]
        self.size = size
        self.stride = stride
        self.metadata = pd.DataFrame()
        self.filters = []
        self.patches, self.metadata = self.generate_patches()

    def generate_patches(self) -> Tuple[List[Patch], pd.DataFrame]:
        """
        Generates patches from the image. If size of the image is not equal to multiplication of size,
        the last patch is cut starting from the edge od the image (both in horizontal and vertical case.)
        Returns:
            List[Patch]: list of Patch objects,
            pd.DataFrame: dataframe with metadata.
        """
        top_left_corners = self._get_top_left_corners()
        patches = [
            Patch(
                image_array=self.cut_patch(x, y),
                size=self.size,
                x=x,
                y=y,
            )
            for x, y, in top_left_corners
        ]
        metadata = []
        for patch in patches:
            metadata.append([patch.x, patch.y])
        columns = ['position_x', 'position_y']
        return patches, pd.DataFrame(data=metadata, columns=columns)

    def _get_top_left_corners(self) -> Iterator[Tuple[int, int]]:
        """
        Generates list of top left corners of patches.
        Returns:
            Iterator[Tuple[int, int]] - generates pairs of x and y positions.
        """
        last_patch_x = self.shape[0] - self.size
        last_patch_y = self.shape[1] - self.size
        rows, cols = set(range(0, last_patch_x, self.stride)), set(range(0, last_patch_y, self.stride))
        if self.shape[0] % self.size != 0:
            rows.add(last_patch_x)
        if self.shape[1] % self.size != 0:
            cols.add(last_patch_y)
        yield from ((row, col)
                    for row in rows
                    for col in cols)

    def cut_patch(self, x: int, y: int) -> np.array:
        """
        Cuts patch from an image at position x, y.
        Args:
            x: x position
            y: y position
        Returns:
            (np.array): array with cut patch
        """
        return self.image[x: x + self.size, y: y + self.size]

    def apply_filters(self, filters: List[Callable]) -> None:
        """
        Applies filters in the list. Filter has to be a function return Tuple[bool, float].
        For examples see: caider.preprocessing.quality_check. Results are save to pd.DataFrame
        and stored in self.metadata variable.
        Args:
             filters: List of filter functions.
        """
        self.filters = filters
        column_names = [[f'{f.__name__}', f'{f.__name__}_value'] for f in self.filters]
        column_names = list(itertools.chain(*column_names))
        self.metadata = self.metadata.reindex(columns=[*self.metadata.columns, *column_names])
        for i, patch in enumerate(self.patches):
            curr_patch_row = []
            for f in filters:
                curr_patch_row.extend(f(patch.image_array))
            self.metadata.loc[i, column_names] = curr_patch_row

    def save_patches(self,
                     dir_path: str,
                     file_format: str = 'patch_{}_{}.png',
                     csv_filename: str = 'patches.csv',
                     only_filtered: bool = True):
        """
        Saves patch images to files, and report with paths and filter value to csv file.
        Args:
            dir_path: directory for saving files,
            file_format: format of the patch's filename, specifies file format,
            has to include placeholder for x, y positions, default: 'patch_{}_{}.png'
            csv_filename: name of the report file, default: 'patches.csv',
            only_filtered: if True, only filtered patches will be saved, otherwise: all patches are saved.

        """
        indices = self.metadata.index.tolist()

        if only_filtered and len(self.metadata) > 0 and len(self.filters) > 0:
            filter_columns = [f.__name__ for f in self.filters]
            indices = self.metadata[eval(" & ".join([f"(self.metadata['{col}'])"
                                                     for col in filter_columns]))].index.tolist()

        self.metadata.insert(len(self.metadata.columns), 'path', '')

        for i, patch in enumerate(self.patches):
            if i not in indices:
                continue
            save_path = os.path.join(dir_path, file_format.format(patch.x, patch.y))
            imsave(save_path, patch.image_array)
            self.metadata.loc[i, 'path'] = save_path
        self.metadata.to_csv(os.path.join(dir_path, csv_filename))
