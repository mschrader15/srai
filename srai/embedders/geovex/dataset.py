"""
HexagonalDataset.

This dataset is used to train a hexagonal encoder model.
As defined in GeoVex paper[1].

References:
    [1] https://openreview.net/forum?id=7bvWopYY1H
"""

from typing import TYPE_CHECKING, Any, Dict, Generic, List, NamedTuple, Set, TypeVar

import numpy as np
import pandas as pd
from tqdm import tqdm

from srai.neighbourhoods import Neighbourhood
from srai.utils._optional import import_optional_dependencies

if TYPE_CHECKING:  # pragma: no cover
    import torch

try:  # pragma: no cover
    from torch.utils.data import Dataset

except ImportError:
    from srai.utils._pytorch_stubs import Dataset


T = TypeVar("T")


class HexagonalDatasetItem(NamedTuple):
    """
    Hexagonal dataset item.

    Attributes:
        hex_matrix (torch.Tensor): Anchor regions.
    """

    hex_matrix: "torch.Tensor"



class HexagonalDataset(Dataset[HexagonalDatasetItem], Generic[T]):
    """
    Dataset for the hexagonal encoder model.

    It works by returning a 3d tensor of hexagonal regions.
    The tensor is a cube with the target hexagonal region in the center,
    and the rings of neighbors around surrounding it.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        neighbor_k_ring=6,
    ):
        """
        Initialize the HexagonalDataset.

        Args:
            data (pd.DataFrame): Data to use for training. Raw counts of features in regions.
            neighbor_k_ring (int, optional): _description_. Defaults to 6.
        """

        import_optional_dependencies(dependency_group="torch", modules=["torch"])
        
        
        self._valid_h3 = []
        all_indices = set(data.index)

        # number of columns in the dataset
        self._N = data.shape[1]
        self._k = neighbor_k_ring

        for i, (h3_index, hex_data) in tqdm(
            enumerate(data.iterrows()), total=len(data)
        ):
            hex_neighbors_h3 = h3.grid_disk(h3_index, neighbor_k_ring)
            # remove the h3_index from the neighbors
            hex_neighbors_h3.remove(h3_index)
            available_neighbors_h3 = list(hex_neighbors_h3.intersection(all_indices))
            if len(available_neighbors_h3) < len(hex_neighbors_h3):
                # skip adding this h3 as a valid input
                continue
            anchor = np.array(h3.cell_to_local_ij(h3_index, h3_index))
            self._valid_h3.append(
                (
                    h3_index,
                    data.index.get_loc(h3_index),
                    [
                        # get the index of the h3 in the dataset
                        (
                            data.index.get_loc(_h),
                            tuple(
                                (
                                    np.array(h3.cell_to_local_ij(h3_index, _h)) - anchor
                                ).tolist()
                            ),
                        )
                        for _h in hex_neighbors_h3
                    ],
                )
            )

        self._data = data.to_numpy(dtype=np.float32)
        self._data_torch = torch.Tensor(self._data)

    def __len__(self):
        return len(self._valid_h3)

    def __getitem__(self, index):
        # construct the 3d tensor
        h3_index, target_idx, neighbors_idxs = self._valid_h3[index]
        return self._build_tensor(target_idx, neighbors_idxs)

    def _build_tensor(self, target_idx, neighbors_idxs):
        # build the 3d tensor
        # it is a tensor with diagonals of length neighbor_k_ring
        # the diagonals are the neighbors of the target h3
        # the target h3 is in the center of the tensor
        # the tensor is 2*neighbor_k_ring + 1 x 2*neighbor_k_ring + 1 x 2*neighbor_k_ring + 1
        # make a tensor of zeros, padded by 1 zero all around to make it even for the convolutions
        tensor = torch.zeros(
            (
                self._N,
                2 * self._k + 2,
                2 * self._k + 2,
            )
        )

        # set the target h3 to the center of the tensor
        tensor[
            :,
            self._k,
            self._k,
        ] = self._data_torch[target_idx]

        # set the neighbors of the target h3 to the diagonals of the tensor
        for neighbor_idx, (i, j) in neighbors_idxs:
            tensor[:, self._k + i, self._k + j] = self._data_torch[neighbor_idx]


        # return the tensor and the target (which is same as the tensor)
        # should we return the target as a copy of the tensor?
        return tensor


    def full_dataset(self):
        h3s = []
        tensors = []
        for h3_, target_idx, neighbors_idxs in tqdm(self._valid_h3, total=len(self._valid_h3)):
            h3s.append(h3_)
            tensors.append(self._build_tensor(target_idx, neighbors_idxs))
        
        return h3s, torch.stack(tensors)

    @property
    def shape(
        self,
    ) -> int:
        return self._N, (2 * self._k + 1), (2 * self._k + 1)


