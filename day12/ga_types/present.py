""" Type representing each present """
from dataclasses import dataclass
from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

PresentMatrix: TypeAlias = NDArray[np.int32]


@dataclass
class Present:
    """ Represents every present rotation as an array """
    masks: list[PresentMatrix]

    @classmethod
    def from_matrix(cls, matrix: PresentMatrix) -> 'Present':
        """ Factory method to create Present from matrix """
        masks = []
        for i in range(0, 8):
            masks.append(cls._get_mask_orientation(matrix, i))

        return cls(masks=masks)

    @classmethod
    def _get_mask_orientation(cls, mask: PresentMatrix, mutation: int) -> PresentMatrix:
        """ Rotate grid by specified number of 90-degree rotations (0-7)
        0: no rotation
        1: 90° right / clockwise
        2: 180°
        3: 90° left / counter-clockwise
        4: vertical flip, no rotation
        5: vertical flip, 90° right / clockwise
        6: vertical flip, 180°
        7: vertical flip, 90° left / counter-clockwise

        Returns a new Present with rotated mask
        """
        mutated = mask

        # flip if required
        if mutation > 3:
            mutated = np.fliplr(mutated)

        # rotate
        rot = mutation % 4
        mutated = np.rot90(mutated, k=-rot)

        # return bitmask
        return mutated
