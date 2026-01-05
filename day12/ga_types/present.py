""" Type representing each present """
from dataclasses import dataclass
from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

PresentMatrix: TypeAlias = NDArray[np.int8]
PresentMatrices: TypeAlias = list[NDArray[np.int8]]

PresentOrientation: TypeAlias = list[int]


@dataclass
class Present:
    """ Represents every present rotation as a bitmask """
    masks: list[PresentOrientation]

    @classmethod
    def from_matrix(cls, matrix: PresentMatrix) -> 'Present':
        """ Factory method to create Present from matrix """
        masks = []
        for i in range(0, 8):
            masks.append(cls._get_mask_orientation(matrix, i))

        return cls(masks=masks)

    @staticmethod
    def _matrix_to_bitmask(matrix: PresentMatrix) -> PresentOrientation:
        """ Convert 3x3 matrix to three 3-digit binary numbers (one per row) """
        binary_rows = []

        for row in matrix:
            # Convert each row to a 3-digit binary number
            binary_str = ''
            for val in row:
                binary_str += '1' if val else '0'

            binary_rows.append(int(binary_str, 2))

        return binary_rows

    @staticmethod
    def _bitmask_to_matrix(bitmask: list[int]) -> PresentMatrix:
        """ Convert bitmask back to 3x3 matrix """
        matrix = []
        for row_mask in bitmask:
            row = []
            # Extract bits from binary number
            for i in range(2, -1, -1):  # Most significant bit first
                bit = (row_mask >> i) & 1
                row.append(bit)
            matrix.append(row)
        return np.array(matrix, dtype=np.int8)

    @classmethod
    def _get_mask_orientation(cls, mask: PresentMatrix, mutation: int):
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
        return cls._matrix_to_bitmask(mutated)
