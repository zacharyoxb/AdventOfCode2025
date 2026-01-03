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
    mask: PresentOrientation

    @classmethod
    def from_matrix(cls, matrix: PresentMatrix) -> 'Present':
        """ Factory method to create Present from matrix """
        mask = cls._matrix_to_bitmask(matrix)

        return cls(mask=mask)

    @staticmethod
    def _matrix_to_bitmask(matrix):
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

    def mutate_grid(self, mutation: int):
        """ Rotate grid by specified number of 90-degree rotations (0-3)
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
        # convert to matrix
        mutated = self._bitmask_to_matrix(self.mask)

        # flip if required
        if mutation > 3:
            mutated = np.fliplr(mutated)

        # rotate
        rot = mutation % 4
        mutated = np.rot90(mutated, k=-rot)

        self.mask = self._matrix_to_bitmask(mutated)
