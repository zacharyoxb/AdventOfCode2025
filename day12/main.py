""" Main day 12 file """
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from data_utils import reader


PresentMatrix: TypeAlias = NDArray[np.float16]

if __name__ == "__main__":
    present_matrices = reader.get_presents("testinput.txt")
    area_info = reader.get_placement_info("testinput.txt")
