""" Day 7 """
from functools import lru_cache


def fire_beam(grid: list[str]) -> int:
    """ Counts the amount of splits when firing a beam """
    s_index = grid[0].find("S")
    test = _fire_beam(grid, s_index, 1, 0)
    return test


def _fire_beam(grid: list[str], beam_x: int, beam_y: int, split_count: int):
    if beam_y >= len(grid) or len(grid[0]) <= beam_x < 0:
        return split_count

    current_char = grid[beam_y][beam_x]

    # if beam is already there, return
    if current_char == '|':
        return split_count

    # replace index position with beam
    grid[beam_y] = grid[beam_y][:beam_x] + '|' + grid[beam_y][beam_x+1:]

    # if there is a beam splitter, split the beam
    if current_char == '^':
        left_splits = _fire_beam(grid, beam_x-1, beam_y+1, 0)
        right_splits = _fire_beam(grid, beam_x+1, beam_y+1, 0)
        # split_count + this split + all splits henceforth
        return split_count + 1 + left_splits + right_splits

    # carry on one line down
    return _fire_beam(grid, beam_x, beam_y+1, split_count)


def fire_quantum_beam(grid: list[str]) -> int:
    """ Counts timelines generated """
    s_index = grid[0].find("S")

    # Convert grid to tuple of tuples (hashable)
    grid_tuple = tuple(grid)

    @lru_cache(maxsize=None)
    def _fire_quantum_beam_cached(beam_x: int, beam_y: int) -> int:
        if beam_y >= len(grid_tuple):
            return 1
        if beam_x < 0 or beam_x >= len(grid_tuple[0]):
            return 1

        current_char = grid_tuple[beam_y][beam_x]

        if current_char == '^':
            left = _fire_quantum_beam_cached(beam_x-1, beam_y+1)
            right = _fire_quantum_beam_cached(beam_x+1, beam_y+1)
            return left + right

        return _fire_quantum_beam_cached(beam_x, beam_y+1)

    return _fire_quantum_beam_cached(s_index, 1)


def day7():
    """ Main function """
    grid: list[str]

    with open("inputs/day7/input.txt", encoding="UTF-8") as f:
        grid = f.readlines()

    total_splits = fire_beam(grid)

    print(total_splits)


def day7_part2():
    """ Main function """
    grid: list[str]

    with open("inputs/day7/input.txt", encoding="UTF-8") as f:
        grid = f.readlines()

    timelines = fire_quantum_beam(grid)

    print(timelines)


if __name__ == "__main__":
    # day7()
    day7_part2()
