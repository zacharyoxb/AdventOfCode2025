""" Day 9 """

from itertools import combinations
from shapely.geometry import Polygon
from shapely import box, make_valid


def get_rectangle_area(vec1: tuple[int, int], vec2: tuple[int, int]) -> int:
    """ Gets the area of the rectangle made given corner coords vec1 and vec2"""
    height = max(vec1[0], vec2[0]) - min(vec1[0], vec2[0])+1
    width = max(vec1[1], vec2[1]) - min(vec1[1], vec2[1])+1
    return height * width


def rect_is_valid(valid_area: Polygon, vec0: tuple[int, int], vec1: tuple[int, int]):
    """ Checks if rectangle is valid """
    (x0, y0), (x1, y1) = vec0, vec1
    x_min, x_max, y_min, y_max = min(x0, x1), max(
        x0, x1), min(y0, y1), max(y0, y1)
    rect_polygon = box(x_min, y_min, x_max, y_max)

    # Check if the rectangle is completely within the valid area
    return valid_area.contains(rect_polygon)


def day9_part2(red_tiles: list[tuple[int, int]]):
    """ Day 9 part 2 """
    # get all the possible red tile combos
    rect_corner_pairs = list(combinations(red_tiles, 2))

    # order rectangles by how big they are, biggest to smallest
    rect_corner_pairs.sort(key=lambda pair: get_rectangle_area(
        *pair), reverse=True)

    # use shapely to construct polygon representing valid coords
    coord_lines = Polygon(red_tiles)
    valid_area = make_valid(coord_lines, method='structure')

    rect_area = 0
    # return first valid rectangle
    for rect_corners in rect_corner_pairs:
        if rect_is_valid(valid_area, *rect_corners):
            rect_area = get_rectangle_area(*rect_corners)
            break

    print(rect_area)


def day9_part1(vectors):
    """ Day 9 part 1 """
    vec_len = len(vectors)

    max_area = 0

    for i in range(vec_len):
        for j in range(i+1, vec_len):
            vec1, vec2 = vectors[i], vectors[j]
            area = get_rectangle_area(vec1, vec2)
            max_area = max(area, max_area)

    print(max_area)


def day9():
    """ Main function """
    vectors: list[tuple[int, int]] = []

    with open("inputs/day9/input.txt", encoding="UTF-8") as f:
        line = f.readline()
        while line != "":
            split_line = line.strip().split(',')
            int_line = tuple(map(int, split_line))
            vectors.append(int_line)
            line = f.readline()

    # part 1
    day9_part1(vectors)

    # part 2
    day9_part2(vectors)


if __name__ == "__main__":
    day9()
