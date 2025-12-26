""" Day 4 """

from dataclasses import dataclass


@dataclass
class Coordinate:
    """ Coordinate datatype """
    x: int
    y: int


# Mappings to each coordinate around a coordinate
DIRECTIONS = [Coordinate(-1, -1), Coordinate(0, -1), Coordinate(1, -1),
              Coordinate(-1, 0),                     Coordinate(1, 0),
              Coordinate(-1, 1), Coordinate(0, 1), Coordinate(1, 1)]


def is_accessible(layout: list[str], x: int, y: int) -> bool:
    """ Returns a boolean representing if the roll at coord x, y is accessible """
    adjacent_rolls = 0
    for direction in DIRECTIONS:
        nx, ny = x + direction.x, y + direction.y
        if (0 <= nx < len(layout[0])) and (0 <= ny < len(layout)) and layout[nx][ny] == '@':
            adjacent_rolls += 1

        if adjacent_rolls >= 4:
            return False
    return adjacent_rolls < 4


def day4():
    """ Main function """
    roll_layout: list[str] = []

    with open("inputs/day4/input.txt", encoding="UTF-8") as f:
        line = f.readline().strip()

        while line != "":
            roll_layout.append(line)
            line = f.readline().strip()

    # track total access count and count on last iteration
    access_count = 0
    # starts on 0 so while loop is entered
    last_count = 1

    while access_count != last_count:
        last_count = access_count

        for x in range(0, len(roll_layout[0])):
            for y in range(0, len(roll_layout)):
                if roll_layout[x][y] == '@' and is_accessible(roll_layout, x, y):
                    roll_layout[x] = roll_layout[x][:y] + \
                        'x' + roll_layout[x][y+1:]
                    access_count += 1

    print(access_count)


if __name__ == "__main__":
    day4()
