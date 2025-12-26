""" Day 1 """


def get_int(turn: str):
    """ Converts strs beginning with L to negatives, strs beginning
    with R to positives.
    """
    replace_l = turn.replace("L", "-")
    replace_l_r = replace_l.removeprefix("R")
    final = replace_l_r.strip()
    return int(final)


def day1():
    """ main function """
    file = []
    with open("inputs/day1/input.txt", encoding="UTF-8") as f:
        file = f.readlines()

    # convert list members into ints
    mapped_turns = list(map(get_int, file))

    valve_pos = 50
    zero_hits = 0

    for full_turn in mapped_turns:
        # if positive / right turn, count up until turn is complete
        if full_turn > 0:
            zero_hits += sum(map(lambda x: x % 100 == 0, range(
                valve_pos, valve_pos+full_turn, 1)))
        # if negative / left turn, count down until turn is complete
        if full_turn < 0:
            zero_hits += sum(map(lambda x: x % 100 == 0, range(
                valve_pos, valve_pos+full_turn, -1)))

        # update valve position
        valve_pos = (valve_pos+full_turn) % 100
    print(zero_hits)


if __name__ == "__main__":
    day1()
