""" Day 3 """


def find_largest(num_string: str) -> str:
    """ Gets the largest 12 digit number that can be made from string. """
    target_len = 12
    result = ""

    start_idx = 0

    while len(result) < target_len:
        # Calculate cutoff point
        cutoff_idx = len(num_string) - (target_len - len(result) - 1)
        # select number to insert
        selection = max(num_string[start_idx:cutoff_idx])
        # get number index, update starting index
        start_idx += num_string[start_idx:cutoff_idx].find(selection)+1
        # append to result
        result += selection

    return result


def day3():
    """ Main function """
    file = []
    joltage_count = 0
    with open("inputs/day3/input.txt", encoding="UTF-8") as f:
        file = f.readlines()

    for line in file:
        largest_str = find_largest(line.strip())
        joltage_count += int(largest_str)
    print(joltage_count)


if __name__ == "__main__":
    day3()
