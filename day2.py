""" Day 2 """


def is_repeating_sequence(original_string: str, sequence: str):
    """ Checks if id string just consists of repeating sequence """
    # if more than 1 instance of split string is found,
    if original_string.count(sequence) >= 2:
        # and string only consists of that sequence,
        if len(original_string.replace(sequence, "")) == 0:
            return True
    return False


def invalid_id_sum(first_id: int, last_id: int):
    """ Checks validity of id range. Returns sum of invalid ids of range first_id-last_id """
    id_total = 0

    for id_to_check in range(first_id, last_id+1):
        str_id = str(id_to_check)

        midpoint = len(str_id) // 2
        # keep splitting down string and checking for duplicates
        for i in range(1, midpoint+1):
            # if string is divisible by i
            if len(str_id) % i == 0:
                # if string contains repeating sequence
                if is_repeating_sequence(str_id, str_id[:i]):
                    id_total += id_to_check
                    break

    return id_total


def day2():
    """ main function """
    file: str = ""
    with open("inputs/day2/input.txt", encoding="UTF-8") as f:
        file = f.readline()

    product_ids = file.split(',')

    invalid_id_total = 0

    for pair in product_ids:
        id1, id2 = pair.split('-')
        invalid_id_total += invalid_id_sum(int(id1), int(id2))

    print(invalid_id_total)


if __name__ == "__main__":
    day2()
