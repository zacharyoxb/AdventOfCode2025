""" Day 6 """
import re


def run_calc(val1: int, val2: int, operator: str):
    """ Runs a calculation based on the operator string """
    if operator == "+":
        return val1 + val2
    if operator == "*":
        return val1 * val2
    return 0


def day6():
    """ Main function """
    all_lines: list[str] = []
    with open("inputs/day6/input.txt", encoding="UTF-8") as f:
        current_line = f.readline()
        while current_line != "":
            all_lines.append(current_line)
            current_line = f.readline()

    # reverse order of list
    all_lines.reverse()

    # get all operators removing spaces
    operators = re.split(r"\s+", all_lines[0].strip())
    # get all numbers removing spaces
    original_nums_str: list[list[str]] = list(
        map(lambda line: re.split(r"\s+", line.strip()), all_lines[1:]))

    # convert to int
    original_nums: list[list[int]] = [
        [int(num) for num in inner_list]
        for inner_list in original_nums_str
    ]

    calculated_nums: list[int] = []

    for num_list in original_nums:
        if len(calculated_nums) == 0:
            calculated_nums = num_list
        else:
            calculated_nums = list(map(
                run_calc, calculated_nums, num_list, operators))

    ans = sum(calculated_nums)
    print(ans)


def run_calc_2(val1: int, val2: int, operator: str) -> int:
    """ Runs calculation for group of digits """

    if operator == "+":
        return val1 + val2

    if operator == "*":
        return val1 * val2

    return 0


def day6_part2():
    """ Second part """
    all_lines: list[str] = []
    with open("inputs/day6/input.txt", encoding="UTF-8") as f:
        current_line = f.readline()
        while current_line != "":
            all_lines.append(current_line)
            current_line = f.readline()

    # get all operators removing spaces
    operators = re.split(r"\s+", all_lines[len(all_lines)-1].strip())
    # get all numbers removing spaces before digits
    original_nums_str: list[list[str]] = all_lines[:len(all_lines)-1]

    # tracks total
    total = 0
    # tracks if the current op total has been initialised
    current_op_init = False
    # tracks the total of the current operation
    current_op_total = 0
    # tracks the index of the current operator
    oper_idx = 0

    # zip each line together to iterate on the at the same time
    for col_chars in zip(*original_nums_str):
        # join digits together and flip (we are reading the file backwards)
        str_values = ''.join(col_chars).strip()
        # If str_values is empty, all operations for group complete, index+1
        if len(str_values) == 0:
            total += current_op_total
            oper_idx += 1
            current_op_init = False
        elif not current_op_init:
            current_op_total = int(str_values)
            current_op_init = True
        else:
            col_digit = int(str_values)
            current_op_total = run_calc_2(current_op_total,
                                          col_digit, operators[oper_idx])

    print(total)


if __name__ == "__main__":
    # day6()
    day6_part2()
