""" Main day 12 file """
from data_utils import reader


if __name__ == "__main__":
    present_tensor = reader.get_presents("testinput.txt")
    area_info = reader.get_placement_info("testinput.txt")
