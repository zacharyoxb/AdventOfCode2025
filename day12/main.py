""" Main day 12 file """
from data_utils import reader
from nn.present_env import PresentPlacementEnv
from nn.present_policy import PresentPlacementPolicy


if __name__ == "__main__":
    present_tensor = reader.get_presents("testinput.txt")
    area_info = reader.get_placement_info("testinput.txt")

    for info in area_info:
        params = PresentPlacementEnv.gen_params(
            (info.width, info.height), present_tensor, info.present_count)

        env = PresentPlacementEnv(params)
        policy = PresentPlacementPolicy((info.height, info.width))
