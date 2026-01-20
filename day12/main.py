""" Main day 12 file """
from data_utils import reader
from nn.present_env import PresentPlacementEnv
from nn.present_policy import PresentPlacementPolicy


if __name__ == "__main__":
    present_tensor = reader.get_presents("testinput.txt")
    area_info = reader.get_placement_info("testinput.txt")

    params = []

    for info in area_info:
        params.append(PresentPlacementEnv.gen_params(
            (info.width, info.height), present_tensor, info.present_count))

    env = PresentPlacementEnv(params[0])
    policy = PresentPlacementPolicy()

    for param in params:
        env.rollout(policy=policy)
