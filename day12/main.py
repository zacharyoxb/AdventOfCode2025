""" Main day 12 file """
import torch
from torchrl.modules import Actor
from tensordict.nn import TensorDictModule

from data_utils import reader
from nn.present_env import PresentPlacementEnv
from nn.present_policy import PresentPlacementPolicy


if __name__ == "__main__":
    buffer = reader.get_data("testinput.txt")
    # env setup
    env = PresentPlacementEnv()
    # policy setup
    policy_net = PresentPlacementPolicy()
    policy_module = TensorDictModule(
        policy_net,
        in_keys=["grid", "presents", "present_count"],
        out_keys=["present_idx", "x", "y", "rot", "flip"]
    )

    # get deterministic actor
    actor = Actor(
        module=policy_module,
        in_keys=["present_idx", "x", "y", "rot",
                 "flip"],
        out_keys=["action"],
    )
