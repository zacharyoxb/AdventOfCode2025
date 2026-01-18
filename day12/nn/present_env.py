""" Neural network for packing presents with orientation masks """

import torch
from tensordict import TensorDict

from torchrl.data import Bounded, Composite, Unbounded, Categorical


class PresentPlacementEnv():
    """ RL environment for present placement """

    def __init__(
            self,
            grid_size: tuple[int, int],
            presents: torch.Tensor,
            present_count: list[int]
    ):
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.grid_size = ()
        self.presents = presents
        self.present_count = None

        self.observation_spec = None
        self.action_spec = None
        self.reward_spec = None
        self.done_spec = None

        self._init_specs(grid_size, present_count)

    def _init_specs(
            self,
            grid_size: tuple[int, int],
            present_count: list[int]
    ):
        self.grid_size = grid_size
        self.present_count = torch.tensor(
            present_count, dtype=torch.float16, device=self.device)
        width, height = self.grid_size

        # Observation spec: what the agent sees
        self.observation_spec = Composite({
            "grid": Bounded(low=0, high=1, shape=torch.Size(self.grid_size),
                            dtype=torch.uint8, device=self.device),
            "presents": Bounded(low=0, high=1, shape=torch.Size([3, 3]),
                                dtype=torch.uint8, device=self.device),
            "present_count": Unbounded(shape=5, dtype=torch.uint64, device=self.device)
        })

        # Action spec: what the agent can do
        self.action_spec = Composite({
            "present_idx": Bounded(low=0, high=5, shape=1, dtype=torch.uint8, device=self.device),
            "x": Bounded(low=0, high=width-3, shape=1, dtype=torch.uint64, device=self.device),
            "y": Bounded(low=0, high=height-3, shape=1, dtype=torch.uint64, device=self.device),
            "rot": Bounded(low=0, high=3, shape=1, dtype=torch.uint8, device=self.device),
            "flip": Bounded(low=0, high=1, shape=torch.Size([2]), dtype=torch.uint8,
                            device=self.device)
        })

        # Reward and done specs
        self.reward_spec = Unbounded(shape=torch.Size(
            [1]), dtype=torch.int64, device=self.device)
        self.done_spec = Categorical(
            n=2, device=self.device)  # 0/1 for False/True

    @staticmethod
    def set_seed(seed: int):
        """
        Set random seeds for reproducibility.

        Args:
            seed: Integer seed value
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def reset(self, tensordict: TensorDict) -> TensorDict:
        """ Initialize new episode - returns FIRST observation """
        grid_size = tensordict.get("grid_size", self.grid_size)
        present_count = tensordict.get("present_count", self.present_count)

        self._init_specs(grid_size, present_count)

        # Create initial grid state
        grid = torch.zeros(torch.Size(self.grid_size), dtype=torch.uint8)

        # Return as TensorDict with observation keys
        return TensorDict({
            "grid": grid,
            "presents": self.presents,
            "present_count": self.present_count
        }, batch_size=tensordict.batch_size, device=self.device)

    def step(self, tensordict: TensorDict) -> TensorDict:
        """ Execute one action - returns NEXT observation + reward + done """
        # Get current state and action
        grid = tensordict.get("grid").clone()
        presents = tensordict.get("presents")
        present_count = tensordict.get("present_count").clone()

        present_idx = tensordict.get(("action, present_idx"))
        x = tensordict.get(("action", "x"))
        y = tensordict.get(("action", "y"))
        rot = tensordict.get(("action", "rot"))
        flip = tensordict.get(("action", "flip"))

        # Get present
        present = self.presents[present_idx]
        present = torch.rot90(present, rot)
        present = torch.flip(present, flip)

        # If collision, exit early
        grid_region = grid[y:y+3, x:x+3]
        if torch.any(present & grid_region):
            obs_dict = {
                "grid": grid,
                "presents": presents,
                "present_count": present_count,
            }
            return TensorDict({
                "next": {
                    "observation": obs_dict
                },
                "reward": torch.tensor(-20),
                "done": torch.tensor(True)
            }, batch_size=tensordict.batch_size, device=self.device)

        # If action used present we cannot place, exit early
        if present_count[present_idx] < 1:
            obs_dict = {
                "grid": grid,
                "presents": presents,
                "present_count": present_count,
            }
            return TensorDict({
                "next": {
                    "observation": obs_dict
                },
                "reward": torch.tensor(-20),
                "done": torch.tensor(True)
            }, batch_size=tensordict.batch_size, device=self.device)

        # Otherwise, update tensors
        present_count[present_idx] -= 1
        grid[y:y+3, x:x+3] = present

        # Base reward
        reward = torch.tensor(2.0)

        # Check if all shapes are placed
        done = torch.tensor(False)
        if torch.sum(present_count) == 0:
            done = torch.tensor(True)

        # Return next state
        obs_dict = {
            "grid": grid,
            "presents": presents,
            "present_count": present_count,
        }

        return TensorDict({
            "next": {
                "observation": obs_dict
            },
            "reward": reward,
            "done": done
        }, batch_size=tensordict.batch_size, device=self.device)
