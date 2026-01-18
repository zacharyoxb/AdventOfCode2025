""" Neural network for packing presents with orientation masks """

import torch
from tensordict import TensorDict

from torchrl.data import Bounded, Composite, Unbounded, Categorical
from torchrl.envs import EnvBase


class PresentPlacementEnv(EnvBase):
    """ RL environment for present placement """

    def __init__(
            self,
            td_params: TensorDict,
            batch_size=None,
            seed=None,
            device="cpu"
    ):
        if batch_size is None:
            batch_size = torch.Size([])

        self.batch_size = batch_size
        super().__init__(device=device, batch_size=self.batch_size)
        self.rng = None

        self._make_spec(td_params)
        self.default_params = td_params

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(int(seed))

    @staticmethod
    def gen_params(
            grid_size: tuple[int, int],
            presents: torch.Tensor,
            present_count: torch.Tensor,
            batch_size=None
    ) -> TensorDict:
        """ Generates parameters for specs """
        if batch_size is None:
            batch_size = []

        w, h = grid_size

        td = TensorDict(
            {
                "grid_size": torch.tensor([w, h], dtype=torch.int64),
                "presents": presents,
                "present_count": present_count,
                "max_present_idx": 4,
                "max_x": w-3,
                "max_y": h-3,
                "max_rot": 3,
                "max_flip": 1
            }
        )

        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    def _make_spec(
            self,
            td_params
    ):
        # Extract all params as individual variables
        grid_size = td_params.get("grid_size")
        w, h = grid_size
        presents = td_params.get("presents")
        present_count = td_params.get("present_count")
        max_present_idx = td_params.get("max_present_idx")
        max_x = td_params.get("max_x")
        max_y = td_params.get("max_y")
        max_rot = td_params.get("max_rot")
        max_flip = td_params.get("max_flip")

        # Observation spec: what the agent sees
        self.observation_spec = Composite({
            "grid": Bounded(low=0, high=1, shape=torch.Size((h, w)),
                            dtype=torch.uint8, device=self.device),
            "presents": Bounded(low=0, high=1, shape=presents.shape,
                                dtype=torch.uint8, device=self.device),
            "present_count": Unbounded(shape=present_count.shape, dtype=torch.int64,
                                       device=self.device),
        })

        # Action spec: what the agent can do
        self.action_spec = Composite({
            "present_idx": Bounded(low=0, high=max_present_idx, shape=1, dtype=torch.uint8,
                                   device=self.device),
            "x": Bounded(low=0, high=max_x, shape=1, dtype=torch.int64,
                         device=self.device),
            "y": Bounded(low=0, high=max_y, shape=1, dtype=torch.int64,
                         device=self.device),
            "rot": Bounded(low=0, high=max_rot, shape=1,
                           dtype=torch.uint8, device=self.device),
            "flip": Bounded(low=0, high=max_flip, shape=torch.Size([2]), dtype=torch.uint8,
                            device=self.device)
        })

        # Reward and done specs
        self.reward_spec = Unbounded(shape=torch.Size(
            [1]), dtype=torch.int64, device=self.device)
        self.done_spec = Categorical(
            n=2, shape=torch.Size([1]), dtype=torch.bool, device=self.device)  # 0/1 for False/True

    def _set_seed(self, seed: int | None = None):
        """
        Set random seeds for reproducibility.

        Args:
            seed: Integer seed value
        """
        self.rng = torch.manual_seed(seed)

    def _reset(self, tensordict, **kwargs) -> TensorDict:
        """ Initialize new episode - returns FIRST observation """

        if tensordict is None:
            tensordict = self.default_params

        grid_size = tensordict.get("grid_size")
        presents = tensordict.get("presents")
        present_count = tensordict.get("present_count")

        self._make_spec(tensordict)

        # zeros expects height first
        w, h = grid_size

        # Create initial grid state
        grid = torch.zeros(
            (h, w), dtype=torch.uint8, device=self.device)

        # Return as TensorDict with observation keys
        return TensorDict({
            "grid": grid,
            "presents": presents,
            "present_count": present_count,
        }, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """ Execute one action - returns NEXT observation + reward + done """
        # Get current state and action
        grid = tensordict.get("grid").clone()
        presents = tensordict.get("presents")
        present_count = tensordict.get("present_count").clone()

        present_idx = int(tensordict.get("present_idx"))
        x = int(tensordict.get("x"))
        y = int(tensordict.get("y"))
        rot = int(tensordict.get("rot"))
        flip = tuple(tensordict.get("flip").tolist())

        # Get present
        present = presents[present_idx]
        present = torch.rot90(present, rot)
        if sum(flip) != 0:
            present = torch.flip(present, flip)

        # If collision, exit early
        grid_region = grid[y:y+3, x:x+3]
        if torch.any(present & grid_region):
            return TensorDict({
                "grid": grid,
                "presents": presents,
                "present_count": present_count,
                "reward": torch.tensor(-20),
                "done": torch.tensor(True)
            }, batch_size=self.batch_size, device=self.device)

        # If action used present we cannot place, exit early
        if present_count[present_idx] < 1:
            return TensorDict({
                "grid": grid,
                "presents": presents,
                "present_count": present_count,
                "reward": torch.tensor(-20),
                "done": torch.tensor(True)
            }, batch_size=self.batch_size, device=self.device)

        # Otherwise, update tensors
        present_count[present_idx] -= 1
        grid[y:y+3, x:x+3] = present

        # Base reward
        reward = torch.tensor(2, dtype=torch.int64)

        # Check if all shapes are placed
        done = torch.tensor(False)
        if torch.sum(present_count) == 0:
            done = torch.tensor(True)

        return TensorDict({
            "grid": grid,
            "presents": presents,
            "present_count": present_count,
            "reward": reward,
            "done": done
        }, batch_size=self.batch_size, device=self.device)
