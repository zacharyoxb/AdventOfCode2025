""" Neural network for packing presents with orientation masks """

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data import Bounded, Composite, Unbounded, Categorical
from torchrl.envs import EnvBase


class PresentPlacementEnv(EnvBase):
    """ RL environment for present placement """

    def __init__(
            self,
            td_params: TensorDict,
            seed=None,
            device="cpu"
    ):

        super().__init__(device=device, batch_size=torch.Size([]))
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
                "params": TensorDict(
                    {
                        "grid_size": (w, h),
                        "presents": presents,
                        "present_count": present_count,
                        "max_present_idx": 4,
                        "max_x": w-3,
                        "max_y": h-3,
                        "max_rot": 3,
                        "max_flip": 1
                    }
                )
            }
        )

        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    def make_composite_from_td(self, td):
        """
        Custom function to convert a ``tensordict`` in a similar spec structure
        of unbounded values.
        """
        composite = Composite(
            {
                key: self.make_composite_from_td(tensor)
                if isinstance(tensor, TensorDictBase)
                else Unbounded(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
                for key, tensor in td.items()
            },
            shape=td.shape,
        )
        return composite

    def _make_spec(
            self,
            td_params
    ):
        # Observation spec: what the agent sees
        self.observation_spec = Composite({
            "grid": Bounded(low=0, high=1, shape=td_params.get(("params", "grid_size")),
                            dtype=torch.uint8, device=self.device),
            "presents": Bounded(low=0, high=1, shape=td_params.get(("params", "presents")).shape,
                                dtype=torch.uint8, device=self.device),
            "present_count": Unbounded(shape=5, dtype=torch.int64, device=self.device),
            "params": self.make_composite_from_td(td_params.get("params"))
        })

        # Action spec: what the agent can do
        self.action_spec = Composite({
            "present_idx": Bounded(low=0, high=5, shape=1, dtype=torch.uint8, device=self.device),
            "x": Bounded(low=0, high=td_params.get(("params", "max_x")), shape=1, dtype=torch.int64,
                         device=self.device),
            "y": Bounded(low=0, high=td_params.get(("params", "max_y")), shape=1, dtype=torch.int64,
                         device=self.device),
            "rot": Bounded(low=0, high=td_params.get(("params", "max_rot")), shape=1,
                           dtype=torch.uint8, device=self.device),
            "flip": Bounded(low=0, high=td_params.get(("params", "max_flip")),
                            shape=torch.Size([2]), dtype=torch.uint8, device=self.device)
        })

        # Reward and done specs
        self.reward_spec = Unbounded(shape=torch.Size(
            [1]), dtype=torch.int64, device=self.device)
        self.done_spec = Categorical(
            n=2, shape=torch.Size([1]), device=self.device)  # 0/1 for False/True

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

        grid_size = tensordict.get(("params", "grid_size"))
        presents = tensordict.get(("params", "presents"))
        present_count = tensordict.get(("params", "present_count"))

        self._make_spec(tensordict)

        # Create initial grid state
        grid = torch.zeros(
            tuple(grid_size), dtype=torch.uint8, device=self.device)

        # Return as TensorDict with observation keys
        return TensorDict({
            "grid": grid,
            "presents": presents,
            "present_count": present_count,
            "params": tensordict.get("params")
        }, batch_size=tensordict.batch_size, device=self.device)

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
            obs_dict = {
                "grid": grid,
                "presents": presents,
                "present_count": present_count,
                "params": tensordict.get("params")
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
                "params": tensordict.get("params")
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
            "params": tensordict.get("params")
        }

        return TensorDict({
            "next": {
                "observation": obs_dict
            },
            "reward": reward,
            "done": done
        }, batch_size=tensordict.batch_size, device=self.device)
