""" Neural network for packing presents with orientation masks """

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import torch
from tensordict import TensorDict

from torchrl.data import Bounded, Composite, Unbounded, Categorical


class Stage(Enum):
    """ Represents current stage """
    SELECT_SHAPE = 0
    PLACE_SHAPE = 1


@dataclass
class StageState:
    """ Stores state of stage """
    current_stage: Stage = Stage.SELECT_SHAPE
    selected_shape: Optional[torch.Tensor] = None
    selected_x: Optional[int] = None
    selected_y: Optional[int] = None


@dataclass
class AgentData:
    """ Stores Agent Data for RL """
    observation_spec: Optional[Composite] = None
    action_spec: Optional[Composite] = None
    reward_spec: Optional[Composite] = None
    done_spec: Optional[Composite] = None


class ShapePlacementEnv():
    """ RL environment for shape placement """

    def __init__(self, grid_size, shapes):
        super().__init__()
        # Track grid
        self.grid_size = grid_size
        self.num_shapes = len(shapes)

        # Track current stage
        self.stage_state = StageState()
        # Agent data
        self.agent_data = AgentData()

        # init spec
        self._make_spec()

    def _reset(self, tensordict: TensorDict) -> TensorDict:
        """ Initialize new episode - returns FIRST observation """
        # Create initial grid state
        grid = torch.zeros(self.grid_size, dtype=torch.float32)

        # Return as TensorDict with observation keys
        return TensorDict({
            "grid": grid,
        }, batch_size=[])

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """ Execute one action - returns NEXT observation + reward + done """
        # 1. Get current state and action
        grid = tensordict["grid"].clone()
        action_type = tensordict["action", "action_type"]
        x = tensordict["action", "x"]
        y = tensordict["action", "y"]

        # 2. Apply your game logic
        if action_type == 0:  # Place
            grid[x, y] = 1.0
            reward = torch.tensor(1.0)
        else:
            reward = torch.tensor(-0.1)

        # 3. Check termination
        done = torch.tensor(False)
        if grid.sum() > 20:  # Example condition
            done = torch.tensor(True)

        # 4. Return NEXT state
        return TensorDict({
            "grid": grid,          # Next observation
            "reward": reward.unsqueeze(-1),  # Shape [1]
            "done": done.unsqueeze(-1)       # Shape [1]
        }, batch_size=tensordict.batch_size)

    def _make_spec(self):
        """ Define observation and action spaces - MUST CALL IN __init__ """
        # Observation spec: what the agent sees
        self.observation_spec = Composite({
            "grid": Bounded(low=0, high=1, shape=self.grid_size, dtype=torch.float32),
        })

        # Action spec: what the agent can do
        self.agent_data.action_spec = Composite({
            # e.g., 0=place, 1=remove, 2=rotate
            "action_type": Categorical(n=3),
            "x": Bounded(low=0, high=9, shape=1, dtype=torch.int64),
            "y": Bounded(low=0, high=9, shape=1, dtype=torch.int64)
        })

        # Reward and done specs
        self.agent_data.reward_spec = Unbounded(shape=torch.Size([1]))
        self.agent_data.done_spec = Categorical(n=2)  # 0/1 for False/True
