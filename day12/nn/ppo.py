""" Code for the neural network itself. """

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from nn.actor_critic import ActorCritic
from nn.env import PresentPlacementEnv
from nn.ppo_config import PPOConfig


@dataclass
class Buffers:
    """Stores experience buffers"""
    states: list[TensorDict] = field(default_factory=list)
    actions: list[TensorDict] = field(default_factory=list)
    rewards: list[TensorDict] = field(default_factory=list)
    dones: list[TensorDict] = field(default_factory=list)
    values: list[TensorDict] = field(default_factory=list)
    log_probs: list[TensorDict] = field(default_factory=list)

    def reset(self):
        """ Resets states of buffers """
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()


class PPO:
    """ Implementation of the PPO actor-critic network"""

    def __init__(self, config=None, device=None):
        # Device / environment setup
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.env = PresentPlacementEnv().to(self.device)

        # Set up network configuration
        self.config = config or PPOConfig()

        # Set up ActorCritic network (the "brain")
        self.actor_critic = ActorCritic().to(device)

        # The optimizer - updates the brain
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=self.config.learning_rate
        )

    def select_action(self, state, training=True):
        """
        Select an action using the current policy

        Args:
            state: Current state as tensordict
            training: Whether to add exploration noise

        Returns:
            action_dict: Dictionary with action components
            log_prob: Log probability of selected action
            value: State value estimate
        """
        # Get observations
        grid = state.get("grid").to(self.device)
        presents = state.get("presents").to(self.device)
        present_count = state.get("present_count").to(self.device)

        # Get policy outputs
        with torch.set_grad_enabled(training):
            outputs = self.actor_critic(
                grid, presents, present_count, training=training
            )

        # Sample discrete actions
        present_idx_probs = F.softmax(outputs["present_idx_logits"], dim=-1)
        present_idx = torch.multinomial(present_idx_probs, 1).squeeze(-1)

        rot_probs = F.softmax(outputs["rot_logits"], dim=-1)
        rot = torch.multinomial(rot_probs, 1).squeeze(-1)

        flip_probs = torch.sigmoid(outputs["flip_logits"])
        flip = (torch.rand_like(flip_probs) < flip_probs).long()

        # Sample continuous positions
        if training:
            # Add exploration noise
            x_std = torch.exp(outputs["x_log_std"])
            y_std = torch.exp(outputs["y_log_std"])

            x_norm = outputs["x_mean"] + torch.randn_like(x_std) * x_std
            y_norm = outputs["y_mean"] + torch.randn_like(y_std) * y_std
        else:
            # Use mean (no exploration)
            x_norm = outputs["x_mean"]
            y_norm = outputs["y_mean"]

        # Clip to valid range
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        y_norm = torch.clamp(y_norm, 0.0, 1.0)

        # Convert to actual coordinates
        h, w = grid.shape[-2], grid.shape[-1]
        max_x = w - 3
        max_y = h - 3

        x = (x_norm * max_x).round().long()
        y = (y_norm * max_y).round().long()

        # Compute log probability
        if training:
            log_prob = self._compute_log_prob(
                outputs,
                present_idx,
                rot,
                flip,
                x_norm,
                y_norm
            )
        else:
            log_prob = None

        # Prepare action dictionary
        action_dict = {
            "present_idx": present_idx.cpu(),
            "x": x.cpu(),
            "y": y.cpu(),
            "rot": rot.cpu(),
            "flip": flip.cpu()
        }

        return action_dict, log_prob, outputs["value"]

    def _compute_log_prob(self, outputs, present_idx, rot, flip, x_norm, y_norm):
        """Compute log probability of actions"""
        log_probs = []

        # Present index log prob
        present_log_probs = F.log_softmax(
            outputs["present_idx_logits"], dim=-1)
        log_probs.append(
            present_log_probs.gather(1, present_idx.unsqueeze(-1)).squeeze(-1)
        )

        # Rotation log prob
        rot_log_probs = F.log_softmax(outputs["rot_logits"], dim=-1)
        log_probs.append(
            rot_log_probs.gather(1, rot.unsqueeze(-1)).squeeze(-1)
        )

        # Flip log prob (bernoulli)
        flip_probs = torch.sigmoid(outputs["flip_logits"])
        flip_log_probs = flip * torch.log(flip_probs + 1e-8) + \
            (1 - flip) * torch.log(1 - flip_probs + 1e-8)
        log_probs.append(flip_log_probs.sum(dim=-1))

        # Continuous positions log prob (normal distribution)
        x_mean = outputs["x_mean"]
        x_std = torch.exp(outputs["x_log_std"])
        x_log_prob = -0.5 * ((x_norm - x_mean) / x_std) ** 2 \
                     - torch.log(x_std) - 0.5 * \
            torch.log(2 * torch.tensor(torch.pi))
        log_probs.append(x_log_prob)

        y_mean = outputs["y_mean"]
        y_std = torch.exp(outputs["y_log_std"])
        y_log_prob = -0.5 * ((y_norm - y_mean) / y_std) ** 2 \
                     - torch.log(y_std) - 0.5 * \
            torch.log(2 * torch.tensor(torch.pi))
        log_probs.append(y_log_prob)

        # Sum all log probabilities
        total_log_prob = torch.stack(log_probs).sum(dim=0)
        return total_log_prob
