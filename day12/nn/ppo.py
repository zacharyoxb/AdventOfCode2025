""" Code for the neural network itself. """


from dataclasses import dataclass
import torch

from nn.actor_critic import ActorCritic
from nn.env import PresentPlacementEnv


@dataclass
class PPOConfig:
    """PPO Hyperparameters Configuration"""

    # Learning rate for the optimizer
    learning_rate: float = 3e-4

    # Gamma (γ): Future reward discount factor
    # How much to care about future rewards vs immediate rewards
    # 0.99 = "Plan for the distant future"
    # 0.9 = "Focus on next few steps"
    gamma: float = 0.99

    # GAE Lambda (λ): Generalized Advantage Estimation parameter
    # Balances bias-variance tradeoff in advantage estimation
    # 0.95 = Good balance between short and long-term estimates
    # 1.0 = Monte Carlo (high variance, low bias)
    # 0.0 = TD(0) (low variance, high bias)
    gae_lambda: float = 0.95

    # Clip Epsilon (ε): PPO clipping parameter
    # Maximum allowed policy change per update
    # 0.2 = "Don't change policy by more than 20%"
    # Prevents destructive updates that ruin learning
    clip_epsilon: float = 0.2

    # Value Loss Coefficient: Critic loss weight
    # How much to care about value predictions vs policy
    # 0.5 = "Equal attention to critic and actor"
    # Higher = focus more on accurate value estimates
    value_loss_coef: float = 0.5

    # Entropy Coefficient: Exploration bonus
    # Encourages trying new actions
    # 0.01 = "Add small exploration bonus"
    # Higher = more random exploration
    entropy_coef: float = 0.01

    # Max Gradient Norm: Gradient clipping threshold
    # Prevents exploding gradients
    # 0.5 = "Clip gradients if norm exceeds 0.5"
    max_grad_norm: float = 0.5

    # PPO Epochs: Number of optimization passes per batch
    # How many times to reuse collected experience
    # 4 = "Learn from same data 4 times"
    ppo_epochs: int = 4

    # Batch Size: Mini-batch size for optimization
    # Number of experiences to learn from at once
    # 64 = "Process 64 experiences per update"
    batch_size: int = 64


class PPO:
    """ Implementation of the PPO actor-critic network"""

    def __init__(self, config=None, device=None):
        # Set up network configuration
        self.config = config or PPOConfig()

        # Device / environment setup
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.env = PresentPlacementEnv().to(self.device)

        # Set up ActorCritic network (the "brain")
        self.actor_critic = ActorCritic().to(device)

        # The optimizer - updates the brain
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=self.config.learning_rate
        )
