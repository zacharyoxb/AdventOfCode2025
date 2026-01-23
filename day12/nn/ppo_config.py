""" Config classes for PPO """
from dataclasses import dataclass, field


@dataclass
class LearningConfig:
    """ Learning rate and optimization parameters """
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5


@dataclass
class DiscountConfig:
    """ Temporal discounting and advantage estimation """
    gamma: float = 0.99
    gae_lambda: float = 0.95


@dataclass
class LossConfig:
    """ Loss function weights and coefficients """
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01


@dataclass
class TrainingConfig:
    """ Training process parameters """
    ppo_epochs: int = 4
    batch_size: int = 64
    num_steps: int = 2048
    total_frames: int = 10_000_000


@dataclass
class PPOConfig:
    """Complete PPO Hyperparameters Configuration"""

    learning: LearningConfig = field(default=LearningConfig())
    discount: DiscountConfig = field(default=DiscountConfig())
    loss: LossConfig = field(default=LossConfig())
    training: TrainingConfig = field(default=TrainingConfig())

    @property
    def learning_rate(self):
        """ Learning rate for the optimizer """
        return self.learning.learning_rate

    @property
    def max_grad_norm(self):
        """ Max Gradient Norm: Gradient clipping threshold """
        return self.learning.max_grad_norm

    @property
    def gamma(self):
        """ Gamma (γ): Future reward discount factor """
        return self.discount.gamma

    @property
    def gae_lambda(self):
        """ GAE Lambda (λ): Generalized Advantage Estimation parameter """
        return self.discount.gae_lambda

    @property
    def clip_epsilon(self):
        """ Clip Epsilon (ε): PPO clipping parameter """
        return self.loss.clip_epsilon

    @property
    def value_loss_coef(self):
        """ Value Loss Coefficient: Critic loss weight """
        return self.loss.value_loss_coef

    @property
    def entropy_coef(self):
        """ Entropy Coefficient: Exploration bonus """
        return self.loss.entropy_coef

    @property
    def ppo_epochs(self):
        """ PPO Epochs: Number of optimization passes per batch """
        return self.training.ppo_epochs

    @property
    def batch_size(self):
        """ Batch Size: Mini-batch size for optimization """
        return self.training.batch_size

    @property
    def num_steps(self):
        """ Number of steps to collect before updating """
        return self.training.num_steps

    @property
    def total_frames(self):
        """ Number of frames to use in total """
        return self.training.total_frames
