""" Code for the neural network itself. """

from dataclasses import dataclass, field

import torch
from torch import distributions as d
from torchrl.data import ReplayBuffer
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, CompositeDistribution

from nn.actor_critic import PresentActorCritic
from nn.env import PresentEnv
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

    def __init__(self, data_buff: ReplayBuffer, config=None, device=None):
        # Device / environment setup
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.env = PresentEnv().to(self.device)

        # Set up network configuration
        self.config = config or PPOConfig()

        # Set up ActorCritic / TD module / Actor
        self.policy = PresentActorCritic().to(device)
        self.td_module = TensorDictModule(
            self.policy,
            in_keys=[
                "grid", "presents", "present_count"
            ],
            out_keys=[
                ("params", "present_idx_logits"),
                ("params", "rot_logits"),
                ("params", "flip_logits"),
                ("params", "x"),
                ("params", "y")
            ]
        )
        self.actor = ProbabilisticActor(
            module=self.td_module,
            spec=self.env.action_spec,
            in_keys=["params"],
            distribution_class=CompositeDistribution,
            distribution_kwargs={
                "present_idx": d.Categorical,
                "rot": d.Categorical,
                "flip": d.Bernoulli,
                "x": d.SoftmaxTransform,
                "y": d.SoftmaxTransform,
            },
            return_log_prob=True
        )

        # Value network
        self.value_module = ValueOperator(
            module=self.policy,
            in_keys=[
                "grid", "presents", "present_count"
            ]
        )

        # Collector
        self.collector = SyncDataCollector(
            self.env,
            self.policy,
            frames_per_batch=self.config.batch_size,
            total_frames=self.config.num_steps,
            split_trajs=False,
            device=self.device
        )

        # Data for model
        self.data = data_buff

        # Loss function config
        self.adv_module = GAE(
            gamma=self.config.gamma,
            lmbda=self.config.gae_lambda,
            value_network=self.value_module,
            average_gae=True,
            device=torch.device(self.device)
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.value_module,
            clip_epsilon=self.config.clip_epsilon,
            entropy_bonus=bool(self.config.entropy_coef),
            entropy_coeff=self.config.entropy_coef,
            critic_coeff=1.0,
            loss_critic_type="smooth_l1"
        )

        self.optimizer = torch.optim.Adam(
            self.loss_module.parameters(),
            lr=self.config.learning_rate
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.config.total_frames // self.config.num_steps, 0.0
        )

    def select_action(self, state, training=True):
        """
        Select an action using the current policy
        """
