import numpy as np
import torch
import torch.nn as nn
import logging
from model import ActorNet
from model import CriticNet
from training_utils import weight_init
from replay_memory import PrioritizedReplayBuffer

# constant signifying that data was collected from a demonstration
DATA_DEMO = 0
# constant signifying that data was collected during training
DATA_RUNTIME = 1


class DDPGfDAgent(nn.Module):
    def __init__(self, conf, device):
        super(DDPGfDAgent, self).__init__()

        self.conf = conf
        self.device = device
        self.logger = logging.getLogger('DDPGfD')

        # Create the actor base and target networks.
        self.actor_b = ActorNet(
            self.conf.state_dim, self.conf.action_dim, self.device)
        self.actor_t = ActorNet(
            self.conf.state_dim, self.conf.action_dim, self.device)

        # Create the critic base and target network.
        self.critic_b = CriticNet(
            self.conf.state_dim, self.conf.action_dim, self.device)
        self.critic_t = CriticNet(
            self.conf.state_dim, self.conf.action_dim, self.device)

        # Set random seed.
        self.rs = np.random.RandomState(self.conf.seed)

        # Initialize base policy parameters.
        self.actor_b.apply(weight_init)
        self.critic_b.apply(weight_init)

        # Initialize target policy parameters.
        self.init_target(self.actor_b, self.actor_t)
        self.init_target(self.critic_b, self.critic_t)

        # Create the replay buffer.
        self.memory = PrioritizedReplayBuffer(
            size=self.conf.replay_buffer_size,
            seed=self.conf.seed,
            alpha=0.3,
            beta_init=1.0,
            beta_inc_n=100,
        )

    @staticmethod
    def obs2tensor(state):
        """Convert observations to a torch-compatible format."""
        return torch.from_numpy(state).float()

    @staticmethod
    def init_target(src, tgt):
        """Initialize target policy parameters."""
        for src_param, tgt_param in zip(src.parameters(), tgt.parameters()):
            tgt_param.data.copy_(src_param.data)

    def update_target(self, src, tgt):
        """Perform soft target updates."""
        for src_param, tgt_param in zip(src.parameters(), tgt.parameters()):
            tgt_param.data.copy_(
                self.conf.tau * src_param.data
                + (1.0 - self.conf.tau) * tgt_param.data)

        self.logger.debug('(Soft)Update target network')
