import numpy as np
import torch
import torch.nn as nn
import logging
from model import ActorNet, CriticNet
from replay_memory import PrioritizedReplayBuffer

DATA_DEMO = 0
DATA_RUNTIME = 1


class DDPGfDAgent(nn.Module):
    def __init__(self, conf, device):
        super(DDPGfDAgent, self).__init__()

        self.conf = conf
        self.device = device
        self.logger = logging.getLogger('DDPGfD')

        self.actor_b = ActorNet(
            self.conf.state_dim, self.conf.action_dim, self.device)
        self.actor_t = ActorNet(
            self.conf.state_dim, self.conf.action_dim, self.device)

        self.critic_b = CriticNet(
            self.conf.state_dim, self.conf.action_dim, self.device)
        self.critic_t = CriticNet(
            self.conf.state_dim, self.conf.action_dim, self.device)

        self.rs = np.random.RandomState(self.conf.seed)

        self.memory = PrioritizedReplayBuffer(
            size=self.conf.replay_buffer_size,
            seed=self.conf.seed,
            alpha=0.3,
            beta_init=1.0,
            beta_inc_n=100,
        )

    @staticmethod
    def obs2tensor(state):
        return torch.from_numpy(state).float()

    def update_target(self, src, tgt, episode=-1):  # update to target network
        if not self.conf.discrete_update or episode == -1:  # soft update
            for src_param, tgt_param in zip(
                    src.parameters(), tgt.parameters()):
                tgt_param.data.copy_(
                    self.conf.tau * src_param.data
                    + (1.0 - self.conf.tau) * tgt_param.data)
            self.logger.debug('(Soft)Update target network')
        else:
            if episode % self.conf.discrete_update_eps == 0:
                for src_param, tgt_param in zip(
                        src.parameters(), tgt.parameters()):
                    tgt_param.data.copy_(src_param.data)
                self.logger.info(
                    '(Discrete)Update target network,episode={}'.format(
                        episode))
