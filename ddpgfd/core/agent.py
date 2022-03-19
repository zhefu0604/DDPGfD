"""TODO."""
import numpy as np
import torch
import torch.nn as nn
import logging

from ddpgfd.core.model import ActorNet
from ddpgfd.core.model import CriticNet
from ddpgfd.core.replay_memory import PrioritizedReplayBuffer
from ddpgfd.core.training_utils import GaussianActionNoise

# constant signifying that data was collected from a demonstration
DATA_DEMO = 0
# constant signifying that data was collected during training
DATA_RUNTIME = 1


class DDPGfDAgent(nn.Module):
    """TODO.

    Attributes
    ----------
    conf : TODO
        TODO
    agent_conf : TODO
        TODO
    device : TODO
        TODO
    logger : TODO
        TODO
    actor_b : TODO
        TODO
    actor_t : TODO
        TODO
    critic_b : TODO
        TODO
    critic_t : TODO
        TODO
    memory : TODO
        TODO
    optimizer_actor : TODO
        TODO
    optimizer_critic : TODO
        TODO
    action_noise : TODO
        TODO
    """

    def __init__(self, conf, device):
        """TODO.

        Parameters
        ----------
        conf : TODO
            TODO
        device : TODO
            TODO
        """
        super(DDPGfDAgent, self).__init__()

        self.conf = conf
        self.agent_conf = self.conf.agent_config
        self.device = device
        self.logger = logging.getLogger('DDPGfD')

        # TODO
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

        # Create the actor base and target networks.
        self.actor_b = ActorNet(
            self.agent_conf.state_dim, self.agent_conf.action_dim, self.device)
        self.actor_t = ActorNet(
            self.agent_conf.state_dim, self.agent_conf.action_dim, self.device)

        # Create the critic base and target network.
        self.critic_b = CriticNet(
            self.agent_conf.state_dim, self.agent_conf.action_dim, self.device)
        self.critic_t = CriticNet(
            self.agent_conf.state_dim, self.agent_conf.action_dim, self.device)

        # Set random seed.
        self.rs = np.random.RandomState(self.agent_conf.seed)

        # Initialize target policy parameters.
        self.init_target(self.actor_b, self.actor_t)
        self.init_target(self.critic_b, self.critic_t)

        # Create the replay buffer.
        self.memory = PrioritizedReplayBuffer(
            size=self.agent_conf.replay_buffer_size,
            seed=self.agent_conf.seed,
            alpha=0.3,
            beta_init=1.0,
            beta_inc_n=100,
        )

        # Crate optimizer.
        self.optimizer_actor = None
        self.optimizer_critic = None
        self._set_optimizer()

        # loss function setting
        reduction = 'none'
        if self.conf.train_config.mse_loss:
            self.q_criterion = nn.MSELoss(reduction=reduction)
        else:
            self.q_criterion = nn.SmoothL1Loss(reduction=reduction)

        # exploration noise
        self.action_noise = GaussianActionNoise(sigma=0.1, ac_dim=1)
        # self.action_noise = OrnsteinUhlenbeckActionNoise(
        #     mu=np.zeros(self.agent_conf.action_dim),
        #     sigma=self.agent_conf.action_noise_std)

    def _set_optimizer(self):
        """Create the optimizer objects."""
        # Create actor optimizer.
        self.optimizer_actor = torch.optim.Adam(
            self.actor_b.parameters(),
            lr=self.conf.train_config.lr_rate,
            weight_decay=self.conf.train_config.w_decay)

        # Create critic optimizer.
        self.optimizer_critic = torch.optim.Adam(
            self.critic_b.parameters(),
            lr=self.conf.train_config.lr_rate,
            weight_decay=self.conf.train_config.w_decay)

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
                self.agent_conf.tau * src_param.data
                + (1.0 - self.agent_conf.tau) * tgt_param.data)

    def update_agent(self, update_step):
        """Sample experience and update.

        Parameters
        ----------
        update_step : int
            number of policy updates to perform

        Returns
        -------
        float
            critic loss
        float
            actor loss
        int
            the number of demonstration in the training batches
        int
            the total number of samples in the training batches
        """
        losses_critic = []
        losses_actor = []
        demo_cnt = []
        batch_sz = 0
        not_done = 1.  # TODO
        if self.memory.ready():
            # Sample a batch of data.
            (batch_s, batch_a, batch_r, batch_s2, batch_gamma,
             batch_flags), weights, idxes = self.memory.sample(
                self.conf.train_config.batch_size)

            # Convert to pytorch compatible object.
            batch_s = batch_s.to(self.device)
            batch_a = batch_a.to(self.device)
            batch_r = batch_r.to(self.device)
            batch_s2 = batch_s2.to(self.device)
            batch_gamma = batch_gamma.to(self.device)
            weights = torch.from_numpy(weights.reshape(-1, 1)).float().to(
                self.device)
            batch_sz += batch_s.shape[0]

            with torch.no_grad():
                # Select action according to policy and add clipped noise.
                noise = (torch.randn_like(batch_a) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip)
                next_action = (self.actor_t(batch_s2) + noise).clamp(-1, 1)

                # Compute the target Q value.
                target_q1, target_q2 = self.critic_t(
                    torch.cat((batch_s2, next_action), dim=1))
                target_q = torch.min(target_q1, target_q2)
                target_q = batch_r + not_done * batch_gamma * target_q

            # Get current Q estimates
            q1, q2 = self.critic_b(torch.cat((batch_s, batch_a), dim=1))

            # Compute critic loss.
            critic_loss = self.q_criterion(q1, target_q).mean() + \
                self.q_criterion(q2, target_q).mean()

            # Optimize the critic.
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()
            losses_critic.append(critic_loss.item())

            # Record Demo count.
            d_flags = torch.from_numpy(batch_flags)
            demo_select = d_flags == DATA_DEMO
            n_act = demo_select.sum().item()
            demo_cnt.append(n_act)

            # Delayed policy updates
            if update_step % self.policy_freq == 0:
                # Compute actor losses.
                actor_loss = -self.critic_b.Q1(
                    torch.cat((batch_s, self.actor_b(batch_s)), dim=1)).mean()

                # Optimize the actor.
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()
                losses_actor.append(actor_loss.item())

                # Update priorities in the replay buffer.
                if not self.agent_conf.no_per:
                    priority = ((q1.detach() - target_q).pow(2) +
                                q_act.detach().pow(2)).numpy().ravel() \
                        + self.agent_conf.const_min_priority
                    priority[batch_flags == DATA_DEMO] += \
                        self.agent_conf.const_demo_priority

                    self.memory.update_priorities(idxes, priority)

                # Update target.
                self.update_target(self.actor_b, self.actor_t)
                self.update_target(self.critic_b, self.critic_t)

        demo_n = max(sum(demo_cnt), 1e-10)

        return np.sum(losses_critic), np.sum(losses_actor), demo_n, batch_sz

    def save(self, progress_path, epoch):
        """TODO.

        Parameters
        ----------

        """
        self._save_model_weight(
            self.actor_b, progress_path, epoch, prefix='actor_b')
        self._save_model_weight(
            self.actor_t, progress_path, epoch, prefix='actor_t')
        self._save_model_weight(
            self.critic_b, progress_path, epoch, prefix='critic_b')
        self._save_model_weight(
            self.critic_t, progress_path, epoch, prefix='critic_t')

    def load(self, progress_path, epoch):
        """TODO.

        Parameters
        ----------

        """
        self.actor_b.load_state_dict(self.restore_model_weight(
            progress_path, epoch, prefix='actor_b'))
        self.actor_t.load_state_dict(self.restore_model_weight(
            progress_path, epoch, prefix='actor_t'))
        self.critic_b.load_state_dict(self.restore_model_weight(
            progress_path, epoch, prefix='critic_b'))
        self.critic_t.load_state_dict(self.restore_model_weight(
            progress_path, epoch, prefix='critic_t'))

    def _save_model_weight(self, model, progress_path, epoch, prefix=''):
        """TODO.

        Parameters
        ----------
        model : TODO
            TODO
        epoch : TODO
            TODO
        prefix : TODO
            TODO
        """
        name = progress_path + prefix + 'model-' + str(epoch) + '.tp'
        torch.save(model.state_dict(), name)

    def _restore_model_weight(self, progress_path, epoch, prefix=''):
        """TODO.

        Parameters
        ----------
        epoch : TODO
            TODO
        prefix : TODO
            TODO
        """
        name = progress_path + prefix + 'model-' + str(epoch) + '.tp'
        return torch.load(name, map_location=self.device)
