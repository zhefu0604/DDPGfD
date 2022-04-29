"""Script containing the base DDPGfD agent class."""
import torch
import torch.nn as nn
import numpy as np
import os
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
    """DDPG from Demonstration agent class.

    See: https://arxiv.org/pdf/1707.08817.pdf

    Attributes
    ----------
    conf : Any
        full configuration parameters
    agent_conf : Any
        agent configuration parameters
    device : torch.device
        context-manager that changes the selected device.
    state_dim : int
        number of elements in the state space
    action_dim : int
        number of elements in the action space
    logger : object
        an object used for logging purposes
    q_criterion : nn._Loss
        loss function used for Q-value error
    il_criterion : nn._Loss
        loss function used for imitation learning component
    ewc : ddpgfd.core.training_utils.EWC or None
        Elastic Weight Consolidation component
    """

    def __init__(self, conf, device, state_dim, action_dim):
        """Instantiate the agent class.

        Parameters
        ----------
        conf : Any
            full configuration parameters
        device : torch.device
            context-manager that changes the selected device.
        state_dim : int
            number of elements in the state space
        action_dim : int
            number of elements in the action space
        """
        super(DDPGfDAgent, self).__init__()

        self.conf = conf
        self.agent_conf = self.conf.agent_config
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.logger = logging.getLogger('DDPGfD')

        # Q-function loss
        reduction = 'none'
        if self.conf.train_config.mse_loss:
            self.q_criterion = nn.MSELoss(reduction=reduction)
        else:
            self.q_criterion = nn.SmoothL1Loss(reduction=reduction)

        # imitation loss
        self.il_criterion = nn.MSELoss(reduction=reduction)

        # Elastic Weight Consolidation component
        self.ewc = None

    def demo2memory(self, demo_dir, optimal):
        """Import demonstration from pkl files to the replay buffer."""
        raise NotImplementedError

    def save(self, progress_path, epoch):
        """Save all model parameters to a given path.

        Parameters
        ----------
        progress_path : str
            the path to the directory where all logs are stored
        epoch : int
            the current training epoch
        """
        raise NotImplementedError

    def load(self, progress_path, epoch):
        """Load all model parameters from a given path.

        Parameters
        ----------
        progress_path : str
            the path to the directory where all logs are stored
        epoch : int
            the training epoch to collect model parameters from
        """
        raise NotImplementedError

    def reset(self):
        """Reset action noise."""
        raise NotImplementedError

    def get_action(self, s):
        """Compute noisy action."""
        raise NotImplementedError

    def add_memory(self, s, a, s2, r, gamma, dtype):
        """Add a sample to the replay buffer(s).

        Parameters
        ----------
        s : list of array_like
            list of observations for each agent
        a : list of array_like
            list of actions for each agent
        s2 : list of array_like
            list of next observations for each agent
        r : list of float
            list of rewards for each agent
        gamma : float
            reward discount
        dtype : int
            0: collected from a demonstration,
            1: collected during training
        """
        raise NotImplementedError

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
        raise NotImplementedError

    # ======================================================================= #
    #                             Utility Methods                             #
    # ======================================================================= #

    def _update_agent_util(self,
                           update_step,
                           expert_size,
                           memory,
                           actor_b,
                           actor_t,
                           critic_b,
                           critic_t,
                           optimizer_actor,
                           optimizer_critic):
        """Perform a policy update.

        Parameters
        ----------
        update_step : int
            number of policy updates to perform
        expert_size : int
            number of samples from an expert controller
        memory : ddpgfd.core.replay_memory.PrioritizedReplayBuffer
            replay buffer object
        actor_b : ddpgfd.core.model.ActorNet
            base actor network
        actor_t : ddpgfd.core.model.ActorNet
            target actor network
        critic_b : ddpgfd.core.model.CriticNet
            base critic network
        critic_t : ddpgfd.core.model.CriticNet
            actor critic network
        optimizer_actor : torch.optim.Adam
            actor optimizer object
        optimizer_critic : torch.optim.Adam
            critic optimizer object

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
        if memory.ready():
            # Sample a batch of data.
            (batch_s, batch_a, batch_r, batch_s2, batch_gamma,
             batch_flags), weights, idxes = memory.sample(
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
                noise = (torch.randn_like(batch_a) *
                         self.agent_conf.policy_noise).clamp(
                    -self.agent_conf.noise_clip, self.agent_conf.noise_clip)
                next_action = (actor_t(batch_s2) + noise).clamp(-1, 1)

                # Compute the target Q value.
                target_q1, target_q2 = critic_t(
                    torch.cat((batch_s2, next_action), dim=1))
                target_q = torch.min(target_q1, target_q2)
                target_q = batch_r + batch_gamma * target_q

            # Get current Q estimates
            q1, q2 = critic_b(torch.cat((batch_s, batch_a), dim=1))

            # Compute critic loss.
            critic_loss = self.q_criterion(q1, target_q).mean() + \
                self.q_criterion(q2, target_q).mean()

            # Optimize the critic.
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
            losses_critic.append(critic_loss.item())

            # Record Demo count.
            d_flags = torch.from_numpy(batch_flags)
            demo_select = d_flags == DATA_DEMO
            n_act = demo_select.sum().item()
            demo_cnt.append(n_act)

            # Delayed policy updates
            if update_step % self.agent_conf.policy_freq == 0:
                # Compute actor losses.
                q_act = critic_b.Q1(
                    torch.cat((batch_s, actor_b(batch_s)), dim=1))
                actor_loss = -q_act.mean()

                # Add EWC loss.
                ewc_lambda = self.agent_conf.ewc_lambda
                if ewc_lambda > 0:
                    actor_loss += ewc_lambda * self.ewc.penalty(actor_b)

                il_lambda = self.agent_conf.il_lambda
                if il_lambda > 0:
                    # Sample expert states/actions.
                    (exprt_s, exprt_a, _, _, _, _), _, _ = memory.sample(
                        self.conf.train_config.batch_size,
                        cur_sz=expert_size)

                    # Add imitation loss.
                    actor_loss += il_lambda * self.il_criterion(
                        exprt_a, actor_b(exprt_s)).mean()

                # Optimize the actor.
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()
                losses_actor.append(actor_loss.item())

                # Update priorities in the replay buffer.
                if not self.agent_conf.no_per:
                    priority = ((q1.detach() - target_q).pow(2) +
                                q_act.detach().pow(2)).numpy().ravel() \
                        + self.agent_conf.const_min_priority
                    priority[batch_flags == DATA_DEMO] += \
                        self.agent_conf.const_demo_priority

                    memory.update_priorities(idxes, priority)

                # Update target.
                self.update_target(actor_b, actor_t)
                self.update_target(critic_b, critic_t)

        demo_n = max(sum(demo_cnt), 1e-10)

        return np.sum(losses_critic), np.sum(losses_actor), demo_n, batch_sz

    def _create_level(self, state_dim, action_dim):
        """Create the necessary components for a level of the policy."""
        # Create the actor base and target networks.
        actor_b = ActorNet(state_dim, action_dim, self.device)
        actor_t = ActorNet(state_dim, action_dim, self.device)

        # Create the critic base and target network.
        critic_b = CriticNet(state_dim, action_dim, self.device)
        critic_t = CriticNet(state_dim, action_dim, self.device)

        # Initialize target policy parameters.
        self.init_target(actor_b, actor_t)
        self.init_target(critic_b, critic_t)

        # Create the replay buffer.
        memory = PrioritizedReplayBuffer(
            size=self.agent_conf.replay_buffer_size,
            seed=self.conf.train_config.seed)

        # Crate optimizer.
        optimizer_actor, optimizer_critic = self._set_optimizer(
            actor_b=actor_b, critic_b=critic_b)

        # exploration noise
        action_noise = GaussianActionNoise(
            std=self.agent_conf.action_noise_std, ac_dim=action_dim)

        return (actor_b, actor_t, critic_b, critic_t, memory, optimizer_actor,
                optimizer_critic, action_noise)

    def _set_optimizer(self, actor_b, critic_b):
        """Create the optimizer objects."""
        # Create actor optimizer.
        optimizer_actor = torch.optim.Adam(
            actor_b.parameters(),
            lr=self.conf.train_config.lr_rate,
            weight_decay=self.conf.train_config.w_decay)

        # Create critic optimizer.
        optimizer_critic = torch.optim.Adam(
            critic_b.parameters(),
            lr=self.conf.train_config.lr_rate)

        return optimizer_actor, optimizer_critic

    @staticmethod
    def _save_model_weight(model, progress_path, epoch, prefix=''):
        """Save a model's parameters is a specified path.

        Parameters
        ----------
        model : torch.nn.Module
            the model whose weights should be saved
        epoch : int
            the current training epoch
        prefix : str
            an auxiliary string specifying the type of model
        """
        name = progress_path + prefix + 'model-' + str(epoch) + '.tp'
        torch.save(model.state_dict(), name)

    @staticmethod
    def _restore_model_weight(progress_path, epoch, prefix=''):
        """Restore a model's parameters from a specified path.

        Parameters
        ----------
        epoch : int
            the training epoch to collect model parameters from
        prefix : str
            an auxiliary string specifying the type of model
        """
        name = os.path.join(
            progress_path,
            "model-params",
            prefix + 'model-' + str(epoch) + '.tp')
        return torch.load(name)

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

    @staticmethod
    def obs2tensor(state):
        """Convert observations to a torch-compatible format."""
        return torch.from_numpy(state).float()
