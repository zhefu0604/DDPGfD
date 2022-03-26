"""Script containing the DDPGfD agent class."""
import numpy as np
import torch
import torch.nn as nn
import logging
import os

from ddpgfd.core.model import ActorNet
from ddpgfd.core.model import CriticNet
from ddpgfd.core.replay_memory import PrioritizedReplayBuffer
from ddpgfd.core.training_utils import EWC
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
    actor_b : ddpgfd.core.model.ActorNet
        base actor network
    actor_t : ddpgfd.core.model.ActorNet
        target actor network
    critic_b : ddpgfd.core.model.CriticNet
        base critic network
    critic_t : ddpgfd.core.model.CriticNet
        target critic network
    memory : ddpgfd.core.replay_memory.PrioritizedReplayBuffer
        replay buffer object
    optimizer_actor : torch.optim.Adam
        an optimizer object for the actor
    optimizer_critic : torch.optim.Adam
        an optimizer object for the critic
    action_noise : ddpgfd.core.training_utils.ActionNoise
        Gaussian action noise object, for exploration purposes
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

        # Create the actor base and target networks.
        self.actor_b = ActorNet(state_dim, action_dim, device)
        self.actor_t = ActorNet(state_dim, action_dim, device)

        # Create the critic base and target network.
        self.critic_b = CriticNet(state_dim, action_dim, device)
        self.critic_t = CriticNet(state_dim, action_dim, device)

        # Set random seed.
        self.rs = np.random.RandomState(self.conf.train_config.seed)

        # Initialize target policy parameters.
        self.init_target(self.actor_b, self.actor_t)
        self.init_target(self.critic_b, self.critic_t)

        # Create the replay buffer.
        self.memory = PrioritizedReplayBuffer(
            size=self.agent_conf.replay_buffer_size,
            seed=self.conf.train_config.seed,
            alpha=0.3,
            beta_init=1.0,
            beta_inc_n=100,
        )

        # Crate optimizer.
        self.optimizer_actor = None
        self.optimizer_critic = None
        self._set_optimizer()

        # Q-function loss
        reduction = 'none'
        if self.conf.train_config.mse_loss:
            self.q_criterion = nn.MSELoss(reduction=reduction)
        else:
            self.q_criterion = nn.SmoothL1Loss(reduction=reduction)

        # imitation loss
        self.il_criterion = nn.MSELoss(reduction=reduction)

        # exploration noise
        self.action_noise = GaussianActionNoise(
            std=self.agent_conf.action_noise_std, ac_dim=action_dim)

        # Elastic Weight Consolidation component
        self.ewc = None

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

    def initialize(self):
        """Initialize the agent.

        This method performs the following tasks:

        1. It loads initial demonstration data from a predefined path.
        2. It loads the weights/biases of the agent from a predefined path.
        3. It initializes the EWC component if weights are to be adjusted to
           minimize catastrophic forgetting.
        4. It pretrains the initial policy for a number of steps on any
           imported initial demonstration data.
        """
        # =================================================================== #
        #                       Load demonstration data                       #
        # =================================================================== #

        if self.conf.demo_config.load_demo_data:
            pass  # TODO

        # =================================================================== #
        #                         Load initial policy                         #
        # =================================================================== #

        if self.agent_conf.pretrain_path != "":
            progress_path = self.agent_conf.pretrain_path
            epoch = self.agent_conf.pretrain_epoch

            # Load initial actor policy parameters.
            self.actor_b.load_state_dict(self._restore_model_weight(
                progress_path, epoch, prefix='actor_b'))
            self.actor_t.load_state_dict(self._restore_model_weight(
                progress_path, epoch, prefix='actor_t'))

            self.logger.info(
                'Loaded policy from progress path {} and epoch {}.'.format(
                    self.agent_conf.pretrain_path,
                    self.agent_conf.pretrain_epoch))

        # =================================================================== #
        #                    Elastic Weight Consolidation                     #
        # =================================================================== #

        if self.agent_conf.ewc_lambda > 0:
            # Make sure demonstration data and an initial policy were loaded.
            assert self.conf.demo_config.load_demo_data
            assert self.agent_conf.pretrain_path != ""

            # Create initial dataset of states.
            dataset = self.memory.get_states()

            # Initialize Elastic Weight Consolidation component.
            self.ewc = EWC(model=self.actor_b, dataset=dataset)

            self.logger.info('Create EWC with lambda: {}'.format(
                self.agent_conf.ewc_lambda))

        # =================================================================== #
        #                    Pretraining on demonstrations                    #
        # =================================================================== #

        if self.conf.train_config.pretrain_step > 0:
            # Make sure demonstration data was loaded.
            assert self.conf.demo_config.load_demo_data

            # Set the agent in training mode.
            self.agent.train()

            # Perform training on demonstration data.
            for t in range(self.conf.train_config.pretrain_step):
                self.agent.update_agent(t)

            self.logger.info('Pretrained policy for {} steps.'.format(
                self.conf.train_config.pretrain_step))

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

    def update_agent(self, update_step, expert_size):
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
                noise = (torch.randn_like(batch_a) *
                         self.agent_conf.policy_noise).clamp(
                    -self.agent_conf.noise_clip, self.agent_conf.noise_clip)
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
            if update_step % self.agent_conf.policy_freq == 0:
                # Compute actor losses.
                q_act = self.critic_b.Q1(
                    torch.cat((batch_s, self.actor_b(batch_s)), dim=1))
                actor_loss = -q_act.mean()

                # Add EWC loss.
                ewc_lambda = self.agent_conf.ewc_lambda
                if ewc_lambda > 0:
                    actor_loss += ewc_lambda * self.ewc.penalty(self.actor_b)

                # il_lambda = self.agent_conf.il_lambda
                il_lambda = 0
                if il_lambda > 0:
                    # Sample expert states/actions.
                    (exprt_s, exprt_a, _, _, _, _), _, _ = self.memory.sample(
                        self.conf.train_config.batch_size,
                        cur_sz=expert_size)

                    # Add imitation loss.
                    actor_loss = il_lambda * self.il_criterion(
                        exprt_a, self.actor_b(exprt_s)).mean()

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
        """Save all model parameters to a given path.

        Parameters
        ----------
        progress_path : str
            the path to the directory where all logs are stored
        epoch : int
            the current training epoch
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
        """Load all model parameters from a given path.

        Parameters
        ----------
        progress_path : str
            the path to the directory where all logs are stored
        epoch : int
            the training epoch to collect model parameters from
        """
        self.actor_b.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='actor_b'))
        self.actor_t.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='actor_t'))
        self.critic_b.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='critic_b'))
        self.critic_t.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='critic_t'))

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

    def _restore_model_weight(self, progress_path, epoch, prefix=''):
        """Restor a model's parameters from a specified path.

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
        print(name)
        return torch.load(name)
