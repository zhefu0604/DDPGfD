"""Script containing the fcnet variant of the DDPGfD agent."""
import numpy as np
import torch
import torch.nn as nn
import os
import pickle

from ddpgfd.agents.base import DDPGfDAgent
from ddpgfd.agents.base import DATA_DEMO
from ddpgfd.core.model import ActorNet
from ddpgfd.core.model import CriticNet
from ddpgfd.core.replay_memory import PrioritizedReplayBuffer
from ddpgfd.core.training_utils import EWC
from ddpgfd.core.training_utils import GaussianActionNoise


class FeedForwardAgent(DDPGfDAgent):
    """DDPG from Demonstration agent class.

    See: https://arxiv.org/pdf/1707.08817.pdf

    Attributes
    ----------
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
        super(FeedForwardAgent, self).__init__(
            conf=conf,
            device=device,
            state_dim=state_dim,
            action_dim=action_dim,
        )

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
        self.expert_size = None
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
            lr=self.conf.train_config.lr_rate)

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

        dconf = self.conf.demo_config
        if dconf.load_demo_data:
            self.expert_size = self.demo2memory(
                dconf.demo_dir, optimal=not dconf.random)

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
            self.train()

            # Perform training on demonstration data.
            for t in range(self.conf.train_config.pretrain_step):
                self.update_agent(t)

            self.logger.info('Pretrained policy for {} steps.'.format(
                self.conf.train_config.pretrain_step))

    def reset(self):
        """See parent class."""
        self.action_noise.reset()

    def get_action(self, s_tensor, n_agents):
        """See parent class."""
        # Compute noisy actions by the policy.
        action = [
            torch.clip(
                self.actor_b(s_tensor[i].to(self.device)[None])[0] +
                torch.from_numpy(self.action_noise()).float(),
                min=-0.99, max=0.99,
            ) for i in range(n_agents)]

        return [act.numpy() for act in action]

    def add_memory(self, s, a, s2, r, gamma, dtype):
        """See parent class"""
        self.memory.add((s, a, r, s2, gamma, dtype))

    def demo2memory(self, demo_dir, optimal):
        """Import demonstration from pkl files to the replay buffer."""
        filenames = [x for x in os.listdir(demo_dir) if x.endswith(".pkl")]

        for ix, f_idx in enumerate(filenames):
            fname = os.path.join(demo_dir, f_idx)
            with open(fname, 'rb') as f:
                data = pickle.load(f)
            for i in range(len(data)):
                # TODO
                if i % 5 != 0:
                    continue

                # Extract demonstration.
                s, a, r, s2 = data[i]

                # Convert to be pytorch compatible.
                s_tensor = torch.from_numpy(s).float()
                s2_tensor = torch.from_numpy(s2).float()
                action = torch.from_numpy(a).float()

                # Add one-step to memory.
                self.memory.add((
                    s_tensor,
                    action,
                    torch.tensor([r]).float(),
                    s2_tensor,
                    torch.tensor([self.agent_conf.gamma]),
                    DATA_DEMO))

            self.logger.info(
                '{} Demo Trajectories Loaded. Total Experience={}'.format(
                    ix + 1, len(self.memory)))

        # Prevent demonstrations from being deleted.
        if optimal:
            expert_size = len(self.memory)
            self.memory.set_protect_size(expert_size)
        else:
            expert_size = 0

        return expert_size

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
        """See parent class."""
        losses_critic = []
        losses_actor = []
        demo_cnt = []
        batch_sz = 0
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
                target_q = batch_r + batch_gamma * target_q

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

                il_lambda = self.agent_conf.il_lambda
                if il_lambda > 0:
                    # Sample expert states/actions.
                    (exprt_s, exprt_a, _, _, _, _), _, _ = self.memory.sample(
                        self.conf.train_config.batch_size,
                        cur_sz=self.expert_size)

                    # Add imitation loss.
                    actor_loss += il_lambda * self.il_criterion(
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
