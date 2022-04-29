"""Script containing the hierarchical variant of the DDPGfD agent."""
import random
import numpy as np
import torch
import os
import pickle
from copy import deepcopy

from ddpgfd.agents.base import DDPGfDAgent
from ddpgfd.agents.base import DATA_DEMO


class HierarchicalAgent(DDPGfDAgent):
    """Hierarchical neural network DDPGfD agent.

    Attributes
    ----------
    meta_actor_b : ddpgfd.core.model.ActorNet
        meta-policy base actor network
    meta_actor_t : ddpgfd.core.model.ActorNet
        meta-policy target actor network
    meta_critic_b : ddpgfd.core.model.CriticNet
        meta-policy base critic network
    meta_critic_t : ddpgfd.core.model.CriticNet
        meta-policy target critic network
    meta_memory : ddpgfd.core.replay_memory.PrioritizedReplayBuffer
        meta-policy replay buffer object
    meta_optimizer_actor : torch.optim.Adam
        meta-policy actor optimizer
    meta_optimizer_critic : torch.optim.Adam
        meta-policy critic optimizer
    meta_action_noise : ddpgfd.core.training_utils.ActionNoise
        meta-policy action noise object, for exploration purposes
    worker_actor_b : ddpgfd.core.model.ActorNet
        worker policy base actor network
    worker_actor_t : ddpgfd.core.model.ActorNet
        worker policy target actor network
    worker_critic_b : ddpgfd.core.model.CriticNet
        worker policy base critic network
    worker_critic_t : ddpgfd.core.model.CriticNet
        worker policy target critic network
    worker_memory : ddpgfd.core.replay_memory.PrioritizedReplayBuffer
        worker policy replay buffer object
    worker_optimizer_actor : torch.optim.Adam
        worker policy actor optimizer
    worker_optimizer_critic : torch.optim.Adam
        worker policy critic optimizer
    worker_action_noise : ddpgfd.core.training_utils.ActionNoise
        worker policy action noise object, for exploration purposes
    expert_size : int
        number of samples from an expert controller
    """

    def __init__(self, conf, device, state_dim, action_dim):
        """See parent class."""
        super(HierarchicalAgent, self).__init__(
            conf=conf,
            device=device,
            state_dim=state_dim,
            action_dim=action_dim,
        )

        meta_action_dim = 1  # TODO
        self.meta_period = 10  # TODO
        self._meta_reward = 0.
        self._meta_obs = None
        self._meta_goal = None

        # time since the rollout started. Used to track when to recompute the
        # desired goal.
        self._t = 0

        # the desired goal in the current timestep
        self._current_goal = None

        # =================================================================== #
        #                          Policy Components                          #
        # =================================================================== #

        # Create meta policy.
        (actor_b, actor_t, critic_b, critic_t, memory, optimizer_actor,
         optimizer_critic, action_noise) = self._create_level(
            state_dim=state_dim,
            action_dim=meta_action_dim)

        self.meta_actor_b = actor_b
        self.meta_actor_t = actor_t
        self.meta_critic_b = critic_b
        self.meta_critic_t = critic_t
        self.meta_memory = memory
        self.meta_optimizer_actor = optimizer_actor
        self.meta_optimizer_critic = optimizer_critic
        self.meta_action_noise = action_noise

        # Create worker policy.
        (actor_b, actor_t, critic_b, critic_t, memory, optimizer_actor,
         optimizer_critic, action_noise) = self._create_level(
            state_dim=state_dim + meta_action_dim,
            action_dim=action_dim)

        self.worker_actor_b = actor_b
        self.worker_actor_t = actor_t
        self.worker_critic_b = critic_b
        self.worker_critic_t = critic_t
        self.worker_memory = memory
        self.worker_optimizer_actor = optimizer_actor
        self.worker_optimizer_critic = optimizer_critic
        self.worker_action_noise = action_noise

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
            self.meta_actor_b.load_state_dict(self._restore_model_weight(
                progress_path, epoch, prefix='meta_actor_b'))
            self.meta_actor_t.load_state_dict(self._restore_model_weight(
                progress_path, epoch, prefix='meta_actor_t'))
            self.worker_actor_b.load_state_dict(self._restore_model_weight(
                progress_path, epoch, prefix='worker_actor_b'))
            self.worker_actor_t.load_state_dict(self._restore_model_weight(
                progress_path, epoch, prefix='worker_actor_t'))

            self.logger.info(
                'Loaded policy from progress path {} and epoch {}.'.format(
                    self.agent_conf.pretrain_path,
                    self.agent_conf.pretrain_epoch))

        # =================================================================== #
        #                    Elastic Weight Consolidation                     #
        # =================================================================== #

        if self.agent_conf.ewc_lambda > 0:
            pass  # TODO

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

    def demo2memory(self, demo_dir, optimal):
        """See parent class."""
        filenames = [x for x in os.listdir(demo_dir) if x.endswith(".pkl")]

        for ix, f_idx in enumerate(filenames):
            fname = os.path.join(demo_dir, f_idx)
            with open(fname, 'rb') as f:
                data = pickle.load(f)

            meta_goal = []
            meta_obs = []
            meta_reward = 0
            for i in range(len(data)):
                # Extract demonstration.
                s, a, r, s2 = data[i]

                if i % self.meta_period == 0:
                    # Add one-step to memory for the meta policy.
                    if i > 0:
                        self.meta_memory.add((
                            self.obs2tensor(meta_obs),
                            self.obs2tensor(meta_goal),
                            torch.tensor([meta_reward]).float(),
                            self.obs2tensor(s),
                            torch.tensor([self.full_conf.agent_config.gamma]),
                            DATA_DEMO))

                    # Update previous observations and cumulative reward.
                    meta_goal = random.uniform(-1, 1)  # TODO: hindsight
                    meta_obs = deepcopy(s)
                    meta_reward = deepcopy(r)
                else:
                    meta_reward += r

                # Compute intrinsic reward.
                n_steps = 5  # TODO
                desired_speed = 0.5 * meta_goal + 0.5
                actual_speed = s2[n_steps]
                r_worker = - (desired_speed - actual_speed) ** 2

                # Concatenate state and goal.
                s1_worker = np.array(list(s) + list(meta_goal))
                s2_worker = np.array(list(s2) + list(meta_goal))

                # Add one-step to memory for the worker policy.
                self.worker_memory.add((
                    self.obs2tensor(s1_worker),
                    self.obs2tensor(a),
                    torch.tensor([r_worker]).float(),
                    self.obs2tensor(s2_worker),
                    torch.tensor([self.full_conf.agent_config.gamma]),
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

    def save(self, progress_path, epoch):
        """See parent class."""
        # Save meta policy parameters.
        self._save_model_weight(
            self.meta_actor_b, progress_path, epoch, prefix='meta_actor_b')
        self._save_model_weight(
            self.meta_actor_t, progress_path, epoch, prefix='meta_actor_t')
        self._save_model_weight(
            self.meta_critic_b, progress_path, epoch, prefix='meta_critic_b')
        self._save_model_weight(
            self.meta_critic_t, progress_path, epoch, prefix='meta_critic_t')

        # Save worker policy parameters.
        self._save_model_weight(
            self.worker_actor_b, progress_path, epoch,
            prefix='worker_actor_b')
        self._save_model_weight(
            self.worker_actor_t, progress_path, epoch,
            prefix='worker_actor_t')
        self._save_model_weight(
            self.worker_critic_b, progress_path, epoch,
            prefix='worker_critic_b')
        self._save_model_weight(
            self.worker_critic_t, progress_path, epoch,
            prefix='worker_critic_t')

    def load(self, progress_path, epoch):
        """See parent class."""
        # Load meta policy parameters.
        self.meta_actor_b.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='meta_actor_b'))
        self.meta_actor_t.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='meta_actor_t'))
        self.meta_critic_b.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='meta_critic_b'))
        self.meta_critic_t.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='meta_critic_t'))

        # Load worker policy parameters.
        self.worker_actor_b.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='worker_actor_b'))
        self.worker_actor_t.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='worker_actor_t'))
        self.worker_critic_b.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='worker_critic_b'))
        self.worker_critic_t.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='worker_critic_t'))

    def reset(self):
        """See parent class."""
        self._t = 0
        self._current_goal = None
        self._meta_reward = 0.
        self._meta_obs = None
        self._meta_goal = None
        self.meta_action_noise.reset()
        self.worker_action_noise.reset()

    def get_action(self, s):
        """See parent class."""
        n_agents = len(s)

        # Compute desired goal.
        if self._t % self.meta_period == 0:
            meta_action = [
                torch.clip(
                    self.meta_actor_b(self.obs2tensor(s[i]).to(
                        self.device)[None])[0] +
                    torch.from_numpy(self.meta_action_noise()).float(),
                    min=-0.99, max=0.99,
                ) for i in range(n_agents)]

            self._current_goal = [act.numpy() for act in meta_action]

        # Compute desired primitive action.
        s_worker = [
            np.array(list(s[i]) + list(self._current_goal[i]))
            for i in range(len(s))]

        action = [
            torch.clip(
                self.worker_actor_b(self.obs2tensor(s_worker[i]).to(
                    self.device)[None])[0] +
                torch.from_numpy(self.worker_action_noise()).float(),
                min=-0.99, max=0.99,
            ) for i in range(n_agents)]

        return action

    def add_memory(self, s, a, s2, r, gamma, dtype):
        """See parent class."""
        n_agents = len(s)

        if self.t % self.meta_period == 0:
            # Add memory to meta policy.
            if self._t > 0:
                for i in range(n_agents):
                    self.memory.add((
                        self.obs2tensor(self._meta_obs[i]),
                        self.obs2tensor(self._meta_goal[i]),
                        torch.tensor([self._meta_reward[i]]).float(),
                        self.obs2tensor(s[i]),
                        torch.tensor([gamma]),
                        dtype))

            # Update previous observations and cumulative reward.
            self._meta_obs = deepcopy(s)
            self._meta_goal = deepcopy(self._current_goal)
            self._meta_reward = deepcopy(r)
        else:
            for i in range(n_agents):
                self._meta_reward[i] += r[i]

        for i in range(n_agents):
            # Compute intrinsic reward.
            n_steps = 5  # TODO
            desired_speed = 0.5 * self._current_goal[i] + 0.5
            actual_speed = s2[n_steps]
            r_worker = - (desired_speed - actual_speed) ** 2

            # Concatenate state and goal.
            s1_worker = np.array(list(s[i]) + list(self._current_goal[i]))
            s2_worker = np.array(list(s2[i]) + list(self._current_goal[i]))

            # Add memory to worker policy.
            self.memory.add((
                self.obs2tensor(s1_worker),
                self.obs2tensor(a[i]),
                torch.tensor([r_worker]).float(),
                self.obs2tensor(s2_worker),
                torch.tensor([gamma]),
                dtype))

        self.t += 1

    def update_agent(self, update_step):
        """See parent class."""
        critic_loss = []
        actor_loss = []
        demo_n = []
        batch_sz = []

        # Update the policy parameters of the meta policy.
        if update_step % self.meta_period == 0:
            c, a, d, b = self._update_agent_util(
                update_step=update_step // self.meta_period,
                expert_size=self.expert_size // self.meta_period,
                memory=self.meta_memory,
                actor_b=self.meta_actor_b,
                actor_t=self.meta_actor_t,
                critic_b=self.meta_critic_b,
                critic_t=self.meta_critic_t,
                optimizer_actor=self.meta_optimizer_actor,
                optimizer_critic=self.meta_optimizer_critic,
            )
            critic_loss.append(c)
            actor_loss.append(a)
            demo_n.append(d)
            batch_sz.append(b)

        # Update the policy parameters of the worker policy.
        c, a, d, b = self._update_agent_util(
            update_step=update_step,
            expert_size=self.expert_size,
            memory=self.worker_memory,
            actor_b=self.worker_actor_b,
            actor_t=self.worker_actor_t,
            critic_b=self.worker_critic_b,
            critic_t=self.worker_critic_t,
            optimizer_actor=self.worker_optimizer_actor,
            optimizer_critic=self.worker_optimizer_critic,
        )
        critic_loss.append(c)
        actor_loss.append(a)
        demo_n.append(d)
        batch_sz.append(b)

        return critic_loss, actor_loss, demo_n, batch_sz
