"""Script containing the fcnet variant of the DDPGfD agent."""
import torch
import os
import pickle

from ddpgfd.agents.base import DDPGfDAgent
from ddpgfd.agents.base import DATA_DEMO
from ddpgfd.core.training_utils import EWC


class FeedForwardAgent(DDPGfDAgent):
    """Feedforward neural network DDPGfD agent.

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
    expert_size : int
        number of samples from an expert controller
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

        # =================================================================== #
        #                          Policy Components                          #
        # =================================================================== #

        (actor_b, actor_t, critic_b, critic_t, memory, optimizer_actor,
         optimizer_critic, action_noise) = self._create_level(
            state_dim=state_dim,
            action_dim=action_dim)

        self.actor_b = actor_b
        self.actor_t = actor_t
        self.critic_b = critic_b
        self.critic_t = critic_t
        self.memory = memory
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.action_noise = action_noise

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

    def demo2memory(self, demo_dir, optimal):
        """See parent class."""
        filenames = [x for x in os.listdir(demo_dir) if x.endswith(".pkl")]

        for ix, f_idx in enumerate(filenames):
            fname = os.path.join(demo_dir, f_idx)
            with open(fname, 'rb') as f:
                data = pickle.load(f)
            for i in range(len(data)):
                # Extract demonstration.
                s, a, r, s2 = data[i]

                # Add one-step to memory.
                self.memory.add((
                    self.obs2tensor(s),
                    self.obs2tensor(a),
                    torch.tensor([r]).float(),
                    self.obs2tensor(s2),
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

    def save(self, progress_path, epoch):
        """See parent class."""
        self._save_model_weight(
            self.actor_b, progress_path, epoch, prefix='actor_b')
        self._save_model_weight(
            self.actor_t, progress_path, epoch, prefix='actor_t')
        self._save_model_weight(
            self.critic_b, progress_path, epoch, prefix='critic_b')
        self._save_model_weight(
            self.critic_t, progress_path, epoch, prefix='critic_t')

    def load(self, progress_path, epoch):
        """See parent class."""
        self.actor_b.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='actor_b'))
        self.actor_t.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='actor_t'))
        self.critic_b.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='critic_b'))
        self.critic_t.load_state_dict(self._restore_model_weight(
            progress_path, epoch, prefix='critic_t'))

    def reset(self):
        """See parent class."""
        self.action_noise.reset()

    def get_action(self, s):
        """See parent class."""
        n_agents = len(s)

        # Compute noisy actions by the policy.
        action = [
            torch.clip(
                self.actor_b(self.obs2tensor(s[i]).to(self.device)[None])[0] +
                torch.from_numpy(self.action_noise()).float(),
                min=-0.99, max=0.99,
            ) for i in range(n_agents)]

        return [act.numpy() for act in action]

    def add_memory(self, s, a, s2, r, gamma, dtype):
        """See parent class"""
        n_agents = len(s)
        for i in range(n_agents):
            self.memory.add((
                self.obs2tensor(s[i]),
                self.obs2tensor(a[i]),
                torch.tensor([r[i]]).float(),
                self.obs2tensor(s2[i]),
                torch.tensor([gamma]),
                dtype))

    def update_agent(self, update_step):
        """See parent class."""
        return self._update_agent_util(
            update_step=update_step,
            expert_size=self.expert_size,
            memory=self.memory,
            actor_b=self.actor_b,
            actor_t=self.actor_t,
            critic_b=self.critic_b,
            critic_t=self.critic_t,
            optimizer_actor=self.optimizer_actor,
            optimizer_critic=self.optimizer_critic,
        )
