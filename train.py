import os
import time
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
import logging
import gym
import pickle
import trajectory.config as config
from training_utils import TrainingProgress, timeSince, load_conf
from agent import DDPGfDAgent, DATA_RUNTIME, DATA_DEMO
from logger import logger_setup
from os.path import join as opj
from gym.envs.registration import register

np.set_printoptions(suppress=True, precision=4)

# Used loggers
DEBUG_LLV = 5  # for masked
loggers = ['RLTrainer', 'DDPGfD', 'TP']
# logging.addLevelName(DEBUG_LLV, 'DEBUGLLV')  # Lower level debugging info
logging_level = logging.DEBUG  # logging.DEBUG


class OrnsteinUhlenbeckActionNoise:
    """TODO."""

    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        """TODO.

        Parameters
        ----------
        mu : TODO
            TODO
        sigma : TODO
            TODO
        theta : TODO
            TODO
        dt : TODO
            TODO
        x0 : TODO
            TODO
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = None
        self.reset()

    def __call__(self):
        """TODO."""
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        """TODO."""
        self.x_prev = self.x0 or np.zeros_like(self.mu)

    def __repr__(self):
        """TODO."""
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self.mu, self.sigma)


class RLTrainer:
    """TODO."""

    def __init__(self, conf_path, evaluate=False):
        """Instantiate the RL algorithm.

        Parameters
        ----------
        conf_path : TODO
            TODO
        evaluate : bool
            TODO
        """
        self.full_conf = load_conf(conf_path)
        self.conf = self.full_conf.train_config

        progress_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'progress')
        result_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'result')

        # Store in xxxx_dir/exp_name+exp_idx/...
        self.tp = TrainingProgress(
            progress_dir, result_dir, self.conf.exp_name)

        logger_setup(
            os.path.join(self.tp.result_path, self.conf.exp_name + '-log.txt'),
            loggers, logging_level)
        self.logger = logging.getLogger('RLTrainer')

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        if self.conf.seed == -1:
            self.conf.seed = os.getpid() + \
                int.from_bytes(os.urandom(4), byteorder="little") >> 1
            self.logger.info('Random Seed={}'.format(self.conf.seed))

        # Random seed
        torch.manual_seed(self.conf.seed)  # cpu
        np.random.seed(self.conf.seed)  # numpy

        # Backup environment config
        if not evaluate:
            self.tp.backup_file(conf_path, 'training.yaml')

        # Construct Env
        self.all_envs = self._create_envs()
        self.env = random.sample(self.all_envs, 1)[0]

        self.logger.info('Environment Loaded')

        self.agent = DDPGfDAgent(self.full_conf.agent_config, self.device)
        self.agent.to(self.device)

        if self.conf.restore:
            self.restore_progress()
        else:
            self.episode = 1

        self.optimizer_actor = None
        self.optimizer_critic = None
        self.set_optimizer()

        # Loss Function setting
        reduction = 'none'
        if self.conf.mse_loss:
            self.q_criterion = nn.MSELoss(reduction=reduction)
        else:
            self.q_criterion = nn.SmoothL1Loss(reduction=reduction)
        self.demo2memory()
        self.action_noise = OrnsteinUhlenbeckActionNoise(
            np.zeros(self.full_conf.agent_config.action_dim),
            self.full_conf.agent_config.action_noise_std)

    @staticmethod
    def _create_envs():
        """TODO."""
        filenames = [
            "2021-03-08-22-35-14_2T3MWRFVXLW056972_masterArray_0_4597.csv",
            "2021-03-08-22-35-14_2T3MWRFVXLW056972_masterArray_1_4927.csv",
            "2021-03-09-13-35-04_2T3MWRFVXLW056972_masterArray_0_6825.csv",
            "2021-03-09-13-35-04_2T3MWRFVXLW056972_masterArray_1_4938.csv",
            "2021-03-10-21-54-16_2T3MWRFVXLW056972_masterArray_0_4523.csv",
            "2021-03-10-21-54-16_2T3MWRFVXLW056972_masterArray_1_4582.csv",
            "2021-03-12-22-20-57_2T3MWRFVXLW056972_masterArray_0_5672.csv",
            "2021-03-12-22-20-57_2T3MWRFVXLW056972_masterArray_1_4817.csv",
            "2021-03-15-12-46-38_2T3MWRFVXLW056972_masterArray_0_4917.csv",
            "2021-03-15-12-46-38_2T3MWRFVXLW056972_masterArray_1_11342.csv",
            "2021-03-17-21-37-10_2T3MWRFVXLW056972_masterArray_0_4463.csv",
            "2021-03-17-21-37-10_2T3MWRFVXLW056972_masterArray_1_4386.csv",
            "2021-03-18-12-42-14_2T3MWRFVXLW056972_masterArray_0_3977.csv",
            "2021-03-18-12-42-14_2T3MWRFVXLW056972_masterArray_1_3918.csv",
            "2021-03-22-22-23-58_2T3MWRFVXLW056972_masterArray_0_4223.csv",
            "2021-03-22-22-23-58_2T3MWRFVXLW056972_masterArray_1_4422.csv",
            # Arwa said this trajectory might has sensor malfunction issue
            # "2021-03-23-21-50-02_2T3MWRFVXLW056972_masterArray_0_4331.csv",
            "2021-03-23-21-50-02_2T3MWRFVXLW056972_masterArray_1_3778.csv",
            # used for dashboard testing
            # "2021-03-24-12-39-15_2T3MWRFVXLW056972_masterArray_0_6438.csv",
            "2021-03-24-12-39-15_2T3MWRFVXLW056972_masterArray_1_4102.csv",
            "2021-03-24-21-34-31_2T3MWRFVXLW056972_masterArray_0_4937.csv",
            "2021-03-24-21-34-31_2T3MWRFVXLW056972_masterArray_1_4364.csv",
            "2021-03-26-21-26-45_2T3MWRFVXLW056972_masterArray_0_4540.csv",
            "2021-03-26-21-26-45_2T3MWRFVXLW056972_masterArray_1_4028.csv",
            "2021-03-29-12-47-15_2T3MWRFVXLW056972_masterArray_0_5016.csv",
            "2021-03-29-12-47-15_2T3MWRFVXLW056972_masterArray_1_4185.csv",
            "2021-03-31-21-39-05_2T3MWRFVXLW056972_masterArray_0_4200.csv",
            "2021-03-31-21-39-05_2T3MWRFVXLW056972_masterArray_1_4622.csv",
            "2021-04-02-21-31-47_2T3MWRFVXLW056972_masterArray_0_4125.csv",
            "2021-04-02-21-31-47_2T3MWRFVXLW056972_masterArray_1_4111.csv",
            "2021-04-05-21-39-05_2T3MWRFVXLW056972_masterArray_0_4357.csv",
            "2021-04-05-21-39-05_2T3MWRFVXLW056972_masterArray_1_4173.csv",
            "2021-04-06-21-18-22_2T3MWRFVXLW056972_masterArray_0_4427.csv",
            "2021-04-06-21-18-22_2T3MWRFVXLW056972_masterArray_1_4496.csv",
            "2021-04-07-12-33-03_2T3MWRFVXLW056972_masterArray_0_11294.csv",
            "2021-04-07-12-33-03_2T3MWRFVXLW056972_masterArray_1_6116.csv",
            "2021-04-07-21-22-07_2T3MWRFVXLW056972_masterArray_0_4101.csv",
            "2021-04-07-21-22-07_2T3MWRFVXLW056972_masterArray_1_4069.csv",
            "2021-04-12-21-34-57_2T3MWRFVXLW056972_masterArray_0_4796.csv",
            "2021-04-12-21-34-57_2T3MWRFVXLW056972_masterArray_1_4436.csv",
            "2021-04-15-21-32-46_2T3MWRFVXLW056972_masterArray_0_3889.csv",
            # used for dashboard testing
            # "2021-04-15-21-32-46_2T3MWRFVXLW056972_masterArray_1_3685.csv",
            "2021-04-16-12-34-41_2T3MWRFVXLW056972_masterArray_0_5778.csv",
            "2021-04-16-12-34-41_2T3MWRFVXLW056972_masterArray_1_4387.csv",
            "2021-04-19-12-27-33_2T3MWRFVXLW056972_masterArray_0_16467.csv",
            "2021-04-19-12-27-33_2T3MWRFVXLW056972_masterArray_1_6483.csv",
            "2021-04-19-22-09-19_2T3MWRFVXLW056972_masterArray_0_4433.csv",
            "2021-04-19-22-09-19_2T3MWRFVXLW056972_masterArray_1_4288.csv",
            "2021-04-20-21-42-34_2T3MWRFVXLW056972_masterArray_0_4025.csv",
            "2021-04-20-21-42-34_2T3MWRFVXLW056972_masterArray_1_3973.csv",
            "2021-04-21-21-45-12_2T3MWRFVXLW056972_masterArray_0_3957.csv",
            "2021-04-21-21-45-12_2T3MWRFVXLW056972_masterArray_1_3621.csv",
            # used for dashboard testing
            # "2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050.csv",
            "2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_1_5292.csv",
            "2021-04-26-21-13-18_2T3MWRFVXLW056972_masterArray_0_4595.csv",
            "2021-04-26-21-13-18_2T3MWRFVXLW056972_masterArray_1_4664.csv",
            "2021-04-27-21-37-32_2T3MWRFVXLW056972_masterArray_0_3836.csv",
            "2021-04-27-21-37-32_2T3MWRFVXLW056972_masterArray_1_3558.csv",
            "2021-04-29-21-16-14_2T3MWRFVXLW056972_masterArray_0_4190.csv",
            "2021-04-29-21-16-14_2T3MWRFVXLW056972_masterArray_1_4005.csv",
        ]

        all_envs = []
        for i, fp in enumerate(filenames):
            register(
                id="TrajectoryEnv-v{}".format(i),
                entry_point="trajectory.env.trajectory_env:TrajectoryEnv",
                kwargs={
                    "config": {
                        'horizon': 1000,
                        'min_headway': 7.0,
                        'max_headway': 120.0,
                        'whole_trajectory': True,
                        'discrete': False,
                        'num_actions': 7,
                        'use_fs': False,
                        'augment_vf': False,
                        'minimal_time_headway': 1.0,
                        'include_idm_mpg': False,
                        'num_concat_states': 1,
                        'num_steps_per_sim': 1,
                        'platoon': '2avs_4%',
                        'av_controller': 'rl',
                        'av_kwargs': '{}',
                        'human_controller': 'idm',
                        'human_kwargs': '{}',
                        'fixed_traj_path': os.path.join(
                            config.PROJECT_PATH,
                            'dataset/data_v2_preprocessed_west/{}'.format(fp)
                        ),
                        'lane_changing': False
                    },
                    "_simulate": True
                })

            # Make the gym environment
            all_envs.append(gym.make("TrajectoryEnv-v{}".format(i)))

        return all_envs

    def restore_progress(self):
        """TODO."""
        # TODO. tps only for restore process from conf
        self.tp.restore_progress(self.conf.tps)

        # TODO
        self.agent.actor_b.load_state_dict(self.tp.restore_model_weight(
                self.conf.tps, self.device, prefix='actor_b'))
        self.agent.actor_t.load_state_dict(self.tp.restore_model_weight(
            self.conf.tps, self.device, prefix='actor_t'))
        self.agent.critic_b.load_state_dict(self.tp.restore_model_weight(
            self.conf.tps, self.device, prefix='critic_b'))
        self.agent.critic_t.load_state_dict(self.tp.restore_model_weight(
            self.conf.tps, self.device, prefix='critic_t'))

        # TODO
        self.episode = self.tp.get_meta('saved_episode') + 1
        np.random.set_state(self.tp.get_meta('np_random_state'))
        torch.random.set_rng_state(self.tp.get_meta('torch_random_state'))

        self.logger.info('Restore Progress, Episode={}'.format(self.episode))

    def summary(self):
        """TODO."""
        # call Test/Evaluation here
        self.tp.add_meta({
            'saved_episode': self.episode,
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.random.get_rng_state()
        })

        # TODO
        self.save_progress(display=True)

    def save_progress(self, display=False):
        """TODO.

        Parameters
        ----------
        display : bool
            TODO
        """
        # TODO
        self.tp.save_model_weight(
            self.agent.actor_b, self.episode, prefix='actor_b')
        self.tp.save_model_weight(
            self.agent.actor_t, self.episode, prefix='actor_t')
        self.tp.save_model_weight(
            self.agent.critic_b, self.episode, prefix='critic_b')
        self.tp.save_model_weight(
            self.agent.critic_t, self.episode, prefix='critic_t')

        # TODO
        self.tp.save_progress(self.episode)
        self.tp.save_conf(self.conf.to_dict())

        # TODO
        if display:
            self.logger.info('Config name ' + self.conf.exp_name)
            self.logger.info('Progress Saved, current episode={}'.format(
                self.episode))

    def set_optimizer(self):
        """TODO."""
        # Create actor optimizer.
        self.optimizer_actor = torch.optim.Adam(
            self.agent.actor_b.parameters(),
            lr=self.conf.lr_rate,
            weight_decay=self.conf.w_decay)

        # Create critic optimizer.
        self.optimizer_critic = torch.optim.Adam(
            self.agent.critic_b.parameters(),
            lr=self.conf.lr_rate,
            weight_decay=self.conf.w_decay)

    def demo2memory(self):
        """TODO."""
        dconf = self.full_conf.demo_config
        if dconf.load_demo_data:
            filenames = [
                x for x in os.listdir(dconf.demo_dir) if x.endswith(".pkl")]
            for ix, f_idx in enumerate(filenames):
                fname = opj(dconf.demo_dir, f_idx)
                print(fname)
                with open(fname, 'rb') as f:
                    data = pickle.load(f)
                for i in range(len(data)):
                    for j in range(data[i]['state'].shape[0]):
                        s = data[i]['state'][j, :]
                        a = data[i]['action'][j, :]
                        r = data[i]['reward'][j, 0]
                        s2 = data[i]['next_state'][j, :]

                        s_tensor = torch.from_numpy(s).float()
                        s2_tensor = torch.from_numpy(s2).float()
                        action = torch.from_numpy(a).float()

                        # Add one-step to memory, last step added in pop with
                        # done=True
                        self.agent.memory.add((
                            s_tensor,
                            action,
                            torch.tensor([r]).float(),
                            s2_tensor,
                            torch.tensor([self.agent.conf.gamma]),
                            DATA_DEMO))

                self.logger.info(
                    '{} Demo Trajectories Loaded. Total Experience={}'.format(
                        ix + 1, len(self.agent.memory)))
            # self.agent.memory.set_protect_size(len(self.agent.memory))
        else:
            self.logger.info('No Demo Trajectory Loaded')

    def update_agent(self, update_step):  # update_step iteration
        """TODO.

        Parameters
        ----------
        update_step : TODO
            TODO

        Returns
        -------
        TODO
            TODO
        TODO
            TODO
        TODO
            TODO
        TODO
            TODO
        """
        # 2. Sample experience and update
        losses_critic = []
        losses_actor = []
        demo_cnt = []
        batch_sz = 0
        if self.agent.memory.ready():
            for _ in range(update_step):
                # TODO: remove gamma and flags
                (batch_s, batch_a, batch_r, batch_s2, batch_gamma,
                 batch_flags), weights, idxes = self.agent.memory.sample(
                    self.conf.batch_size)

                batch_s, batch_a, batch_r, batch_s2, batch_gamma, weights = \
                    batch_s.to(self.device), batch_a.to(self.device), \
                    batch_r.to(self.device), batch_s2.to(self.device), \
                    batch_gamma.to(self.device), \
                    torch.from_numpy(weights.reshape(-1, 1)).float().to(
                        self.device)

                batch_sz += batch_s.shape[0]
                with torch.no_grad():
                    action_tgt = self.agent.actor_t(batch_s)
                    y_tgt = batch_r + batch_gamma * self.agent.critic_t(
                        torch.cat((batch_s, action_tgt), dim=1))

                self.agent.zero_grad()
                # Critic loss
                self.optimizer_critic.zero_grad()
                Q_b = self.agent.critic_b(torch.cat((batch_s, batch_a), dim=1))
                loss_critic = (self.q_criterion(Q_b, y_tgt) * weights).mean()

                # Record Demo count
                d_flags = torch.from_numpy(batch_flags)
                demo_select = d_flags == DATA_DEMO
                N_act = demo_select.sum().item()
                demo_cnt.append(N_act)
                loss_critic.backward()
                self.optimizer_critic.step()

                # Actor loss
                self.optimizer_actor.zero_grad()
                action_b = self.agent.actor_b(batch_s)
                Q_act = self.agent.critic_b(
                    torch.cat((batch_s, action_b), dim=1))
                loss_actor = -torch.mean(Q_act)
                loss_actor.backward()
                self.optimizer_actor.step()

                priority = ((Q_b.detach() - y_tgt).pow(2) + Q_act.detach().pow(
                    2)).numpy().ravel() + self.agent.conf.const_min_priority
                priority[batch_flags == DATA_DEMO] += \
                    self.agent.conf.const_demo_priority

                if not self.agent.conf.no_per:
                    self.agent.memory.update_priorities(idxes, priority)

                losses_actor.append(loss_actor.item())
                losses_critic.append(loss_critic.item())

        if np.sum(demo_cnt) == 0:
            demo_n = 1e-10
        else:
            demo_n = np.sum(demo_cnt)

        return np.sum(losses_critic), np.sum(losses_actor), demo_n, batch_sz

    def pretrain(self):
        """Perform training on initial demonstration data."""
        assert self.full_conf.demo_config.load_demo_data
        self.agent.train()
        start_time = time.time()
        self.logger.info('Run Pretrain')
        for step in np.arange(self.conf.pretrain_save_step,
                              self.conf.pretrain_step + 1,
                              self.conf.pretrain_save_step):
            losses_critic, losses_actor, demo_n, batch_sz = self.update_agent(
                self.conf.pretrain_save_step)

            self.logger.info(
                '{}-Pretrain Step {}/{}, '
                '(Mean): actor_loss={:.8f}, '
                'critic_loss={:.8f}, '
                'batch_sz={}, '
                'Demo_ratio={:.8f}'.format(
                    timeSince(start_time),
                    step,
                    self.conf.pretrain_step,
                    losses_actor / batch_sz,
                    losses_critic / batch_sz, batch_sz,
                    demo_n / batch_sz)
            )

            self.tp.record_step(
                epoch=step,
                prefix='pre_train',
                new_dict={
                    'actor_loss_mean': losses_actor / batch_sz,
                    'critic_loss_mean': losses_critic / batch_sz,
                    'batch_sz': batch_sz,
                    'Demo_ratio': demo_n / batch_sz,
                 },
                display=False,
            )

            self.episode = 'pre_{}'.format(step)
            self.summary()

            self.tp.plot_data(
                prefix='pre_train',
                ep_start=self.conf.pretrain_save_step,
                ep_end=step,
                file_name='result-pretrain-{}.png'.format(self.episode),
                title=self.conf.exp_name+str(self.conf.exp_idx)+'-Pretrain',
                grid=False,
                ep_step=self.conf.pretrain_save_step,
            )

        self.episode = 1  # Restore

    def train(self):
        """TODO."""
        self.agent.train()
        # Define criterion

        start_time = time.time()
        while self.episode <= self.conf.n_episode:  # self.iter start from 1
            # Episodic statistics
            eps_since = time.time()
            eps_reward = 0
            eps_length = 0
            eps_actor_loss = 0
            eps_critic_loss = 0
            eps_batch_sz = 0
            eps_demo_n = 0

            # Choose a new environment.
            self.env = random.sample(self.all_envs, 1)[0]

            # Reset the environment.
            s0 = self.env.reset()
            n_agents = len(s0)

            # Reset action noise.
            self.action_noise.reset()

            done = False
            s_tensor = [self.agent.obs2tensor(s0[i]) for i in range(n_agents)]
            action_lst = []

            while not done:
                # 1. Run environment step
                with torch.no_grad():
                    # s_tensor = self.agent.obs2tensor(state)
                    action = [
                        self.agent.actor_b(s_tensor[i].to(
                            self.device)[None])[0] +
                        torch.from_numpy(self.action_noise()).float()
                        for i in range(n_agents)]
                    action_lst.extend([act.numpy() for act in action])
                    s2, r, done, _ = self.env.step([a.numpy() for a in action])
                    s2_tensor = [
                        self.agent.obs2tensor(s2[i]) for i in range(n_agents)]
                    for i in range(n_agents):
                        self.agent.memory.add((
                            s_tensor[i],
                            action[i],
                            torch.tensor([r[i]]).float(),
                            s2_tensor[i],
                            torch.tensor([self.agent.conf.gamma]),
                            DATA_RUNTIME))  # Add one-step to memory

                # 3. Record episodic statistics
                eps_reward += np.mean(r)
                eps_length += 1

                s_tensor = s2_tensor

            # Perform policy update.
            # for _ in range(10):  TODO
            losses_critic, losses_actor, demo_n, batch_sz = self.update_agent(
                self.conf.update_step)
            eps_actor_loss += losses_actor
            eps_critic_loss += losses_critic
            eps_batch_sz += batch_sz
            eps_demo_n += demo_n

            self.logger.info(
                '{}: '
                'Episode {}-Last:{}: '
                'Actor_loss={:.8f}, '
                'Critic_loss={:.8f}, '
                'Step={}, '
                'Reward={}, '
                'Demo_ratio={:.8f}, '
                'action_mean={:.8f}, '
                'action_std=={:.8f}'.format(
                    timeSince(start_time),
                    self.episode,
                    timeSince(eps_since),
                    eps_actor_loss / eps_batch_sz,
                    eps_critic_loss / eps_batch_sz,
                    eps_length, eps_reward, eps_demo_n / eps_batch_sz,
                    np.mean(action_lst),
                    np.std(action_lst),
                ))

            # Update target
            self.agent.update_target(
                self.agent.actor_b, self.agent.actor_t, self.episode)
            self.agent.update_target(
                self.agent.critic_b, self.agent.critic_t, self.episode)

            self.tp.record_step(
                epoch=self.episode,
                prefix='episode',
                new_dict={
                    'total_reward': eps_reward,
                    'length': eps_length,
                    'avg_reward': eps_reward / eps_length,
                    'elapsed_time': timeSince(eps_since, return_seconds=True),
                    'actor_loss_mean': eps_actor_loss / eps_batch_sz,
                    'critic_loss_mean': eps_critic_loss / eps_batch_sz,
                    'eps_length': eps_length,
                    'Demo_ratio': eps_demo_n / eps_batch_sz
                },
                display=False)

            if self.episode % self.conf.save_every == 0:
                self.summary()

            self.episode += 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'conf',
        type=str,
        help='Training Configuration')
    parser.add_argument(
        '--eval',
        action='store_true', default=False,
        help='Evaluation mode')
    parser.add_argument(
        '--collect',
        action='store_true', default=False,
        help='Collect Demonstration Data')
    parser.add_argument(
        '-n_collect',
        type=int, default=100,
        help='Number of episode for demo collection')

    args = parser.parse_args()
    conf_path = args.conf

    trainer = RLTrainer(conf_path, args.eval)
    if args.eval:
        trainer.eval(save_fig=False)
    elif args.collect:
        trainer.collect_demo(args.n_collect)
    else:
        if trainer.conf.pretrain_demo:
            trainer.pretrain()
        trainer.train()


if __name__ == '__main__':
    os.putenv('DISPLAY', ':0')
    main()
