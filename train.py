"""TODO."""
import os
import sys
import time
import csv
import argparse
import random
import torch
import numpy as np
import logging
import gym
import pickle
import trajectory.config as config
from gym.envs.registration import register
from collections import deque
from ddpgfd.core.agent import DDPGfDAgent
from ddpgfd.core.agent import DATA_RUNTIME
from ddpgfd.core.agent import DATA_DEMO
from ddpgfd.core.logger import logger_setup
from ddpgfd.core.training_utils import TrainingProgress
from ddpgfd.core.training_utils import load_conf

np.set_printoptions(suppress=True, precision=4)


class RLTrainer:

    def __init__(self, conf_path, evaluate=False):
        """Instantiate the RL algorithm.

        Parameters
        ----------
        conf_path : str
            path to the configuration yaml file
        evaluate : bool
            whether running evaluations
        """
        self.full_conf = load_conf(conf_path)
        self.conf = self.full_conf.train_config

        result_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'result')

        # Store in xxxx_dir/exp_name+exp_idx/...
        self.tp = TrainingProgress(result_dir, self.conf.exp_name)

        logger_setup(
            os.path.join(self.tp.result_path, self.conf.exp_name + '-log.txt'),
            loggers=['RLTrainer', 'DDPGfD', 'TP'],
            level=logging.DEBUG)
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

        self.agent = DDPGfDAgent(self.full_conf, self.device)
        self.agent.to(self.device)

        if self.conf.restore:
            self.restore_progress()
        else:
            self.episode = 1

        # Initialize replay buffer with demonstrations.
        self.demo2memory()

        # init
        self.steps = 0
        self.epoch_episode_steps = []
        self.epoch_episode_rewards = []
        self.epoch_episodes = 0
        self.epoch = 0
        self.episode_rew_history = deque(maxlen=100)

    @staticmethod
    def _create_envs():
        """Create a separate environment for each trajectory."""
        filenames = [
            "2021-03-08-22-35-14_2T3MWRFVXLW056972_masterArray_0_4597.csv",
            # "2021-03-08-22-35-14_2T3MWRFVXLW056972_masterArray_1_4927.csv",
            # "2021-03-09-13-35-04_2T3MWRFVXLW056972_masterArray_0_6825.csv",
            # "2021-03-09-13-35-04_2T3MWRFVXLW056972_masterArray_1_4938.csv",
            # "2021-03-10-21-54-16_2T3MWRFVXLW056972_masterArray_0_4523.csv",
            # "2021-03-10-21-54-16_2T3MWRFVXLW056972_masterArray_1_4582.csv",
            # "2021-03-12-22-20-57_2T3MWRFVXLW056972_masterArray_0_5672.csv",
            # "2021-03-12-22-20-57_2T3MWRFVXLW056972_masterArray_1_4817.csv",
            # "2021-03-15-12-46-38_2T3MWRFVXLW056972_masterArray_0_4917.csv",
            # "2021-03-15-12-46-38_2T3MWRFVXLW056972_masterArray_1_11342.csv",
            # "2021-03-17-21-37-10_2T3MWRFVXLW056972_masterArray_0_4463.csv",
            # "2021-03-17-21-37-10_2T3MWRFVXLW056972_masterArray_1_4386.csv",
            # "2021-03-18-12-42-14_2T3MWRFVXLW056972_masterArray_0_3977.csv",
            # "2021-03-18-12-42-14_2T3MWRFVXLW056972_masterArray_1_3918.csv",
            # "2021-03-22-22-23-58_2T3MWRFVXLW056972_masterArray_0_4223.csv",
            # "2021-03-22-22-23-58_2T3MWRFVXLW056972_masterArray_1_4422.csv",
            # # Arwa said this trajectory might has sensor malfunction issue
            # # "2021-03-23-21-50-02_2T3MWRFVXLW056972_masterArray_0_4331.csv",
            # "2021-03-23-21-50-02_2T3MWRFVXLW056972_masterArray_1_3778.csv",
            # # used for dashboard testing
            # # "2021-03-24-12-39-15_2T3MWRFVXLW056972_masterArray_0_6438.csv",
            # "2021-03-24-12-39-15_2T3MWRFVXLW056972_masterArray_1_4102.csv",
            # "2021-03-24-21-34-31_2T3MWRFVXLW056972_masterArray_0_4937.csv",
            # "2021-03-24-21-34-31_2T3MWRFVXLW056972_masterArray_1_4364.csv",
            # "2021-03-26-21-26-45_2T3MWRFVXLW056972_masterArray_0_4540.csv",
            # "2021-03-26-21-26-45_2T3MWRFVXLW056972_masterArray_1_4028.csv",
            # "2021-03-29-12-47-15_2T3MWRFVXLW056972_masterArray_0_5016.csv",
            # "2021-03-29-12-47-15_2T3MWRFVXLW056972_masterArray_1_4185.csv",
            # "2021-03-31-21-39-05_2T3MWRFVXLW056972_masterArray_0_4200.csv",
            # "2021-03-31-21-39-05_2T3MWRFVXLW056972_masterArray_1_4622.csv",
            # "2021-04-02-21-31-47_2T3MWRFVXLW056972_masterArray_0_4125.csv",
            # "2021-04-02-21-31-47_2T3MWRFVXLW056972_masterArray_1_4111.csv",
            # "2021-04-05-21-39-05_2T3MWRFVXLW056972_masterArray_0_4357.csv",
            # "2021-04-05-21-39-05_2T3MWRFVXLW056972_masterArray_1_4173.csv",
            # "2021-04-06-21-18-22_2T3MWRFVXLW056972_masterArray_0_4427.csv",
            # "2021-04-06-21-18-22_2T3MWRFVXLW056972_masterArray_1_4496.csv",
            # "2021-04-07-12-33-03_2T3MWRFVXLW056972_masterArray_0_11294.csv",
            # "2021-04-07-12-33-03_2T3MWRFVXLW056972_masterArray_1_6116.csv",
            # "2021-04-07-21-22-07_2T3MWRFVXLW056972_masterArray_0_4101.csv",
            # "2021-04-07-21-22-07_2T3MWRFVXLW056972_masterArray_1_4069.csv",
            # "2021-04-12-21-34-57_2T3MWRFVXLW056972_masterArray_0_4796.csv",
            # "2021-04-12-21-34-57_2T3MWRFVXLW056972_masterArray_1_4436.csv",
            # "2021-04-15-21-32-46_2T3MWRFVXLW056972_masterArray_0_3889.csv",
            # # used for dashboard testing
            # # "2021-04-15-21-32-46_2T3MWRFVXLW056972_masterArray_1_3685.csv",
            # "2021-04-16-12-34-41_2T3MWRFVXLW056972_masterArray_0_5778.csv",
            # "2021-04-16-12-34-41_2T3MWRFVXLW056972_masterArray_1_4387.csv",
            # "2021-04-19-12-27-33_2T3MWRFVXLW056972_masterArray_0_16467.csv",
            # "2021-04-19-12-27-33_2T3MWRFVXLW056972_masterArray_1_6483.csv",
            # "2021-04-19-22-09-19_2T3MWRFVXLW056972_masterArray_0_4433.csv",
            # "2021-04-19-22-09-19_2T3MWRFVXLW056972_masterArray_1_4288.csv",
            # "2021-04-20-21-42-34_2T3MWRFVXLW056972_masterArray_0_4025.csv",
            # "2021-04-20-21-42-34_2T3MWRFVXLW056972_masterArray_1_3973.csv",
            # "2021-04-21-21-45-12_2T3MWRFVXLW056972_masterArray_0_3957.csv",
            # "2021-04-21-21-45-12_2T3MWRFVXLW056972_masterArray_1_3621.csv",
            # # used for dashboard testing
            # # "2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050.csv",
            # "2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_1_5292.csv",
            # "2021-04-26-21-13-18_2T3MWRFVXLW056972_masterArray_0_4595.csv",
            # "2021-04-26-21-13-18_2T3MWRFVXLW056972_masterArray_1_4664.csv",
            # "2021-04-27-21-37-32_2T3MWRFVXLW056972_masterArray_0_3836.csv",
            # "2021-04-27-21-37-32_2T3MWRFVXLW056972_masterArray_1_3558.csv",
            # "2021-04-29-21-16-14_2T3MWRFVXLW056972_masterArray_0_4190.csv",
            # "2021-04-29-21-16-14_2T3MWRFVXLW056972_masterArray_1_4005.csv",
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
        """Restore progress from a previous run to continue training."""
        self.tp.restore_progress(self.conf.tps)

        # Restore weights and biases.
        self.agent.actor_b.load_state_dict(self.tp.restore_model_weight(
                self.conf.tps, self.device, prefix='actor_b'))
        self.agent.actor_t.load_state_dict(self.tp.restore_model_weight(
            self.conf.tps, self.device, prefix='actor_t'))
        self.agent.critic_b.load_state_dict(self.tp.restore_model_weight(
            self.conf.tps, self.device, prefix='critic_b'))
        self.agent.critic_t.load_state_dict(self.tp.restore_model_weight(
            self.conf.tps, self.device, prefix='critic_t'))

        # Restore random seeds and number of episodes previously trained.
        self.episode = self.tp.get_meta('saved_episode') + 1
        np.random.set_state(self.tp.get_meta('np_random_state'))
        torch.random.set_rng_state(self.tp.get_meta('torch_random_state'))

        self.logger.info('Restore Progress, Episode={}'.format(self.episode))

    def summary(self):
        """Log data to tensorboard and save policy parameters."""
        # call Test/Evaluation here
        self.tp.add_meta({
            'saved_episode': self.episode,
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.random.get_rng_state()
        })

        # Save the policy weights, biases, and configuration parameters.
        self.save_progress()

    def save_progress(self):
        """Save the policy weights, biases, and configuration parameters."""
        # Save policy parameters.
        self.tp.save_model_weight(
            self.agent.actor_b, self.episode, prefix='actor_b')
        self.tp.save_model_weight(
            self.agent.actor_t, self.episode, prefix='actor_t')
        self.tp.save_model_weight(
            self.agent.critic_b, self.episode, prefix='critic_b')
        self.tp.save_model_weight(
            self.agent.critic_t, self.episode, prefix='critic_t')

        # Save configuration.
        self.tp.save_progress(self.episode)
        self.tp.save_conf(self.conf.to_dict())

        self.logger.info('Config name ' + self.conf.exp_name)
        self.logger.info('Progress Saved, current episode={}'.format(
            self.episode))

    def demo2memory(self):
        """Import demonstration from pkl files to the replay buffer."""
        dconf = self.full_conf.demo_config
        if dconf.load_demo_data:
            filenames = [
                x for x in os.listdir(dconf.demo_dir) if x.endswith(".pkl")]
            for ix, f_idx in enumerate(filenames):
                fname = os.path.join(dconf.demo_dir, f_idx)
                with open(fname, 'rb') as f:
                    data = pickle.load(f)
                for i in range(len(data)):
                    # Extract demonstration.
                    s, a, r, s2 = data[i]

                    # Convert to be pytorch compatible.
                    s_tensor = torch.from_numpy(s).float()
                    s2_tensor = torch.from_numpy(s2).float()
                    action = torch.from_numpy(a).float()

                    # Add one-step to memory.
                    self.agent.memory.add((
                        s_tensor,
                        action,
                        torch.tensor([r]).float(),
                        s2_tensor,
                        torch.tensor([self.agent.agent_conf.gamma]),
                        DATA_DEMO))

                self.logger.info(
                    '{} Demo Trajectories Loaded. Total Experience={}'.format(
                        ix + 1, len(self.agent.memory)))

            # Prevent demonstrations from being deleted.
            if not dconf.random:
                self.agent.memory.set_protect_size(len(self.agent.memory))
        else:
            self.logger.info('No Demo Trajectory Loaded')

    def pretrain(self):
        """Perform training on initial demonstration data."""
        assert self.full_conf.demo_config.load_demo_data
        self.agent.train()
        start_time = time.time()
        self.logger.info('Run Pretrain')
        self.episode = 'pre_{}'.format(self.conf.pretrain_step)

        # Perform training on demonstration data.
        loss_critic, loss_actor, demo_n, batch_sz = self.agent.update_agent(
            self.conf.pretrain_step)

        # Log training performance.
        self._log_training(
            start_time=start_time,
            eps_actor_loss=loss_actor,
            eps_critic_loss=loss_critic,
            eps_batch_sz=batch_sz,
            eps_demo_n=demo_n,
            action_mean=None,  # no actions sampled
            action_std=None,  # no actions sampled
        )

        self.summary()

        self.episode = 1  # Restore

    def train(self):
        """Perform end-to-end training procedure."""
        self.agent.train()

        start_time = time.time()
        while self.episode <= self.conf.n_episode:  # self.iter start from 1
            # Episodic statistics
            eps_reward = 0
            eps_length = 0
            eps_actor_loss = 0
            eps_critic_loss = 0
            eps_batch_sz = 0
            eps_demo_n = 0

            # Clear memory from the prior environment. This is to deal with the
            # explosion in memory.
            if self.env.sim is not None:
                self.env.sim.data_by_vehicle.clear()

            # Choose a new environment.
            self.env = random.sample(self.all_envs, 1)[0]

            # Reset the environment.
            s0 = self.env.reset()
            n_agents = len(s0)

            # Reset action noise.
            self.agent.action_noise.reset()

            done = False
            s_tensor = [self.agent.obs2tensor(s0[i]) for i in range(n_agents)]
            action_lst = []

            while not done:
                with torch.no_grad():
                    # Compute noisy actions by the policy.
                    action = [
                        self.agent.actor_b(s_tensor[i].to(
                            self.device)[None])[0] +
                        torch.from_numpy(self.agent.action_noise()).float()
                        for i in range(n_agents)]
                    action_lst.extend([act.numpy() for act in action])

                    # Run environment step.
                    s2, r, done, _ = self.env.step([a.numpy() for a in action])

                    # Add one-step to memory.
                    s2_tensor = [
                        self.agent.obs2tensor(s2[i]) for i in range(n_agents)]
                    for i in range(n_agents):
                        self.agent.memory.add((
                            s_tensor[i],
                            action[i],
                            torch.tensor([r[i]]).float(),
                            s2_tensor[i],
                            torch.tensor([self.agent.agent_conf.gamma]),
                            DATA_RUNTIME))

                    s_tensor = s2_tensor

                # Record episodic statistics.
                eps_reward += np.mean(r)
                eps_length += 1

            # More bookkeeping.
            self.epoch_episode_rewards.append(eps_reward)
            self.episode_rew_history.append(eps_reward)
            self.epoch_episode_steps.append(eps_length)
            self.steps += eps_length
            self.epoch_episodes += 1

            # Perform policy update.
            loss_critic, loss_actor, demo_n, batch = self.agent.update_agent(
                self.conf.update_step)

            eps_actor_loss += loss_actor
            eps_critic_loss += loss_critic
            eps_batch_sz += batch
            eps_demo_n += demo_n

            # Update target.
            self.agent.update_target(self.agent.actor_b, self.agent.actor_t)
            self.agent.update_target(self.agent.critic_b, self.agent.critic_t)

            if self.episode % self.conf.save_every == 0:
                # Log training performance.
                self._log_training(
                    start_time=start_time,
                    eps_actor_loss=eps_actor_loss,
                    eps_critic_loss=eps_critic_loss,
                    eps_batch_sz=eps_batch_sz,
                    eps_demo_n=eps_demo_n,
                    action_mean=np.mean(action_lst),
                    action_std=np.std(action_lst),
                )

                # TODO
                self.summary()

                # Reset epoch statistics.
                self.epoch_episodes = 0
                self.epoch_episode_rewards.clear()
                self.epoch_episode_steps.clear()

                # Update training epoch.
                self.epoch += 1

            self.episode += 1

    def _log_training(self,
                      start_time,
                      eps_actor_loss,
                      eps_critic_loss,
                      eps_batch_sz,
                      eps_demo_n,
                      action_mean,
                      action_std):
        """Log training statistics.

        Parameters
        ----------
        start_time : float
            the time when training began. This is used to print the total
            training time.
        """
        # Log statistics.
        duration = time.time() - start_time

        combined_stats = {
            # Rollout statistics.
            'rollout/episodes': self.epoch_episodes,
            'rollout/episode_steps': np.mean(self.epoch_episode_steps),
            'rollout/return': np.mean(self.epoch_episode_rewards),
            'rollout/return_history': np.mean(self.episode_rew_history),
            'rollout/actor_loss_mean': eps_actor_loss / eps_batch_sz,
            'rollout/critic_loss_mean': eps_critic_loss / eps_batch_sz,
            'rollout/demo_ratio': eps_demo_n / eps_batch_sz,
            'action_mean': action_mean,
            'action_std':  action_std,

            # Total statistics.
            'total/epochs': self.epoch + 1,
            'total/steps': self.steps,
            'total/duration': duration,
            'total/steps_per_second': self.steps / duration,
            'total/episodes': self.episode,

        }

        # Save combined_stats in a csv file.
        file_path = os.path.join(self.tp.result_path, "train.csv")
        exists = os.path.exists(file_path)
        with open(file_path, 'a') as f:
            w = csv.DictWriter(f, fieldnames=combined_stats.keys())
            if not exists:
                w.writeheader()
            w.writerow(combined_stats)

        # Print statistics.
        print("-" * 67)
        for key in sorted(combined_stats.keys()):
            val = combined_stats[key]
            print("| {:<30} | {:<30} |".format(key, val))
        print("-" * 67)
        print('')


def main(args):
    """Perform the training procedure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'conf',
        type=str,
        help='Training Configuration')
    parser.add_argument(
        '--eval',
        action='store_true', default=False,
        help='Evaluation mode')

    # Parse command-line arguments.
    flags = parser.parse_args(args)

    # Create the RL trainer object.
    trainer = RLTrainer(flags.conf, flags.eval)

    # Pretrain the policy for a number of steps.
    if trainer.conf.pretrain_demo:
        trainer.pretrain()

    # Run the training procedure.
    trainer.train()


if __name__ == '__main__':
    os.putenv('DISPLAY', ':0')
    main(sys.argv[1:])
