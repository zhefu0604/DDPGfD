"""Run a training experiment using the DDPGfD algorithm.

Usage
-----

    python train.py "/path/to/config-file.yaml"
"""
import os
import sys
import time
import csv
import argparse
import torch
import numpy as np
import logging
from copy import deepcopy
from collections import deque
from ddpgfd.agents.base import DATA_RUNTIME
from ddpgfd.agents.fcnet import FeedForwardAgent
from ddpgfd.core.env import TrajectoryEnv
from ddpgfd.core.logger import logger_setup
from ddpgfd.core.training_utils import TrainingProgress
from ddpgfd.core.training_utils import load_conf


class RLTrainer(object):
    """RL algorithm object.

    Attributes
    ----------
    full_conf : object
        full configuration parameters
    conf : object
        training configuration parameters
    tp : ddpgfd.core.training_utils.TrainingProgress
        a logger for training progress
    logger : object
        an object used for logging purposes
    device : torch.device
        context-manager that changes the selected device.
    env : ddpgfd.core.env.TrajectoryEnv
        the training environment
    agent : ddpgfd.agents.base.DDPGfDAgent
        the training agent
    episode : int
        number of rollouts so far
    steps : int
        total number of environment steps
    epoch_episode_steps : list of float
        list of total steps in each epsiode since this epoch started
    epoch_episode_rewards : list of float
        list of cumulative rewards since this epoch started
    epoch_episodes : int
        number of episodes associated with this epoch of training
    epoch : int
        total number of training epochs so far
    episode_rew_history : collections.deque
        list of cumulative rewards for the past 100 rollouts
    info_at_done : collections.deque
        list of info dicts at the final step of the past 100 rollouts
    """

    def __init__(self, conf_path):
        """Instantiate the RL algorithm.

        Parameters
        ----------
        conf_path : str
            path to the configuration yaml file
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

        # Random seed.
        torch.manual_seed(self.conf.seed)  # cpu
        np.random.seed(self.conf.seed)  # numpy

        # Construct Env.
        self.env = TrajectoryEnv(self.full_conf.env_config)
        self.logger.info('Environment Loaded')

        # Create the agent class.
        self.agent = FeedForwardAgent(
            conf=self.full_conf,
            device=self.device,
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
        )
        self.agent.to(self.device)

        # init
        self.episode = 1
        self.steps = 0
        self.epoch_episode_steps = []
        self.epoch_episode_rewards = []
        self.epoch_episodes = 0
        self.epoch = 0
        self.episode_rew_history = deque(maxlen=100)
        self.info_at_done = deque(maxlen=100)

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
        self.agent.save(self.tp.progress_path, epoch=self.epoch)

        # Save configuration.
        self.tp.save_progress(self.epoch)
        self.tp.save_conf(self.conf.to_dict())

        self.logger.info('Config name ' + self.conf.exp_name)
        self.logger.info('Progress Saved, current epoch={}'.format(self.epoch))

    def train(self):
        """Perform end-to-end training procedure."""
        # Set the agent in training mode.
        self.agent.train()

        epoch_actor_loss = []
        epoch_critic_loss = []
        epoch_batch_sz = []
        epoch_demo_n = []
        start_time = time.time()
        while self.episode <= self.conf.n_episode:  # self.iter start from 1
            # Episodic statistics
            eps_reward = 0
            eps_length = 0

            # Reset the environment.
            s = self.env.reset()

            # Reset action noise.
            self.agent.reset()

            done = False
            info = {}
            action_lst = []
            while not done:
                with torch.no_grad():
                    # Compute noisy actions by the policy.
                    ac = self.agent.get_action(s)
                    action_lst.extend(ac)

                    # Run environment step.
                    s2, r, done, info = self.env.step(ac)

                    # Add one-step to memory.
                    self.agent.add_memory(
                        s=s,
                        a=ac,
                        r=r,
                        s2=s2,
                        gamma=self.full_conf.agent_config.gamma,
                        dtype=DATA_RUNTIME,
                    )

                    s = deepcopy(s2)

                # Record episodic statistics.
                self.steps += 1
                eps_reward += np.mean(r)
                eps_length += 1

                # Perform policy update.
                if self.steps % self.conf.update_step == 0:
                    update_step = self.steps // self.conf.update_step
                    q, a, d, b = self.agent.update_agent(update_step)

                    epoch_actor_loss.append(a)
                    epoch_critic_loss.append(q)
                    epoch_batch_sz.append(b)
                    epoch_demo_n.append(d)

            # More bookkeeping.
            self.epoch_episode_rewards.append(eps_reward)
            self.episode_rew_history.append(eps_reward)
            self.epoch_episode_steps.append(eps_length)
            self.epoch_episodes += 1
            self.info_at_done.append(info)

            if self.episode % self.conf.save_every == 0:
                # Log training performance.
                self._log_training(
                    start_time=start_time,
                    eps_actor_loss=np.mean(epoch_actor_loss),
                    eps_critic_loss=np.mean(epoch_critic_loss),
                    eps_batch_sz=np.mean(epoch_batch_sz),
                    eps_demo_n=np.mean(epoch_demo_n),
                    action_mean=np.mean(action_lst),
                    action_std=np.std(action_lst),
                )

                self.summary()

                # Reset epoch statistics.
                epoch_actor_loss.clear()
                epoch_critic_loss.clear()
                epoch_batch_sz.clear()
                epoch_demo_n.clear()
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

        # Information passed by the environment.
        for key in self.info_at_done[0].keys():
            if key != "mpg":
                continue
            combined_stats['info_at_done/{}'.format(key)] = np.mean([
                x[key] for x in self.info_at_done])

        for key in self.info_at_done[0].keys():
            combined_stats['rollout/{}'.format(key)] = np.mean([
                x[key]
                for x in list(self.info_at_done)[-self.conf.save_every:]])

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

    # Parse command-line arguments.
    flags = parser.parse_args(args)

    # Create the RL trainer object.
    trainer = RLTrainer(flags.conf)

    # Run the training procedure.
    trainer.train()


if __name__ == '__main__':
    os.putenv('DISPLAY', ':0')
    main(sys.argv[1:])
