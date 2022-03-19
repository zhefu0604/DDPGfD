"""TODO."""
import os
import sys
import time
import csv
import argparse
import torch
import numpy as np
import logging
import pickle
from collections import deque
from ddpgfd.core.agent import DDPGfDAgent
from ddpgfd.core.agent import DATA_RUNTIME
from ddpgfd.core.agent import DATA_DEMO
from ddpgfd.core.env import TrajectoryEnv
from ddpgfd.core.logger import logger_setup
from ddpgfd.core.training_utils import TrainingProgress
from ddpgfd.core.training_utils import load_conf

np.set_printoptions(suppress=True, precision=4)


class RLTrainer:
    """TODO.

    Attributes
    ----------
    """

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
        self.env = TrajectoryEnv()

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

    def restore_progress(self):
        """Restore progress from a previous run to continue training."""
        self.tp.restore_progress(self.conf.tps)

        # Restore weights and biases.
        self.agent.load(self.tp.progress_path, epoch=self.conf.tps)

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
        self.agent.save(self.tp.progress_path, epoch=self.epoch)

        # Save configuration.
        self.tp.save_progress(self.epoch)
        self.tp.save_conf(self.conf.to_dict())

        self.logger.info('Config name ' + self.conf.exp_name)
        self.logger.info('Progress Saved, current epoch={}'.format(self.epoch))

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
        loss_critic = []
        loss_actor = []
        demo_n = []
        batch_sz = []
        for t in range(self.conf.pretrain_step):
            q, a, d, b = self.agent.update_agent(t)
            loss_critic.append(q)
            loss_actor.append(a)
            demo_n.append(d)
            batch_sz.append(b)

        # Log training performance.
        self._log_training(
            start_time=start_time,
            eps_actor_loss=np.mean(loss_actor),
            eps_critic_loss=np.mean(loss_critic),
            eps_batch_sz=np.mean(batch_sz),
            eps_demo_n=np.mean(demo_n),
            action_mean=None,  # no actions sampled
            action_std=None,  # no actions sampled
        )

        self.summary()

        self.episode = 1  # Restore

    def train(self):
        """Perform end-to-end training procedure."""
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
                        torch.clip(
                            self.agent.actor_b(s_tensor[i].to(
                                self.device)[None])[0] +
                            torch.from_numpy(
                                self.agent.action_noise()).float(),
                            min=-0.99,
                            max=0.99,
                        ) for i in range(n_agents)]
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
                self.steps += 1
                eps_reward += np.mean(r)
                eps_length += 1

                # Perform policy update.
                if self.steps % self.conf.update_step == 0:
                    q, a, d, b = self.agent.update_agent(
                        self.steps // self.conf.update_step)

                    epoch_actor_loss.append(a)
                    epoch_critic_loss.append(q)
                    epoch_batch_sz.append(b)
                    epoch_demo_n.append(d)

            # More bookkeeping.
            self.epoch_episode_rewards.append(eps_reward)
            self.episode_rew_history.append(eps_reward)
            self.epoch_episode_steps.append(eps_length)
            self.epoch_episodes += 1

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
