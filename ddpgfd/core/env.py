"""Environment compatible with trajectory_training."""
import os
import random
import gym
import trajectory.config as config
from gym.envs.registration import register

# TODO
N_DOWNSTREAM = 4
# simulation update steps per action assignment
SIMS_PER_STEP = 1
# whether to use all trajectories or those with min speeds below some cutoff
ALL_ENVS = False
# minimum cutoff speed for that a (full) trajectory should have during training
MIN_SPEED = 22
# the names of all valid trajectories for training purposes
FILENAMES = [
    "2021-03-08-22-35-14_2T3MWRFVXLW056972_masterArray_0_4597",
    "2021-03-08-22-35-14_2T3MWRFVXLW056972_masterArray_1_4927",
    "2021-03-09-13-35-04_2T3MWRFVXLW056972_masterArray_0_6825",
    "2021-03-09-13-35-04_2T3MWRFVXLW056972_masterArray_1_4938",
    "2021-03-10-21-54-16_2T3MWRFVXLW056972_masterArray_0_4523",
    "2021-03-10-21-54-16_2T3MWRFVXLW056972_masterArray_1_4582",
    "2021-03-12-22-20-57_2T3MWRFVXLW056972_masterArray_0_5672",
    "2021-03-12-22-20-57_2T3MWRFVXLW056972_masterArray_1_4817",
    "2021-03-15-12-46-38_2T3MWRFVXLW056972_masterArray_0_4917",
    "2021-03-15-12-46-38_2T3MWRFVXLW056972_masterArray_1_11342",
    "2021-03-17-21-37-10_2T3MWRFVXLW056972_masterArray_0_4463",
    "2021-03-17-21-37-10_2T3MWRFVXLW056972_masterArray_1_4386",
    "2021-03-18-12-42-14_2T3MWRFVXLW056972_masterArray_0_3977",
    "2021-03-18-12-42-14_2T3MWRFVXLW056972_masterArray_1_3918",
    "2021-03-22-22-23-58_2T3MWRFVXLW056972_masterArray_0_4223",
    "2021-03-22-22-23-58_2T3MWRFVXLW056972_masterArray_1_4422",
    # Arwa said this trajectory might have sensor malfunction issue
    # "2021-03-23-21-50-02_2T3MWRFVXLW056972_masterArray_0_4331",
    "2021-03-23-21-50-02_2T3MWRFVXLW056972_masterArray_1_3778",
    # used for dashboard testing
    # "2021-03-24-12-39-15_2T3MWRFVXLW056972_masterArray_0_6438",
    "2021-03-24-12-39-15_2T3MWRFVXLW056972_masterArray_1_4102",
    "2021-03-24-21-34-31_2T3MWRFVXLW056972_masterArray_0_4937",
    "2021-03-24-21-34-31_2T3MWRFVXLW056972_masterArray_1_4364",
    "2021-03-26-21-26-45_2T3MWRFVXLW056972_masterArray_0_4540",
    "2021-03-26-21-26-45_2T3MWRFVXLW056972_masterArray_1_4028",
    "2021-03-29-12-47-15_2T3MWRFVXLW056972_masterArray_0_5016",
    "2021-03-29-12-47-15_2T3MWRFVXLW056972_masterArray_1_4185",
    "2021-03-31-21-39-05_2T3MWRFVXLW056972_masterArray_0_4200",
    "2021-03-31-21-39-05_2T3MWRFVXLW056972_masterArray_1_4622",
    "2021-04-02-21-31-47_2T3MWRFVXLW056972_masterArray_0_4125",
    "2021-04-02-21-31-47_2T3MWRFVXLW056972_masterArray_1_4111",
    "2021-04-05-21-39-05_2T3MWRFVXLW056972_masterArray_0_4357",
    "2021-04-05-21-39-05_2T3MWRFVXLW056972_masterArray_1_4173",
    "2021-04-06-21-18-22_2T3MWRFVXLW056972_masterArray_0_4427",
    "2021-04-06-21-18-22_2T3MWRFVXLW056972_masterArray_1_4496",
    "2021-04-07-12-33-03_2T3MWRFVXLW056972_masterArray_0_11294",
    "2021-04-07-12-33-03_2T3MWRFVXLW056972_masterArray_1_6116",
    "2021-04-07-21-22-07_2T3MWRFVXLW056972_masterArray_0_4101",
    "2021-04-07-21-22-07_2T3MWRFVXLW056972_masterArray_1_4069",
    "2021-04-12-21-34-57_2T3MWRFVXLW056972_masterArray_0_4796",
    "2021-04-12-21-34-57_2T3MWRFVXLW056972_masterArray_1_4436",
    "2021-04-15-21-32-46_2T3MWRFVXLW056972_masterArray_0_3889",
    # used for dashboard testing
    # "2021-04-15-21-32-46_2T3MWRFVXLW056972_masterArray_1_3685",
    "2021-04-16-12-34-41_2T3MWRFVXLW056972_masterArray_0_5778",
    "2021-04-16-12-34-41_2T3MWRFVXLW056972_masterArray_1_4387",
    "2021-04-19-12-27-33_2T3MWRFVXLW056972_masterArray_0_16467",
    "2021-04-19-12-27-33_2T3MWRFVXLW056972_masterArray_1_6483",
    "2021-04-19-22-09-19_2T3MWRFVXLW056972_masterArray_0_4433",
    "2021-04-19-22-09-19_2T3MWRFVXLW056972_masterArray_1_4288",
    "2021-04-20-21-42-34_2T3MWRFVXLW056972_masterArray_0_4025",
    "2021-04-20-21-42-34_2T3MWRFVXLW056972_masterArray_1_3973",
    "2021-04-21-21-45-12_2T3MWRFVXLW056972_masterArray_0_3957",
    "2021-04-21-21-45-12_2T3MWRFVXLW056972_masterArray_1_3621",
    # used for dashboard testing
    # "2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050",
    "2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_1_5292",
    "2021-04-26-21-13-18_2T3MWRFVXLW056972_masterArray_0_4595",
    "2021-04-26-21-13-18_2T3MWRFVXLW056972_masterArray_1_4664",
    "2021-04-27-21-37-32_2T3MWRFVXLW056972_masterArray_0_3836",
    "2021-04-27-21-37-32_2T3MWRFVXLW056972_masterArray_1_3558",
    "2021-04-29-21-16-14_2T3MWRFVXLW056972_masterArray_0_4190",
    "2021-04-29-21-16-14_2T3MWRFVXLW056972_masterArray_1_4005",
]


class TrajectoryEnv(gym.Env):
    """Environment compatible with trajectory_training.

    Separate environments are created for each trajectory, and at reset, one
    trajectory is sampled and advanced for the corresponding rollout.

    Holdout set trajectories are not included during training.
    """

    def __init__(self, conf):
        """Instantiate the environment."""
        # minimum speed in each trajectory, precomputed for all available ones
        self._min_speed = [
            22.1035, 21.3898, 0., 18.2044, 24.6943, 23.7686, 18.5114, 22.7613,
            7.57993, 0., 18.8923, 26.3929, 25.554, 29.4043, 27.6191, 22.1418,
            31.6758, 27.9987, 17.5997, 26.7967, 24.227, 27.7384, 9.99979,
            26.6608, 26.0128, 23.6203, 28.8373, 29.15, 23.43, 27.1137, 25.3643,
            22.3794, 0., 0.51424, 28.3918, 28.8141, 21.4412, 24.8492, 24.089,
            4.07274, 26.09, 0., 0., 23.8667, 25.9932, 25.7978, 29.208, 28.445,
            30.4969, 0., 24.3386, 24.9839, 20.2515, 30.5814, 23.0111, 28.9075,
        ]

        # valid indices
        if ALL_ENVS:
            self._valid_indices = list(range(len(FILENAMES)))
        else:
            self._valid_indices = [
                ix for ix in range(len(FILENAMES))
                if self._min_speed[ix] < MIN_SPEED]

        # time horizon for each trajectory. Used to weight the sampling of
        # trajectories by their time horizon.
        self._horizons = [
            4598, 4928, 6826, 4939, 4524, 4583, 5673, 4818, 4918, 11343, 4464,
            4387, 3978, 3919, 4224, 4423, 3779, 4103, 4938, 4365, 4541, 4029,
            5017, 4186, 4201, 4623, 4126, 4112, 4358, 4174, 4428, 4497, 11295,
            6117, 4102, 4070, 4797, 4437, 3890, 5779, 4388, 16468, 6484, 4434,
            4289, 4026, 3974, 3958, 3622, 5293, 4596, 4665, 3837, 3559, 4191,
            4006,
        ]
        if not ALL_ENVS:
            self._horizons = [
                self._horizons[i] for i in range(len(self._horizons))
                if i in self._valid_indices]

        # Construct all valid environments.
        self.all_envs = self._create_envs(conf, self._valid_indices)

        # Choose a current environment.
        self.current_env = self._choose_env()

    @staticmethod
    def _create_envs(conf, valid_indices):
        """Create a separate environment for each trajectory."""
        all_envs = []
        for i in valid_indices:
            env_config = {
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
                'platoon': 'av',  # '2avs_4%',
                'av_controller': 'rl',
                'av_kwargs': '{}',
                'human_controller': 'idm',
                'human_kwargs': '{}',
                'fixed_traj_path': os.path.join(
                    config.PROJECT_PATH,
                    'dataset/data_v2_preprocessed_west/{}/'
                    'trajectory.csv'.format(FILENAMES[i])
                ),
                'lane_changing': True,
            }
            env_config.update(conf)

            register(
                id="TrajectoryEnv-v{}".format(i),
                entry_point="trajectory.env.trajectory_env:TrajectoryEnv",
                kwargs={"config": env_config, "_simulate": True})

            # Make the gym environment.
            all_envs.append(gym.make("TrajectoryEnv-v{}".format(i)))

        return all_envs

    def _choose_env(self, proportional=False):
        """Return a valid environment weighted on the time horizon of each.

        Parameters
        ----------
        proportional : bool
            whether to choose an environment proportional to its time horizon
        """
        if proportional:
            n_envs = len(self._horizons)
            ix = random.choices(
                list(range(n_envs)), weights=self._horizons, k=1)[0]
            return self.all_envs[ix]
        else:
            return random.sample(self.all_envs, 1)[0]

    def step(self, action):
        """Advance the simulation by one step."""
        # Run for a few steps.
        for _ in range(SIMS_PER_STEP - 1):
            self.current_env.step(action)

        # Return after the final step.
        obs, rew, done, info = self.current_env.step(action)

        # done = done or any(
        #     av.get_headway() > 500 for av in self.current_env.avs)

        return obs, rew, done, info

    def reset(self):
        """Reset the environment.

        A new trajectory is chosen and memory is cleared here.
        """
        # Clear memory from the prior environment. This is to deal with the
        # explosion in memory.
        if self.current_env.sim is not None:
            self.current_env.sim.data_by_vehicle.clear()

        # Choose a new environment.
        self.current_env = self._choose_env()

        return self.current_env.reset()

    def render(self, mode="human"):
        """See parent class."""
        pass

    @property
    def observation_space(self):
        """Return the observation space."""
        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(15 + 2 * N_DOWNSTREAM,))

    @property
    def action_space(self):
        """Return the action space."""
        return gym.spaces.Box(low=-1, high=1, shape=(1,))
