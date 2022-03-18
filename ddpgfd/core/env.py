import os
import random
import gym
import trajectory.config as config
from gym.envs.registration import register


class TrajectoryEnv(gym.Env):

    def __init__(self):
        # Construct Env
        self.all_envs = self._create_envs()

        # Choose a current environment.
        self.current_env = random.sample(self.all_envs, 1)[0]

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

    def step(self, action):
        return self.current_env.step(action)

    def reset(self):
        # Clear memory from the prior environment. This is to deal with the
        # explosion in memory.
        if self.current_env.sim is not None:
            self.current_env.sim.data_by_vehicle.clear()

        # Choose a new environment.
        self.current_env = random.sample(self.all_envs, 1)[0]

        return self.current_env.reset()

    def render(self, mode="human"):
        pass
