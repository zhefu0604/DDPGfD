"""Generate demonstrations from an emission file."""
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import bisect
import os
import json

from trajectory.env.energy_models import PFMCompactSedan


# =========================================================================== #
#                           observation parameters                            #
# =========================================================================== #

# number of backward timesteps to add to observation
N_STEPS = 5
# headway scaling term
HEADWAY_SCALE = 200.
# speed scaling term
SPEED_SCALE = 40.
# time discretization
DT = 0.1
# number of downstream edges to be sensed
N_DOWNSTREAM = 4
# scaling term for distances to downstream edges
DISTANCE_SCALE = 1610.


# =========================================================================== #
#                              action parameters                              #
# =========================================================================== #

# scaling term for the actions (so that values range from -1 to 1)
ACTION_SCALE = 1.


# =========================================================================== #
#                         reward function parameters                          #
# =========================================================================== #

# scaling term for the rewards
REWARD_SCALE = 0.1
# minimum desirable space headway. If set to -1, no penalty is applied.
H_LOW = -1
# maximum desirable space headway. If set to -1, no penalty is applied.
H_HIGH = 150
# minimum desirable time headway. If set to -1, no penalty is applied.
TH_LOW = 1.5
# maximum desirable time headway. If set to -1, no penalty is applied.
TH_HIGH = -1
# number of timesteps to average the energy reward across
ENERGY_STEPS = 1
# maximum magnitude of accelerations after which penalties are incurred
MAX_ACCEL = 0.25


def parse_args(args):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate demonstrations from an emission file.",
        epilog="python generate_video.py INFILE OUTFILE")

    parser.add_argument(
        'infile', type=str, help='the path to the emission file')
    parser.add_argument(
        'outfile', type=str, help='the path to save the demonstrations in')

    parser.add_argument(
        '--downstream_path',
        type=str,
        default=None,
        help='the path to the folder containing trajectory information')

    return parser.parse_args(args)


def get_leader_avg_speed(segments, avg_speed, times, pos, t, n):
    """Return a list of relevant macroscopic data.

    Parameters
    ----------
    segments : array_like
        the starting position of every whose macroscopic state is estimated
    avg_speed : array_like
        the most recent average speed of every segment
    times : array_like
        the times when each traffic state estimate was taken
    pos : array_like
        list of vehicle positions over time
    t : float
        current time, in seconds
    n : int
        number of downstream edges to be sensed

    Returns
    -------
    list of float
        downstream segment distances and average speeds
    """
    if n == 0:
        return []

    t_index = bisect.bisect(times, t) - 1
    x_index = bisect.bisect(segments, pos)

    return list(
        avg_speed[t_index, x_index:x_index + n] / SPEED_SCALE) + list(
        (np.array(segments[x_index:x_index + n]) - pos) / DISTANCE_SCALE)


def huber_loss(a, delta=1.0):
    """Return Huber loss function.

    Parameters
    ----------
    a : float
        the error
    delta : float
        cutoff point for quadratic loss

    Returns
    -------
    float : int
        the loss
    """
    if abs(a) <= delta:
        return 0.5 * a ** 2
    else:
        return delta * (abs(a) - 0.5 * delta)


def reward_fn(headway, accel, realized_accel, speed, t, energy_model):
    """Compute the reward at a specific time index.

    Parameters
    ----------
    headway : array_like
        list of vehicle headways over time
    accel : array_like
        list of desired vehicle accelerations over time
    realized_accel : array_like
        list of realized vehicle accelerations over time
    speed : array_like
        list of vehicle speed over time
    t : int
        the time index
    energy_model : any
        the energy model used to compute fuel consumption

    Returns
    -------
    float
        the reward
    """
    reward = 0.
    # reward = 0.001 * speed[t]

    # time headway reward
    th = min(headway[t] / max(speed[t], 0.01), 20)
    if TH_LOW > 0 and th < TH_LOW:
        reward -= 2 * 0.1 * huber_loss(th - TH_LOW, delta=1.00)
    if TH_HIGH > 0 and th > TH_HIGH:
        reward -= huber_loss(th - TH_HIGH, delta=0.25)

    # space headway reward
    h = headway[t]
    if H_LOW > 0 and h < H_LOW:
        reward -= 2 * huber_loss((h - H_LOW) / 10., delta=1.00)
    if H_HIGH > 0 and h > H_HIGH:
        reward -= 0.1 * huber_loss((h - H_HIGH) / 10., delta=0.25)

    # # acceleration reward
    # a = realized_accel[t]
    # if a > MAX_ACCEL:
    #     reward -= 0.1 * (a - MAX_ACCEL) ** 2
    # if a < -MAX_ACCEL:
    #     reward -= 0.1 * (a + MAX_ACCEL) ** 2
    #
    # # failsafe reward
    # # reward -= (accel[t] - realized_accel[t]) ** 2

    # energy consumption reward
    sum_energy = sum([
        energy_model.get_instantaneous_fuel_consumption(
            speed=speed_i, accel=accel_i, grade=0)
        for speed_i, accel_i in zip(speed[max(t-ENERGY_STEPS+1, 0): t+1],
                                    accel[max(t-ENERGY_STEPS+1, 0): t+1])])
    sum_speed = sum(np.clip(
        speed[max(t-ENERGY_STEPS+1, 0): t+1], a_min=0.1, a_max=np.inf))

    reward -= min(sum_energy / sum_speed, 0.2)

    return reward


def obs(pos, headway, speed, leader_speed, avg_speed, segments, times, t, n):
    """Compute the observation at a specific time index.

    Parameters
    ----------
    pos : array_like
        list of vehicle positions over time
    headway : array_like
        list of vehicle headways over time
    speed : array_like
        list of vehicle speed over time
    leader_speed : array_like
        list of vehicle lead speeds over time
    avg_speed : array_like
        the most recent average speed of every segment
    segments : array_like
        the starting position of every whose macroscopic state is estimated
    times : array_like
        the times when each traffic state estimate was taken
    t : int
        the time index
    n : int
        number of downstream edges to be sensed

    Returns
    -------
    array_like
        the observation
    """
    min_t = max(0, t - N_STEPS + 1)
    max_t = t + 1
    n_missed = N_STEPS - max_t + min_t

    return np.array(
        [0.] * n_missed +
        list(speed[min_t: max_t] / SPEED_SCALE) +
        [0.] * n_missed +
        list((leader_speed[min_t: max_t]-speed[min_t: max_t]) / SPEED_SCALE) +
        [0.] * n_missed +
        list(headway[min_t: max_t] / HEADWAY_SCALE) +
        get_leader_avg_speed(
            segments, avg_speed, times, pos=pos[t], t=t * DT, n=n)
    )


def action(accel, t):
    """Compute the action at a specific time index.

    Parameters
    ----------
    accel : array_like
        list of vehicle accelerations over time
    t : int
        the time index

    Returns
    -------
    array_like
        the observation
    """
    return np.array([ACTION_SCALE * accel[t]])


def main(args):
    """Run the main operation."""
    # Parse command-line arguments.
    flags = parse_args(args)

    # Load data.
    df = pd.read_csv(flags.infile)

    avg_speed = None
    segments = None
    times = None
    downstream_path = flags.downstream_path
    if downstream_path is not None:
        with open(os.path.join(downstream_path, "segments.json"), "r") as f:
            segments = json.load(f)

        times = sorted(list(pd.read_csv(
            os.path.join(downstream_path, "speed.csv"))["time"]))

        avg_speed = np.genfromtxt(
            os.path.join(downstream_path, "speed.csv"),
            delimiter=",", skip_header=1)[:, 1:]

    av_ids = [x for x in np.unique(df.id) if "av" in x]
    energy_model = PFMCompactSedan()

    data = []
    for av_id in av_ids:
        # Extract data for this AV.
        df_i = df[df.id == av_id]
        df_i = df_i.sort_values("time")

        pos = np.array(df_i.pos)
        headway = np.array(df_i.headway)
        accel = np.array(df_i.target_accel_no_noise_no_failsafe)
        realized_accel = np.array(df_i.accel)
        speed = np.array(df_i.speed)
        leader_speed = np.array(df_i.leader_speed)

        for t in range(len(headway) - 1):
            s = obs(
                pos=pos,
                headway=headway,
                speed=speed,
                leader_speed=leader_speed,
                avg_speed=avg_speed,
                segments=segments,
                times=times,
                t=t,
                n=N_DOWNSTREAM,
            )

            s2 = obs(
                pos=pos,
                headway=headway,
                speed=speed,
                leader_speed=leader_speed,
                avg_speed=avg_speed,
                segments=segments,
                times=times,
                t=t+1,
                n=N_DOWNSTREAM,
            )

            a = action(
                accel=accel,
                t=t,
            )

            r = reward_fn(
                headway=headway,
                accel=accel,
                realized_accel=realized_accel,
                speed=speed,
                t=t,
                energy_model=energy_model,
            )

            # Convert data to correct format.
            data.append((s, a, r, s2))

    # Save new data.
    with open(flags.outfile, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main(sys.argv[1:])
