"""Generate demonstrations from an emission file."""
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

from trajectory.env.energy_models import PFMCompactSedan

# number of backward timesteps to add to observation
N_STEPS = 5
# headway scaling term
HEADWAY_SCALE = 100.
# speed scaling term
SPEED_SCALE = 40.


def parse_args(args):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate demonstrations from an emission file.",
        epilog="python generate_video.py INFILE OUTFILE")

    parser.add_argument(
        'infile', type=str, help='the path to the emission file')
    parser.add_argument(
        'outfile', type=str, help='the path to save the demonstrations in')

    return parser.parse_args(args)


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


def reward_fn(headway, accel, realized_accel, speed, leader_speed, t,
              energy_model):
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
    leader_speed : array_like
        list of vehicle lead speeds over time
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

    # reward function parameters
    scale = 0.1
    h_low = -1
    h_high = 150
    th_low = 2.0
    th_high = -1
    energy_steps = 50

    # time headway reward
    th = min(headway[t] / max(speed[t], 0.01), 20)
    if th_low > 0 and th < th_low:
        reward -= 2 * huber_loss(th - th_low, delta=1.00)
    if th_high > 0 and th > th_high:
        reward -= huber_loss(th - th_high, delta=0.25)

    # space headway reward
    h = min(headway[t], 300)
    if h_low > 0 and h < h_low:
        reward -= 2 * huber_loss((h - h_low) / 10., delta=1.00)
    if h_high > 0 and h > h_high:
        reward -= huber_loss((h - h_high) / 10., delta=0.25)

    # acceleration reward
    # reward -= 0.1 * realized_accel[t] ** 2

    # failsafe reward
    # reward -= (accel[t] - realized_accel[t]) ** 2

    # energy consumption reward
    sum_energy = sum([
        energy_model.get_instantaneous_fuel_consumption(
            speed=speed_i, accel=accel_i, grade=0)
        for speed_i, accel_i in zip(speed[max(t-energy_steps+1, 0): t+1],
                                    accel[max(t-energy_steps+1, 0): t+1])])

    sum_speed = sum(np.clip(
        speed[max(t-energy_steps+1, 0): t+1], a_min=0.1, a_max=np.inf))

    reward -= 2.5 * min(sum_energy / sum_speed, 0.2)

    return scale * reward


def obs(headway, accel, speed, leader_speed, t):
    """Compute the observation at a specific time index.

    Parameters
    ----------
    headway : array_like
        list of vehicle headways over time
    accel : array_like
        list of vehicle accelerations over time
    speed : array_like
        list of vehicle speed over time
    leader_speed : array_like
        list of vehicle lead speeds over time
    t : int
        the time index

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
        list(headway[min_t: max_t] / HEADWAY_SCALE))


def action(headway, accel, speed, leader_speed, t):
    """Compute the action at a specific time index.

    Parameters
    ----------
    headway : array_like
        list of vehicle headways over time
    accel : array_like
        list of vehicle accelerations over time
    speed : array_like
        list of vehicle speed over time
    leader_speed : array_like
        list of vehicle lead speeds over time
    t : int
        the time index

    Returns
    -------
    array_like
        the observation
    """
    return np.array([2. * accel[t]])


def main(args):
    """Run the main operation."""
    # Parse command-line arguments.
    flags = parse_args(args)

    # Load data.
    df = pd.read_csv(flags.infile)

    av_ids = [x for x in np.unique(df.id) if "av" in x]
    energy_model = PFMCompactSedan()

    data = []
    for av_id in av_ids:
        # Extract data for this AV.
        df_i = df[df.id == av_id]
        df_i = df_i.sort_values("time")

        headway = np.array(df_i.headway)
        accel = np.array(df_i.target_accel_no_noise_no_failsafe)
        realized_accel = np.array(df_i.accel)
        speed = np.array(df_i.speed)
        leader_speed = np.array(df_i.leader_speed)

        for t in range(len(headway) - 1):
            s = obs(
                headway=headway,
                accel=accel,
                speed=speed,
                leader_speed=leader_speed,
                t=t,
            )

            s2 = obs(
                headway=headway,
                accel=accel,
                speed=speed,
                leader_speed=leader_speed,
                t=t + 1,
            )

            a = action(
                headway=headway,
                accel=accel,
                speed=speed,
                leader_speed=leader_speed,
                t=t,
            )

            r = reward_fn(
                headway=headway,
                accel=accel,
                realized_accel=realized_accel,
                speed=speed,
                leader_speed=leader_speed,
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
