"""Generate demonstrations from an emission file."""
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

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


def reward_fn(headway,
              accel,
              realized_accel,
              speed,
              leader_speed,
              instant_energy_consumption,
              t):
    """Compute the reward at a specific time index.

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
    instant_energy_consumption : array_like
        list of vehicle instantaneous energy consumption over time
    t : int
        the time index

    Returns
    -------
    float
        the reward
    """
    scale = 0.1
    low = 2
    high = 4
    energy_steps = 100
    reward = 0.

    th = max(min((headway[t] / max(speed[t], 1e-10)), 20), 0)

    # time headway reward
    if th < low:
        reward -= huber_loss(th - low, delta=1.00)
    elif th > high:
        reward -= huber_loss(th - high, delta=0.25)

    # failsafe reward
    reward -= (accel[t] - realized_accel[t]) ** 2

    # energy consumption reward
    reward -= np.mean(np.clip(
        instant_energy_consumption[max(t-energy_steps, 0): t+1],
        a_min=0, a_max=np.inf))

    return scale * reward


def obs(headway,
        accel,
        speed,
        leader_speed,
        instant_energy_consumption,
        t):
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
    instant_energy_consumption : array_like
        list of vehicle instantaneous energy consumption over time
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


def action(headway,
           accel,
           speed,
           leader_speed,
           instant_energy_consumption,
           t):
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
    instant_energy_consumption : array_like
        list of vehicle instantaneous energy consumption over time
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
        instant_energy_consumption = np.array(df_i.instant_energy_consumption)

        for t in range(len(headway) - 1):
            s = obs(
                headway,
                accel,
                speed,
                leader_speed,
                instant_energy_consumption,
                t,
            )

            s2 = obs(
                headway,
                accel,
                speed,
                leader_speed,
                instant_energy_consumption,
                t + 1,
            )

            a = action(
                headway,
                accel,
                speed,
                leader_speed,
                instant_energy_consumption,
                t,
            )

            r = reward_fn(
                headway,
                accel,
                realized_accel,
                speed,
                leader_speed,
                instant_energy_consumption,
                t,
            )

            # Convert data to correct format.
            data.append((s, a, r, s2))

    # Save new data.
    with open(flags.outfile, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main(sys.argv[1:])
