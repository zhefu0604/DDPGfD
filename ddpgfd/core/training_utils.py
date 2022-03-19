"""TODO."""
import numpy as np
import errno
import torch
import pickle
import os
import matplotlib
import prodict
import yaml
import logging
from shutil import copy2

matplotlib.use('Agg')

np.set_printoptions(suppress=True, precision=5)


def load_conf(path):
    """Load data from a yaml file."""
    with open(path, 'r') as stream:
        yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)
    return prodict.Prodict.from_dict(yaml_dict)


def check_path(path):
    """Try to create a directory, and raise and error if failed."""
    try:
        os.makedirs(path)  # Support multi-level
        print(path + ' created')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class GaussianActionNoise(object):
    """TODO.

    Attributes
    ----------
    sigma : TODO
        TODO
    """

    def __init__(self, sigma, ac_dim=1):
        """TODO.

        Attributes
        ----------
        sigma : TODO
            TODO
        """
        self.ac_dim = ac_dim
        self.sigma = sigma

    def __call__(self):
        """TODO."""
        return np.random.normal(loc=0., scale=self.sigma, size=(self.ac_dim,))

    def reset(self):
        """TODO."""
        pass


class OrnsteinUhlenbeckActionNoise:
    """TODO.

    Attributes
    ----------
    theta : TODO
        TODO
    mu : TODO
        TODO
    sigma : TODO
        TODO
    dt : TODO
        TODO
    x0 : TODO
        TODO
    x_prev : TODO
        TODO
    """

    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        """TODO.

        Attributes
        ----------
        theta : TODO
            TODO
        mu : TODO
            TODO
        sigma : TODO
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


class TrainingProgress(object):
    """TODO.

    Attributes
    ----------
    result_path : TODO
        TODO
    progress_path : TODO
        TODO
    meta_dict : dict
        TODO
    record_dict : TODO
        TODO
    logger : TODO
        TODO
    """

    def __init__(self,
                 result_path,
                 folder_name,
                 tp_step=None,
                 meta_dict=None,
                 record_dict=None,
                 restore=False):
        """TODO.

        Header => Filename header,append file name behind
        Data Dict => Appendable data (loss,time,acc....)
        Meta Dict => One time data (config,weight,.....)

        Parameters
        ----------
        result_path : TODO
            TODO
        folder_name : TODO
            TODO
        tp_step : TODO
            TODO
        meta_dict : TODO
            TODO
        record_dict : TODO
            TODO
        restore : TODO
            TODO
        """
        self.result_path = os.path.join(result_path, folder_name) + "/"
        self.progress_path = os.path.join(
            result_path, folder_name, "model-params") + "/"
        check_path(self.progress_path)
        check_path(self.result_path)

        if restore:
            assert tp_step is not None, \
                'Explicitly assign the TP step you want to restore'
            self.restore_progress(tp_step)
        else:
            self.meta_dict = meta_dict or {}  # one time values
            self.record_dict = record_dict or {}  # Recommend, record with step

        self.logger = logging.getLogger('TP')

    def add_meta(self, new_dict):
        """Update the internal meta_dict with new data."""
        self.meta_dict.update(new_dict)

    def get_meta(self, key):
        """Return an element from meta_dict."""
        try:
            return self.meta_dict[key]
        except KeyError:  # New key
            self.logger.error('TP Error: Cannot find meta, key={}'.format(key))
            return None

    def record_step(self, epoch, prefix, new_dict, display=False):
        """TODO.

        Parameters
        ----------
        epoch : TODO
            TODO
        prefix : TODO
            TODO
        new_dict : TODO
            TODO
        display : TODO
            TODO
        """
        # record every epoch, prefix=train/test/validation....
        key = prefix + str(epoch)
        if key in self.record_dict.keys():
            self.record_dict[key].update(new_dict)
        else:
            self.record_dict[key] = new_dict
        if display:
            str_display = ''
            for k, v in new_dict.items():
                if isinstance(v, float):
                    str_display += k + ': {:0.5f}, '.format(v)
                else:
                    str_display += k + ': ' + str(v) + ', '
            self.logger.info(key + ': ' + str_display)

    def save_progress(self, tp_step, override_path=None):
        """TODO.

        Parameters
        ----------
        tp_step : TODO
            TODO
        override_path : TODO
            TODO
        """
        name = self.progress_path + str(tp_step) + '.tpdata' \
            if override_path is None else override_path
        check_path(os.path.dirname(name))
        with open(name, "wb") as f:
            pickle.dump((self.meta_dict, self.record_dict), f, protocol=2)

    def restore_progress(self, tp_step, override_path=None):
        """TODO.

        Parameters
        ----------
        tp_step : TODO
            TODO
        override_path : TODO
            TODO
        """
        name = self.progress_path + str(tp_step) + '.tpdata' \
            if override_path is None else override_path
        with open(name, 'rb') as f:
            self.meta_dict, self.record_dict = pickle.load(f)

    def backup_file(self, src, file_name):  # Saved in result
        """TODO.

        Parameters
        ----------
        src : TODO
            TODO
        file_name : TODO
            TODO
        """
        self.logger.info('Backup ' + src)
        copy2(src, self.result_path + file_name)

    def save_conf(self, dict, prefix=''):
        """TODO.

        Parameters
        ----------
        dict : TODO
            TODO
        prefix : TODO
            TODO
        """
        path = self.result_path + prefix + 'conf.yaml'
        with open(path, 'w') as outfile:
            yaml.dump(dict, outfile)
