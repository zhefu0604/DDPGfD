"""Various utility methods when training a policy."""
import numpy as np
import errno
import pickle
import os
import prodict
import yaml
import logging
from shutil import copy2
from torch.nn import functional as F
from torch.autograd import Variable
import tqdm
from copy import deepcopy


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


class ActionNoise(object):
    """Base action noise object. Used for exploration purposes."""

    def __call__(self):
        """Run a forward pass of the object."""
        raise NotImplementedError

    def reset(self):
        """Reset the object between rollouts."""
        raise NotImplementedError


class GaussianActionNoise(ActionNoise):
    """Gaussian action noise."""

    def __init__(self, std, ac_dim=1):
        """Instantiate the noise object.

        Parameters
        ----------
        std : float
            standard deviation of Gaussian noise
        """
        self.ac_dim = ac_dim
        self.std = std

    def __call__(self):
        """See parent class."""
        return np.random.normal(loc=0., scale=self.std, size=(self.ac_dim,))

    def reset(self):
        """See parent class."""
        pass


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """Ornstein-Uhlenbeck action noise."""

    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        """Instantiate the noise object."""
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.x_prev = None
        self.reset()

    def __call__(self):
        """See parent class."""
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        """See parent class."""
        self.x_prev = self.x0 or np.zeros_like(self.mu)


class EWC(object):
    """Elastic Weight Consolidation module.

    See: https://www.pnas.org/doi/10.1073/pnas.1611835114
    """

    def __init__(self, model, dataset):
        """Instantiate the EWC module.

        Parameters
        ----------
        model : nn.Module
            the source model
        dataset : list of array_like
            a list of states from the old task
        """
        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if
                       p.requires_grad}
        self._means = {}
        # self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = Variable(p.data)

    def _diag_fisher(self):
        """Compute the diagonal fisher information matrix of the model."""
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = Variable(p.data)

        self.model.eval()
        for x in tqdm.tqdm(self.dataset):
            self.model.zero_grad()
            x = Variable(x)
            output = self.model(x)
            print(output)
            label = output.max(1)[1].view(-1)
            print(label)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            from torch import autograd
            # loss.backward()

            for n, p in self.model.named_parameters():
                print(np.sum(autograd.grad(loss, p)[0].detach().numpy()))
                precision_matrices[n].data += p.grad.data ** 2 / len(
                    self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        """Return the EWC penalty for the current model."""
        loss = 0
        for n, p in model.named_parameters():
            # _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            _loss = (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


class TrainingProgress(object):
    """A logger for training progress."""

    def __init__(self,
                 result_path,
                 folder_name,
                 tp_step=None,
                 meta_dict=None,
                 record_dict=None,
                 restore=False):
        """Instantiate the object.

        Parameters
        ----------
        result_path : str
            the higher-level path to save data in
        folder_name : str
            the name of the folder to save data in
        tp_step : int or None
            the training epoch
        meta_dict : dict or None
            One time data (config,weight,.....)
        record_dict : dict or None
            Appendable data (loss,time,acc....)
        restore : bool
            whether to restore previous data
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
        """Record new training parameters.

        Parameters
        ----------
        epoch : int
            the current training epoch
        prefix : str
            a prefix to add to the file name
        new_dict : dict
            dictionary of parameters to log
        display : bool
            whether to print to terminal once the operation is complete
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
        """Save the meta and record dict for the current epoch.

        Parameters
        ----------
        tp_step : int
            the training epoch to save data to
        override_path : str or None
            the path to save data tp. If set to None, the epoch is used to
            estimate the path
        """
        name = self.progress_path + str(tp_step) + '.tpdata' \
            if override_path is None else override_path
        check_path(os.path.dirname(name))
        with open(name, "wb") as f:
            pickle.dump((self.meta_dict, self.record_dict), f, protocol=2)

    def restore_progress(self, tp_step, override_path=None):
        """Restore meta_dict and record_dict from a given path.

        Parameters
        ----------
        tp_step : int
            the training epoch to restore data from
        override_path : str or None
            the path to restore data from. If set to None, the epoch is used to
            estimate the path
        """
        name = self.progress_path + str(tp_step) + '.tpdata' \
            if override_path is None else override_path
        with open(name, 'rb') as f:
            self.meta_dict, self.record_dict = pickle.load(f)

    def backup_file(self, src, file_name):
        """Save in result.

        Parameters
        ----------
        src : str
            the source path to copy data from
        file_name : str
            the path to save results in
        """
        self.logger.info('Backup ' + src)
        copy2(src, self.result_path + file_name)

    def save_conf(self, new_dict, prefix=''):
        """Save training parameters in a conf.yaml file.

        Parameters
        ----------
        new_dict : dict
            dictionary of training parameters
        prefix : str
            a prefix to add to the file name
        """
        path = self.result_path + prefix + 'conf.yaml'
        with open(path, 'w') as outfile:
            yaml.dump(new_dict, outfile)
