import numpy as np
import time
import math
import errno
import torch
import pickle
import os
import matplotlib
import prodict
import yaml
import logging
import matplotlib.pyplot as plt
import torch.nn.init as init
from torch import nn
from shutil import copy2
from collections import OrderedDict
from torch.utils.data import Dataset

matplotlib.use('Agg')

np.set_printoptions(suppress=True, precision=5)


def load_conf(path):
    """Load data from a yaml file."""
    with open(path, 'r') as stream:
        yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)
    return prodict.Prodict.from_dict(yaml_dict)


def time_since(since, return_seconds=False):
    """Return the elapsed time (in MM SS)."""
    now = time.time()
    s = now - since
    if return_seconds:
        return s
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def check_path(path):
    """Try to create a directory, and raise and error if failed."""
    try:
        os.makedirs(path)  # Support multi-level
        print(path + ' created')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class TrainingProgress:

    def __init__(self,
                 progress_path,
                 result_path,
                 folder_name,
                 tp_step=None,
                 meta_dict=None,
                 record_dict=None,
                 restore=False):
        """
        Header => Filename header,append file name behind
        Data Dict => Appendable data (loss,time,acc....)
        Meta Dict => One time data (config,weight,.....)
        """
        self.progress_path = os.path.join(progress_path, folder_name) + '/'
        self.result_path = os.path.join(result_path, folder_name) + '/'
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

    def save_model_weight(self, model, epoch, prefix=''):
        name = self.progress_path + prefix + 'model-' + str(epoch) + '.tp'
        torch.save(model.state_dict(), name)

    def restore_model_weight(self, epoch, device, prefix=''):
        name = self.progress_path + prefix + 'model-' + str(epoch) + '.tp'
        return torch.load(name, map_location=device)

    def add_meta(self, new_dict):
        self.meta_dict.update(new_dict)

    def get_meta(self, key):
        try:
            return self.meta_dict[key]
        except KeyError:  # New key
            self.logger.error('TP Error: Cannot find meta, key={}'.format(key))
            return None

    def record_step(self, epoch, prefix, new_dict, display=False):  # use this
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

    def get_step_data(self, data_key, prefix, ep_start, ep_end, ep_step=1):
        data = []
        for ep in range(ep_start, ep_end, ep_step):
            key = prefix + str(ep)
            try:
                data.append(self.record_dict[key][data_key])
            except KeyError:
                self.logger.warning(
                    'TP Warning, Invalid epoch={}, Data Ignored!'.format(ep))
        return data

    def get_step_data_all(self, prefix, ep_start, ep_end, ep_step=1):
        ep_end += 1
        data_keys = list(self.record_dict[prefix + str(ep_start)].keys())
        data_keys.sort()  # Item keys
        append_dict = OrderedDict()
        for ep in range(ep_start, ep_end, ep_step):
            key = prefix + str(ep)
            for k, v in self.record_dict[key].items():
                try:
                    append_dict[k].append(v)
                except KeyError:
                    append_dict[k] = [v]
        return append_dict

    def save_progress(self, tp_step, override_path=None):
        name = self.progress_path + str(tp_step) + '.tpdata' \
            if override_path is None else override_path
        check_path(os.path.dirname(name))
        with open(name, "wb") as f:
            pickle.dump((self.meta_dict, self.record_dict), f, protocol=2)

    def restore_progress(self, tp_step, override_path=None):
        name = self.progress_path + str(tp_step) + '.tpdata' \
            if override_path is None else override_path
        with open(name, 'rb') as f:
            self.meta_dict, self.record_dict = pickle.load(f)

    def plot_data(self,
                  prefix,
                  ep_start,
                  ep_end,
                  file_name,
                  title,
                  ep_step=1,
                  grid=True):
        ep_end += 1
        data_keys = list(self.record_dict[prefix + str(ep_start)].keys())
        data_keys.sort()  # Item keys
        append_dict = {}
        for ep in range(ep_start, ep_end, ep_step):
            key = prefix + str(ep)
            for k, v in self.record_dict[key].items():
                try:
                    append_dict[k].append(v)
                except KeyError:
                    append_dict[k] = [v]
        n_cols = 3
        n_rows = int(len(data_keys) / n_cols + 1)
        fig = plt.figure(dpi=800, figsize=(n_cols * 3, n_rows * 3))
        fig.suptitle(title)
        x_ticks = list(range(ep_start, ep_end, ep_step))
        keys = sorted(append_dict.keys())
        # for i, (k, v) in enumerate(append_dict.items()):
        for i, k in enumerate(keys):
            v = append_dict[k]
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            if grid:
                ax.grid(True)
            ax.plot(x_ticks, v)
            ax.set_xticks(x_ticks)
            ax.xaxis.set_tick_params(labelsize=4)
            ax.set_title(k)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(self.result_path + file_name)
        plt.clf()
        plt.close(fig)

    def plot_data_overlap(self,
                          prefix,
                          ep_start,
                          ep_end,
                          file_name,
                          title,
                          ep_step=1,
                          keys=None):
        ep_end += 1
        data_keys = list(self.record_dict[prefix + str(ep_start)].keys())
        data_keys.sort()  # Item keys
        append_dict = {}
        for ep in range(ep_start, ep_end, ep_step):
            key = prefix + str(ep)
            for k, v in self.record_dict[key].items():
                try:
                    append_dict[k].append(v)
                except KeyError:
                    append_dict[k] = [v]
        if keys is not None:
            append_dict = {k: append_dict[k] for k in keys}
        fig = plt.figure(dpi=800, figsize=(6, 3))
        fig.suptitle(title)
        x_ticks = list(range(ep_start, ep_end, ep_step))
        keys = sorted(append_dict.keys())
        # for i, (k, v) in enumerate(append_dict.items()):
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        # ax.set_xticks(x_ticks)
        ax.xaxis.set_tick_params(labelsize=4)
        for i, k in enumerate(keys):
            v = append_dict[k]
            # if i == 0:
            #     ax.plot(x_ticks, v, '--', label=k, linewidth=1)
            # else:
            ax.plot(x_ticks, v, label=k, linewidth=1)
        ax.legend()
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(self.result_path + file_name)
        plt.clf()
        plt.close(fig)

    def backup_file(self, src, file_name):  # Saved in result
        self.logger.info('Backup ' + src)
        copy2(src, self.result_path + file_name)

    def save_conf(self, dict, prefix=''):
        path = self.result_path + prefix + 'conf.yaml'
        with open(path, 'w') as outfile:
            yaml.dump(dict, outfile)


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
