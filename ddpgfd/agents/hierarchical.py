"""Script containing the hierarchical variant of the DDPGfD agent."""
import numpy as np
import torch
import torch.nn as nn
import os
import pickle

from ddpgfd.agents.base import DDPGfDAgent
from ddpgfd.agents.base import DATA_DEMO
from ddpgfd.core.model import ActorNet
from ddpgfd.core.model import CriticNet
from ddpgfd.core.replay_memory import PrioritizedReplayBuffer
from ddpgfd.core.training_utils import EWC
from ddpgfd.core.training_utils import GaussianActionNoise


class HierarchicalAgent(DDPGfDAgent):

    pass  # TODO
