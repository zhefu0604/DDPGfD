"""Script containing the base DDPGfD agent class."""
import torch
import torch.nn as nn
import logging

# constant signifying that data was collected from a demonstration
DATA_DEMO = 0
# constant signifying that data was collected during training
DATA_RUNTIME = 1


class DDPGfDAgent(nn.Module):
    """TODO.

    Attributes
    ----------
    conf : Any
        full configuration parameters
    agent_conf : Any
        agent configuration parameters
    device : torch.device
        context-manager that changes the selected device.
    state_dim : int
        number of elements in the state space
    action_dim : int
        number of elements in the action space
    logger : TODO
        TODO
    """

    def __init__(self, conf, device, state_dim, action_dim):
        """Instantiate the agent class.

        Parameters
        ----------
        conf : Any
            full configuration parameters
        device : torch.device
            context-manager that changes the selected device.
        state_dim : int
            number of elements in the state space
        action_dim : int
            number of elements in the action space
        """
        super(DDPGfDAgent, self).__init__()

        self.conf = conf
        self.agent_conf = self.conf.agent_config
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.logger = logging.getLogger('DDPGfD')

    def initialize(self):
        """Initialize the agent."""
        raise NotImplementedError

    def reset(self):
        """Reset action noise."""
        raise NotImplementedError

    def get_action(self, s_tensor, n_agents):
        """Compute noisy action.

        Parameters
        ----------
        s_tensor : TODO
            TODO
        n_agents : TODO
            TODO
        """
        raise NotImplementedError

    def add_memory(self, s, a, s2, r, gamma, dtype):
        """TODO.

        Parameters
        ----------
        s : TODO
            TODO
        a : TODO
            TODO
        s2 : TODO
            TODO
        r : TODO
            TODO
        gamma : TODO
            TODO
        dtype : TODO
            TODO
        """
        raise NotImplementedError

    def update_agent(self, update_step):
        """Sample experience and update.

        Parameters
        ----------
        update_step : int
            number of policy updates to perform

        Returns
        -------
        float
            critic loss
        float
            actor loss
        int
            the number of demonstration in the training batches
        int
            the total number of samples in the training batches
        """
        raise NotImplementedError

    @staticmethod
    def obs2tensor(state):
        """Convert observations to a torch-compatible format."""
        return torch.from_numpy(state).float()
