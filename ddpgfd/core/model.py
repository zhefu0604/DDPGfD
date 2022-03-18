"""TODO."""
import torch.nn as nn
import numpy as np
import torch.nn.init as init

# number of nodes per hidden layer. There are 2 hidden layers.
HIDDEN_LAYERS = 256


def init_fanin(tensor):
    """TODO."""
    fanin = tensor.size(1)
    v = 1.0 / np.sqrt(fanin)
    init.uniform(tensor, -v, v)


class ActorNet(nn.Module):
    """TODO.

    Attributes
    ----------
    device : TODO
        TODO
    net : TODO
        TODO
    """

    def __init__(self, in_dim, out_dim, device):
        """TODO.

        Parameters
        ----------
        in_dim : TODO
            TODO
        out_dim : TODO
            TODO
        device : TODO
            TODO
        """
        super(ActorNet, self).__init__()
        self.device = device

        # Create the network.
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_LAYERS, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS, out_dim),
            nn.Tanh(),
        )  # +-1 output

        # Initialize weights and biases.
        for i, x in enumerate(self.net.modules()):
            if i in [1, 4]:
                init_fanin(x.weight)
            if i == 7:
                nn.init.uniform_(x.weight, -3e-3, 3e-3)
                nn.init.uniform_(x.bias, -3e-3, 3e-3)

    def forward(self, state):
        """Run a forward pass of the actor.

        Parameters
        ----------
        state : TODO
            the input state, (N,in_dim)

        Returns
        -------
        TODO
            deterministic action, (N,out_dim)
        """
        action = self.net(state)
        return action


class CriticNet(nn.Module):
    """TODO.

    Attributes
    ----------
    device : TODO
        TODO
    net : TODO
        TODO
    """

    def __init__(self, s_dim, a_dim, device):
        """TODO.

        Parameters
        ----------
        s_dim : TODO
            TODO
        a_dim : TODO
            TODO
        device : TODO
            TODO
        """
        super(CriticNet, self).__init__()
        self.device = device
        in_dim = s_dim + a_dim

        # Create the network.
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_LAYERS, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS, 1),
        )

        # Initialize weights and biases.
        for i, x in enumerate(self.net.modules()):
            if i in [1, 4]:
                init_fanin(x.weight)
            if i == 7:
                nn.init.uniform_(x.weight, -3e-3, 3e-3)
                nn.init.uniform_(x.bias, -3e-3, 3e-3)

    def forward(self, sa_pairs):
        """Run a forward pass of the critic.

        Parameters
        ----------
        sa_pairs : TODO
            state-action pairs, (N, in_dim)

        Returns
        -------
        TODO
            Q-values , (N,1)
        """
        return self.net(sa_pairs)
