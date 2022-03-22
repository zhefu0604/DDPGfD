"""Script containing the actor and critic models."""
import torch.nn as nn
import numpy as np
import torch.nn.init as init

# number of nodes per hidden layer. There are 2 hidden layers.
HIDDEN_LAYERS = 256


def init_fanin(tensor):
    """Return fan-in initial parameters."""
    fanin = tensor.size(1)
    v = 1.0 / np.sqrt(fanin)
    init.uniform(tensor, -v, v)


class ActorNet(nn.Module):
    """Actor Network Module.

    Attributes
    ----------
    device : torch.device
        context-manager that changes the selected device.
    net : torch.nn.Sequential
        the network model
    """

    def __init__(self, in_dim, out_dim, device):
        """Instantiate the actor network.

        Parameters
        ----------
        in_dim : int
            number of elements in the state space
        out_dim : int
            number of elements in the action space
        device : torch.device
            context-manager that changes the selected device.
        """
        super(ActorNet, self).__init__()
        self.device = device

        # Create the network.
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_LAYERS),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS),
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

    def forward(self, state):
        """Run a forward pass of the actor.

        Parameters
        ----------
        state : torch.Tensor
            the input state, (N,in_dim)

        Returns
        -------
        torch.Tensor
            deterministic action, (N,out_dim)
        """
        action = self.net(state)
        return action


class CriticNet(nn.Module):
    """Critic Network Module.

    Attributes
    ----------
    device : torch.device
        context-manager that changes the selected device.
    q1 : torch.nn.Sequential
        the Q1 network model
    q2 : torch.nn.Sequential
        the Q1 network model
    """

    def __init__(self, s_dim, a_dim, device):
        """Instantiate the critic network.

        Parameters
        ----------
        s_dim : int
            number of elements in the state space
        a_dim : int
            number of elements in the action space
        device : torch.device
            context-manager that changes the selected device.
        """
        super(CriticNet, self).__init__()
        self.device = device
        in_dim = s_dim + a_dim

        # Create the network.
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS, 1),
        )

        # Initialize weights and biases.
        for i, x in enumerate(self.q1.modules()):
            if i in [1, 3]:
                init_fanin(x.weight)
            if i == 5:
                nn.init.uniform_(x.weight, -3e-3, 3e-3)

        for i, x in enumerate(self.q2.modules()):
            if i in [1, 3]:
                init_fanin(x.weight)
            if i == 5:
                nn.init.uniform_(x.weight, -3e-3, 3e-3)

    def forward(self, sa_pairs):
        """Run a forward pass of the critic.

        Parameters
        ----------
        sa_pairs : torch.Tensor
            state-action pairs, (N, in_dim)

        Returns
        -------
        torch.Tensor
            Q1 values, (N, 1)
        torch.Tensor
            Q2 values, (N, 1)
        """
        return self.q1(sa_pairs), self.q2(sa_pairs)

    def Q1(self, sa_pairs):
        """Return the Q-value of the first critic.

        Parameters
        ----------
        sa_pairs : torch.Tensor
            state-action pairs, (N, in_dim)

        Returns
        -------
        torch.Tensor
            Q-values, (N, 1)
        """
        return self.q1(sa_pairs)
