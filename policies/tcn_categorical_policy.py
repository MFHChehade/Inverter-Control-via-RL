import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .policy import Policy

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNCategoricalPolicy(Policy):
    def __init__(self, env, lr=1e-2, num_channels=[25, 50, 50, 25], kernel_size=3, dropout=0.05):
        '''
        env (gym.Env): the environment
        lr (float): learning rate
        num_channels (list of int): Number of channels for each layer of the TCN
        kernel_size (int): Size of the kernel in the convolutions
        '''
        super(TCNCategoricalPolicy, self).__init__()
        self.N = env.observation_space.shape[0]
        self.M = env.action_space.n

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.N if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], self.M)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def pi(self, s_t):
        '''
        returns the probability distribution over actions
        s_t (np.ndarray): the current state, expected shape (batch_size, sequence_length, features)
        '''
        s_t = torch.as_tensor(s_t).double().unsqueeze(0)  # Add batch dimension if not present
        s_t = s_t.transpose(1, 2)  # TCN needs batch_size, channels, length
        tcn_out = self.tcn(s_t)
        tcn_out = tcn_out[:, :, -1]  # only use the last timestep's output
        logits = self.linear(tcn_out)
        pi = torch.distributions.Categorical(logits=logits)
        return pi
