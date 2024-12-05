import torch
from .policy import Policy

class LSTMCategoricalPolicy(Policy):
    def __init__(self, env, lr=1e-2, hidden_size=128):
        '''
        env (gym.Env): the environment
        lr (float): learning rate
        hidden_size (int): size of the LSTM hidden layer
        '''
        self.N = env.observation_space.shape[0]
        self.M = env.action_space.n
        self.hidden_size = hidden_size

        # LSTM layer for processing sequences
        self.lstm = torch.nn.LSTM(self.N, hidden_size, batch_first=True).double()
        # Linear layer to map from LSTM output to action logits
        self.linear = torch.nn.Linear(hidden_size, self.M).double()

        # Initialize the optimizer
        self.opt = torch.optim.Adam(list(self.lstm.parameters()) + list(self.linear.parameters()), lr=lr)

        # Hidden state and cell state initialization
        self.hidden = None

    def reset_hidden_state(self):
        # Reset the hidden and cell state at the start of each episode
        self.hidden = None

    def pi(self, s_t):
        '''
        returns the probability distribution over actions
        s_t (np.ndarray): the current state, expected shape (sequence_length, features) or (batch_size, sequence_length, features)
        '''
        s_t = torch.as_tensor(s_t).double()
        if s_t.dim() == 2:
            s_t = s_t.unsqueeze(0)  # Add batch dimension if not present

        if self.hidden is None:
            # Initialize hidden state with zeros
            h0 = torch.zeros(1, s_t.size(0), self.hidden_size).double()  # Adjusted to handle dynamic batch size
            c0 = torch.zeros(1, s_t.size(0), self.hidden_size).double()
            self.hidden = (h0, c0)
        
        # Forward pass through LSTM
        lstm_out, self.hidden = self.lstm(s_t, self.hidden)

        # Only take the output from the last time step
        last_time_step_out = lstm_out[:, -1, :]
        
        # Forward pass through the linear layer
        logits = self.linear(last_time_step_out)

        # Create a categorical distribution from the logits
        pi = torch.distributions.Categorical(logits=logits)
        return pi
