import torch
from .policy import Policy

class DiagonalGaussianPolicy(Policy):
    def __init__(self, env, lr=1e-2, hidden_sizes=[64], activation=torch.nn.ReLU):
        '''
        env (gym.Env): the environment
        lr (float): learning rate
        hidden_sizes (list of int): sizes of hidden layers
        activation (callable): activation function to use in the neural network, default is torch.nn.ReLU
        '''
        self.N = env.observation_space.shape[0]
        self.M = env.action_space.shape[0]

        # Construct the layers of the MLP
        layers = []
        input_size = self.N
        for size in hidden_sizes:
            layers.append(torch.nn.Linear(input_size, size))
            layers.append(activation())  # Dynamically use the specified activation function
            input_size = size
        # Output layer
        layers.append(torch.nn.Linear(input_size, self.M))

        self.mu = torch.nn.Sequential(*layers).double()

        # Initialize weights and biases to 0 for the output layer
        with torch.no_grad():
            self.mu[-1].weight.fill_(0)
            self.mu[-1].bias.fill_(0)

        # Log-sigma initialization
        self.log_sigma = torch.ones(self.M, dtype=torch.double, requires_grad=True)

        self.opt = torch.optim.Adam(list(self.mu.parameters()) + [self.log_sigma], lr=lr)

    def pi(self, s_t):
        '''
        returns the probability distribution over actions
        s_t (np.ndarray): the current state
        '''
        s_t = torch.as_tensor(s_t).double()
        mu = self.mu(s_t)
        log_sigma = self.log_sigma
        sigma = torch.exp(log_sigma)
        pi = torch.distributions.MultivariateNormal(mu, torch.diag(sigma))
        return pi
