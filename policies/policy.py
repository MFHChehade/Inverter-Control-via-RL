import torch
import numpy as np

class Policy:
    def __init__(self, action_dim):
        """
        Initialize the Policy with the dimensionality of the action space.
        
        action_dim (int): Dimensionality of the action space.
        """
        self.action_dim = action_dim
        
        # Example actor network: mean and log_std parameters
        self.actor_mean = torch.nn.Linear(10, action_dim)  # Replace input size (10) with state_dim
        self.actor_log_std = torch.nn.Parameter(torch.zeros(action_dim))  # Learnable log std
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)  # Optimizer for the policy

    def parameters(self):
        """
        Gather parameters for optimization.
        """
        return list(self.actor_mean.parameters()) + [self.actor_log_std]

    def pi(self, s_t):
        """
        Returns a multivariate Gaussian distribution over actions.

        s_t (np.ndarray): The current state, shape: (batch_size, state_dim).
        """
        s_t = torch.tensor(s_t, dtype=torch.float32)  # Convert state to tensor
        mean = self.actor_mean(s_t)  # Predicted mean from the actor network
        log_std = self.actor_log_std.expand_as(mean)  # Expand log std to match mean's shape
        std = torch.exp(log_std)  # Convert log std to standard deviation
        return torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))

    def act(self, s_t):
        """
        Sample actions and compute log probabilities.
        
        s_t (np.ndarray): The current state.
        Returns:
        - a_t (np.ndarray): Sampled actions, shape: (batch_size, action_dim).
        - log_prob (np.ndarray): Log probabilities of the sampled actions, shape: (batch_size,).
        """
        pi = self.pi(s_t)
        a_t = pi.sample()
        log_prob = pi.log_prob(a_t).detach().numpy()
        return a_t.numpy(), log_prob

    def learn(self, states, actions, advantages, log_probs_old=None, epsilon=None, method="PPO"):
        """
        Learn by updating policy parameters. Supports both PPO and A2C.

        Args:
        - states (np.ndarray): List of states, shape: (T, E, state_dim).
        - actions (np.ndarray): List of actions, shape: (T, E, action_dim).
        - advantages (np.ndarray): List of advantages, shape: (T, E).
        - log_probs_old (np.ndarray): Log probabilities of the actions (PPO), shape: (T, E).
        - epsilon (float): PPO clipping threshold.
        - method (str): "PPO" or "A2C", specifying the training method.
        """
        actions = torch.tensor(actions, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        if method == "PPO":
            # Ensure required parameters for PPO are provided
            if log_probs_old is None or epsilon is None:
                raise ValueError("log_probs_old and epsilon must be provided for PPO training.")

            log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
            pi = self.pi(states)
            log_probs = pi.log_prob(actions)

            # PPO ratio
            r_theta = torch.exp(log_probs - log_probs_old)

            # Clipped surrogate loss
            clipped = torch.where(
                advantages > 0,
                torch.min(r_theta, torch.tensor(1 + epsilon, dtype=r_theta.dtype, device=r_theta.device)),
                torch.max(r_theta, torch.tensor(1 - epsilon, dtype=r_theta.dtype, device=r_theta.device))
            )
            loss = torch.mean(-clipped * advantages)

        elif method == "A2C":
            # A2C uses the simple policy gradient loss
            pi = self.pi(states)
            log_probs = pi.log_prob(actions)
            loss = torch.mean(-log_probs * advantages)

        else:
            raise ValueError(f"Unsupported method: {method}. Use 'PPO' or 'A2C'.")

        # Backpropagation
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Return an optional metric (e.g., approximate KL for PPO)
        if method == "PPO":
            approx_kl = (log_probs_old - log_probs).mean().item()
            return approx_kl
        return None
