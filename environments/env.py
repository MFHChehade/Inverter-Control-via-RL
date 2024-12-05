import numpy as np
import gym
from gym import spaces

class InverterControlEnv(gym.Env):
    def __init__(self):
        super(InverterControlEnv, self).__init__()
        
        # System parameters
        self.V_nom = 120  # V
        self.S_nom = 1.5e3  # VA
        self.I_nom = 4.167  # A
        self.I_max = 4.167  # A
        self.E = 120  # V
        self.omega_nom = 2 * np.pi * 60  # rad/s
        self.R_sys = 1.3  # ohms
        self.L = 3.5e-3  # H
        self.Delta_t = 1e-4  # Time step

        # State matrices
        self.A = np.eye(2) + self.Delta_t * np.array([
            [-self.R_sys / self.L, self.omega_nom],
            [-self.omega_nom, -self.R_sys / self.L]
        ])
        self.B = self.Delta_t * np.array([
            [np.sqrt(2) / self.L, 0],
            [0, np.sqrt(2) * self.E / self.L]
        ])

        # Reference current and initial state
        self.x_t, self.x_star = self.generate_x_star_and_x_init_single(self.I_max)

        # Define action space (discrete version)
        min_action = np.array([0, 0])
        max_action = np.array([5, 3])

        # Generate discrete action space
        self.num_bins = 20  # Increase the number of discrete actions to 200
        # self.action_mapping = self.generate_action_mapping(min_action, max_action, self.num_bins)
        # self.action_space = spaces.Discrete(len(self.action_mapping))  # Total number of discrete actions

        # Uncomment this line for continuous action space instead
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Weight matrices for reward function
        self.Q = np.array([[1, 0], [0, 0.1]])  # State tracking error weight
        self.R = 5 * self.B  # Control effort weight (as specified)

        # Simulation parameters
        self.num_steps = 0
        self.max_steps = 100  # Maximum steps in an episode

    def generate_action_mapping(self, min_action, max_action, num_bins):
        """
        Generate a mapping from discrete indices to action values.

        Parameters:
            min_action (numpy.ndarray): Minimum values for each action dimension.
            max_action (numpy.ndarray): Maximum values for each action dimension.
            num_bins (int): Number of discrete possibilities per dimension.

        Returns:
            numpy.ndarray: A 2D array of all possible actions.
        """
        action_ranges = [np.linspace(min_action[i], max_action[i], num_bins) for i in range(len(min_action))]
        return np.array(np.meshgrid(*action_ranges)).T.reshape(-1, len(min_action))

    def step(self, action):
        """
        Takes an action (u_t), applies it to the system, and returns the next state, reward, done, and info.
        """
        # Map the discrete action index to a continuous action value
        # action = self.action_mapping[action]

        # Apply action to system dynamics
        delta_x_t_plus_1 = self.A @ (self.x_t - self.x_star) + self.B @ action
        
        # Apply saturation
        self.x_t = self.sat(delta_x_t_plus_1 + self.x_star)
        # print(self.x_t)

        # Compute reward as quadratic cost
        state_error = self.x_t - self.x_star
        control_effort = action
        reward = -(
            state_error.T @ self.Q @ state_error +
            control_effort.T @ self.R @ control_effort
        )/100

        # Check if episode is done
        self.num_steps += 1
        done = self.num_steps >= self.max_steps

        # Return next state, reward, done, and additional info
        return self.x_t, reward, done, {}

    def reset(self, x_init=None, x_star=None):
        """
        Resets the environment to the initial state.
        """
        self.x_t, self.x_star = self.generate_x_star_and_x_init_single(self.I_max)
        if x_init is not None:
            self.x_t = x_init
        if x_star is not None:
            self.x_star = x_star
        self.num_steps = 0  # Reset step counter
        return self.x_t

    def sat(self, z):
        """
        Saturation function to enforce current limits.
        """
        norm = np.linalg.norm(z)
        if norm <= self.I_max:
            return z
        else:
            return z / norm * self.I_max

    def render(self, mode="human"):
        """
        Optional: Implement rendering logic for visualization.
        """
        print(f"State: {self.x_t}")

    def generate_x_star_and_x_init_single(self, I_max, num_r=50, num_theta=50, seed=None):
        """
        Generate a single x_star and x_init using independent polar grid-based sampling.

        Parameters:
            I_max (float): Current magnitude limit of the inverter.
            num_r (int): Number of divisions in the radial direction for sampling (default: 50).
            num_theta (int): Number of divisions in the angular direction for sampling (default: 50).
            seed (int, optional): Random seed for reproducibility.

        Returns:
            x_star (numpy.ndarray): A single reference state as a 1D array of shape (2,).
            x_init (numpy.ndarray): A single initial state as a 1D array of shape (2,).
        """
        if seed is not None:
            np.random.seed(seed)

        # Define the radial and angular ranges for sampling
        R = np.linspace(0, I_max, num_r)  # Radial component
        Theta = np.linspace(0, 2 * np.pi - 2 * np.pi / num_theta, num_theta)  # Angular component

        # Randomly sample a point in the polar grid for x_star
        r_star = np.random.choice(R)
        theta_star = np.random.choice(Theta)
        x_star = np.array([r_star * np.cos(theta_star + np.pi / 4), r_star * np.sin(theta_star + np.pi / 4)])

        # Randomly sample a point in the polar grid for x_init
        r_init = np.random.choice(R)
        theta_init = np.random.choice(Theta)
        x_init = np.array([r_init * np.cos(theta_init + np.pi / 4), r_init * np.sin(theta_init + np.pi / 4)])

        return x_star, x_init
    
    
    def test_agent(self, agent, gamma=1, T=100, num_runs=1000):
        '''
        - agent (Agent): A trained agent.
        - gamma (float): The discount factor.
        - T (int): The number of timesteps to run the environment for in each epoch.
        - num_runs (int): Number of test runs.
        '''

        # Store total rewards from each run
        all_costs = []

        for run in range(num_runs):
            # for learning
            states = np.empty((T, 1, self.observation_space.shape[0]))
            if isinstance(self.action_space, gym.spaces.Discrete):
                # discrete action spaces only need to store a
                # scalar for each action.
                actions = np.empty((T, 1))
            else:
                # continuous action spaces need to store a
                # vector for each action.
                actions = np.empty((T, 1, self.action_space.shape[0]))
            rewards = np.empty((T, 1))
            dones = np.empty((T, 1))

            s_t = self.reset()

            for t in range(T):
                # Organized inverse sin/cos information
                # print(f"Run {run+1}, Step {t+1}")
                # print(f"Month (inverse sin): {np.arcsin(s_t[3])}, Month (inverse cos): {np.arccos(s_t[4])}")
                # print(f"Day (inverse sin): {np.arcsin(s_t[5])}, Day (inverse cos): {np.arccos(s_t[6])}")

                a_t, _ = agent.act(s_t)
                s_t_next, r_t, d_t = self.step(a_t)

                # for learning
                states[t] = s_t
                actions[t] = a_t
                rewards[t] = r_t
                dones[t] = d_t

                s_t = s_t_next

            cost = rewards.sum() 
            all_costs.append(cost)

        return sum(all_costs)/num_runs


    def reset_test(self):
        """
        Resets the environment to the initial state for testing.
        """
        self.reset()