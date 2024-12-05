import numpy as np

class StateActionOccupancyEstimator:
    def __init__(self, state_range=[(-4.167, 4.167), (-4.167, 4.167)], 
                 action_range=[(-1, 5), (-1, 3)], 
                 state_bins=10, action_bins=10):
        """
        Initialize the estimator with common parameters.

        Parameters:
            state_range (list of tuples): Ranges for state dimensions [(min_x0, max_x0), (min_x1, max_x1)].
            action_range (list of tuples): Ranges for action dimensions [(min_u0, max_u0), (min_u1, max_u1)].
            state_bins (int): Number of bins per dimension for the state space.
            action_bins (int): Number of bins per dimension for the action space.
        """
        self.state_range = state_range
        self.action_range = action_range
        self.state_bins = state_bins
        self.action_bins = action_bins

        # Define bin edges for state and action
        self.x0_bins = np.linspace(state_range[0][0], state_range[0][1], state_bins + 1)
        self.x1_bins = np.linspace(state_range[1][0], state_range[1][1], state_bins + 1)
        self.u0_bins = np.linspace(action_range[0][0], action_range[0][1], action_bins + 1)
        self.u1_bins = np.linspace(action_range[1][0], action_range[1][1], action_bins + 1)

        self.state_edges = [self.x0_bins, self.x1_bins]
        self.action_edges = [self.u0_bins, self.u1_bins]


    def estimate(self, env, agent, num_steps=100, num_runs=10, batch_size=8):
        """
        Estimate global state-action occupancy for the given environment and agent, 
        with support for batched states.

        Parameters:
            env (object): The environment with a reset() and step(action) method.
            agent (object): The agent with a policy(state) method returning an action.
            num_steps (int): Number of steps to simulate per run.
            num_runs (int): Number of runs/epochs to average over.
            batch_size (int): Number of parallel batches/scenarios.

        Returns:
            numpy.ndarray: A 4D array of shape 
            (state_bins, state_bins, action_bins, action_bins) 
            with the estimated state-action occupancy.
        """
        # Initialize cumulative occupancy counts
        cumulative_occupancy = np.zeros((self.state_bins, self.state_bins, 
                                        self.action_bins, self.action_bins), dtype=int)

        for run in range(num_runs):
            # Reset the environment to get the initial states for all batches
            states = env.reset()  # Shape: (batch_size, state_dim)

            for _ in range(num_steps):
                # Get actions for all batches from the agent
                actions, _ = agent.act(states)  # Shape: (batch_size, action_dim)

                # Take a step in the environment for all batches
                next_states, _, _ = env.step(actions)  # Shape: (batch_size, state_dim)

                # Process each scenario in the batch
                for batch_idx in range(batch_size):
                    state = states[batch_idx]  # Current state of the batch
                    action = actions[batch_idx]  # Corresponding action

                    # Discretize state and action
                    x0_bin_idx = np.digitize(state[0], self.x0_bins) - 1
                    x1_bin_idx = np.digitize(state[1], self.x1_bins) - 1
                    u0_bin_idx = np.digitize(action[0], self.u0_bins) - 1
                    u1_bin_idx = np.digitize(action[1], self.u1_bins) - 1

                    # Ensure indices are within bounds
                    x0_bin_idx = min(max(x0_bin_idx, 0), self.state_bins - 1)
                    x1_bin_idx = min(max(x1_bin_idx, 0), self.state_bins - 1)
                    u0_bin_idx = min(max(u0_bin_idx, 0), self.action_bins - 1)
                    u1_bin_idx = min(max(u1_bin_idx, 0), self.action_bins - 1)

                    # Increment occupancy
                    cumulative_occupancy[x0_bin_idx, x1_bin_idx, u0_bin_idx, u1_bin_idx] += 1

                # Update the states to the next states
                states = next_states

        # Calculate occupancy percentage
        total_counts = np.sum(cumulative_occupancy)
        occupancy_percentage = (cumulative_occupancy / total_counts) if total_counts > 0 else cumulative_occupancy

        return occupancy_percentage


    def get_occupancy(self, occupancy_percentage, s, a):
        """
        Get the occupancy measure for a specific state and action.

        Parameters:
            occupancy_percentage (numpy.ndarray): The 4D occupancy percentage array.
            s (tuple): The state (s0, s1).
            a (tuple): The action (a0, a1).

        Returns:
            float: The occupancy percentage for the given state and action.
        """
        x0_bin_idx = np.digitize(s[0], self.x0_bins) - 1
        x1_bin_idx = np.digitize(s[1], self.x1_bins) - 1
        u0_bin_idx = np.digitize(a[0], self.u0_bins) - 1
        u1_bin_idx = np.digitize(a[1], self.u1_bins) - 1

        # Ensure indices are within bounds
        x0_bin_idx = min(max(x0_bin_idx, 0), self.state_bins - 1)
        x1_bin_idx = min(max(x1_bin_idx, 0), self.state_bins - 1)
        u0_bin_idx = min(max(u0_bin_idx, 0), self.action_bins - 1)
        u1_bin_idx = min(max(u1_bin_idx, 0), self.action_bins - 1)

        occupancy = occupancy_percentage[x0_bin_idx, x1_bin_idx, u0_bin_idx, u1_bin_idx]
        return occupancy
    

    def print_occupancy_measure(self, occupancy_measure):
        """
        Print the occupancy measure in a formatted and easy-to-read way.

        Parameters:
        - occupancy_measure: A 4D numpy array representing the occupancy measure.
        - state_edges: List of edges for state bins.
        - action_edges: List of edges for action bins.
        - state_bins: Number of bins in the state space.
        - action_bins: Number of bins in the action space.
        """
        print("\nState-Action Occupancy Measure")
        print("=" * 80)
        print(f"State Bins: {self.state_bins}, Action Bins: {self.action_bins}\n")
        print("{:<30} | {:<30} | {:<15}".format("State (s0, s1)", "Actions (u0, u1)", "Occupancy"))
        print("-" * 80)

        for i in range(self.state_bins):
            for j in range(self.state_bins):
                # Compute the state occupancy measure by summing over all actions
                state_occupancy = np.sum(occupancy_measure[i, j, :, :])

                # Skip printing for states with zero occupancy
                if state_occupancy == 0:
                    continue

                # Get the state tuple as a midpoint of the bins
                state_tuple = (
                    (self.state_edges[0][i] + self.state_edges[0][i + 1]) / 2,
                    (self.state_edges[1][j] + self.state_edges[1][j + 1]) / 2,
                )

                # Extract and flatten action probabilities
                action_probabilities = occupancy_measure[i, j, :, :].flatten()

                # Generate readable output for action probabilities
                action_str = ", ".join(
                    f"{p:.4f}" for p in action_probabilities if p > 0
                )  # Show only non-zero probabilities

                print(
                    f"({state_tuple[0]:.2f}, {state_tuple[1]:.2f})".ljust(30)
                    + " | "
                    + action_str.ljust(30)
                    + " | "
                    + f"{state_occupancy:.4f}".ljust(15)
                )
        print("-" * 80)
        print("Note: Probabilities are shown only for actions with non-zero occupancy.\n")







# def estimate_state_occupancy(env, agent, initial_state, num_bins=10, num_steps=100, num_runs=10, batch_size=8):
#     """
#     Estimate state occupancy for each batch in a given environment and agent over multiple runs as percentages.

#     Parameters:
#         env (object): The environment with a reset() and step(action) method.
#         agent (object): The agent with a policy(state) method returning an action.
#         initial_state (numpy.ndarray): The initial state to start the simulation (shape: (batch_size, state_dim)).
#         num_bins (int): Number of bins per dimension for the heatmap.
#         num_steps (int): Number of steps to simulate per run.
#         num_runs (int): Number of runs/epochs to average over.
#         batch_size (int): Number of parallel simulations (batches).

#     Returns:
#         numpy.ndarray: A 3D array of shape (batch_size, num_bins, num_bins) with the estimated state occupancy as percentages.
#     """
#     # Initialize cumulative occupancy counts for each batch
#     cumulative_occupancy = np.zeros((batch_size, num_bins, num_bins), dtype=int)

#     # Define fixed symmetric range for x0 and x1
#     x_min, x_max = -4.167, 4.167
#     x0_bins = np.linspace(x_min, x_max, num_bins + 1)
#     x1_bins = np.linspace(x_min, x_max, num_bins + 1)

#     # Outer loop for multiple runs
#     for run in range(num_runs):
#         # Reset the environment to the initial state for each run
#         env.reset()
#         env.state = initial_state  # Assuming env.state supports batch_size states

#         # Track state trajectories for all batches
#         state_trajectories = []

#         # Simulate the environment for num_steps
#         for _ in range(num_steps):
#             # Get actions for all batches from the agent
#             actions, _ = agent.act(env.state)  # Shape: (batch_size, action_dim)

#             # Take a step in the environment for all batches
#             next_states, _, _ = env.step(actions)  # Shape: (batch_size, state_dim)

#             # Record the states for all batches
#             state_trajectories.append(next_states)

#         # Convert the trajectory to a numpy array with shape (num_steps, batch_size, state_dim)
#         state_trajectories = np.array(state_trajectories)

#         # Bin the state trajectories for all batches
#         for step in range(state_trajectories.shape[0]):
#             for batch in range(batch_size):
#                 state = state_trajectories[step, batch]

#                 x0_bin_idx = np.digitize(state[0], x0_bins) - 1
#                 x1_bin_idx = np.digitize(state[1], x1_bins) - 1

#                 # Ensure indices are within bounds
#                 x0_bin_idx = min(max(x0_bin_idx, 0), num_bins - 1)
#                 x1_bin_idx = min(max(x1_bin_idx, 0), num_bins - 1)

#                 # Increment the count for the corresponding bin for this batch
#                 cumulative_occupancy[batch, x0_bin_idx, x1_bin_idx] += 1

#     # Normalize the occupancy counts to get percentages for each batch
#     total_counts = np.sum(cumulative_occupancy, axis=(1, 2), keepdims=True)  # Shape: (batch_size, 1, 1)
#     occupancy_percentage = (cumulative_occupancy / total_counts) if np.any(total_counts > 0) else cumulative_occupancy

#     return occupancy_percentage
