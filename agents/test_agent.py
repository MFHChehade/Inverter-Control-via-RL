import seaborn as sns; sns.set()
import numpy as np
import gym
from rl_monitoring_utils.return_and_advantage import calculate_returns

def test_agent(env, agent, gamma=0.99, T=1000):

    '''
    - env (gym.Env): The environment to train the agent on.
    - agent (Agent): A trained agent.
    - gamma (float): The discount factor.
    - T (int): The number of timesteps to run the environment for in each epoch.
    '''

    # for learning
    states = np.empty((T, env.num_envs, agent.N))
    if isinstance(env.action_space, gym.spaces.Discrete):
        # discrete action spaces only need to store a
        # scalar for each action.
        actions = np.empty((T, env.num_envs))
    else:
        # continuous action spaces need to store a
        # vector for each action.
        actions = np.empty((T, env.num_envs, agent.M))
    rewards = np.empty((T, env.num_envs))
    dones = np.empty((T, env.num_envs))

    # for plotting
    totals = []

    s_t = env.reset()

    for t in range(T):
        a_t, _ = agent.act(s_t)
        s_t_next, r_t, d_t = env.step(a_t)

        # for learning
        states[t] = s_t
        actions[t] = a_t
        rewards[t] = r_t
        dones[t] = d_t

        s_t = s_t_next

    # returns = calculate_returns(rewards, dones, gamma)

    cost = rewards.sum()/dones.sum()

    return cost