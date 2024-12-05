import seaborn as sns; sns.set()
import numpy as np
from rl_monitoring_utils.return_and_advantage import calculate_returns, calculate_advantages
import gym

def A2C(env, agent, value_estimator,
        gamma=0.99, lam=0.95,
        epochs=256, train_V_iters=80, T=4052):
    """
    Advantage Actor-Critic (A2C) Implementation using PPO-like structure.
    """
    states = np.empty((T+1, env.num_envs, agent.N))
    if isinstance(env.action_space, gym.spaces.Discrete):
        actions = np.empty((T, env.num_envs))
    else:
        actions = np.empty((T, env.num_envs, agent.M))
    rewards = np.empty((T, env.num_envs))
    dones = np.empty((T, env.num_envs))
    log_probs_old = np.empty((T, env.num_envs))  # Optional for debug

    totals = []

    s_t = env.reset()
    for epoch in range(epochs):
        # Collect trajectory
        for t in range(T):
            a_t, log_prob = agent.act(s_t)
            s_t_next, r_t, d_t = env.step(a_t)

            states[t] = s_t
            actions[t] = a_t
            rewards[t] = r_t
            dones[t] = d_t
            log_probs_old[t] = log_prob  # Optional for debug

            s_t = s_t_next

        states[T] = s_t  # Add final state

        # Bootstrap final value
        V_last = value_estimator.predict(states[-1]).detach().numpy()
        rewards[-1] += gamma * (1 - dones[-1]) * V_last
        returns = calculate_returns(rewards, dones, gamma)

        # Update value function
        for i in range(train_V_iters):
            V_pred = value_estimator.predict(states)
            value_estimator.learn(V_pred[:-1], returns)

        # Compute advantages
        V_pred = V_pred.detach().numpy()
        TD_errors = rewards + gamma * (1 - dones) * V_pred[1:] - V_pred[:-1]
        advantages = calculate_advantages(TD_errors, lam, gamma)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient update
        for i in range(T):
            agent.learn(states[:-1], actions, advantages, log_probs_old, method="A2C")

        # Track total rewards
        totals.append(rewards.sum() / dones.sum())
        print(f'{epoch}/{epochs}: {totals[-1]:.2f}\r', end='')

    sns.lineplot(x=range(len(totals)), y=totals)
    return agent, totals
