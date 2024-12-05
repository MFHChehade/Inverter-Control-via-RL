import copy
import numpy as np
import gym

class VectorizedEnvWrapper(gym.Wrapper):
    def __init__(self, env, num_envs=1):
        '''
        env (gym.Env): to make copies of
        num_envs (int): number of copies
        '''
        super().__init__(env)
        self.num_envs = num_envs
        self.envs = [copy.deepcopy(env) for n in range(num_envs)]

    def reset(self, x_init=None, x_star=None):
        '''
        Return and reset each environment
        '''
        return np.asarray([env.reset(x_init, x_star) for env in self.envs])

    def step(self, actions):
        next_states, rewards, dones = [], [], []
        for env, action in zip(self.envs, actions):
            # Pass the action directly without .item()
            next_state, reward, done, _ = env.step(action)  
            if done:
                next_states.append(env.reset())
            else:
                next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        return np.array(next_states), np.array(rewards), np.array(dones)