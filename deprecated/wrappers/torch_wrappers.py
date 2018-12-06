import gym
import torch


class TorchWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        ob = self.env.reset()
        ob = torch.FloatTensor([ob])
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob     = torch.FloatTensor([ob])
        reward = torch.FloatTensor([reward])
        done   = torch.FloatTensor([done])
        return ob, reward, done, info


class AtariTorchWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(shp[2], shp[0], shp[1]),
                                                dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        ob = torch.FloatTensor(ob).permute(2, 0, 1)
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = torch.FloatTensor(ob).permute(2, 0, 1)
        return ob, reward, done, info
