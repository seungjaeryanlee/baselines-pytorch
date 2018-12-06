import gym
import torch


class TorchWrapper(gym.Wrapper):
    """
    OpenAI Environment Wrapper that changes output types of `env.reset()` and `env.step()` to `torch.tensor`.
    """
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
