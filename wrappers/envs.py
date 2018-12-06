import gym

from .atari_wrappers import make_atari, wrap_deepmind
from .torch_wrappers import TorchWrapper, AtariPermuteWrapper


def make_env(env_id):
    """
    Return an OpenAI Gym environment wrapped with appropriate wrappers. Throws
    error if env_id is not recognized.

    Parameters
    ----------
    env_id : str
        OpenAI Gym ID for environment.

    Returns
    -------
    env
        Wrapped OpenAI environment.

    """
    if env_id == 'CartPole-v0':
        env = gym.make(env_id)
        env = TorchWrapper(env)
    elif env_id == 'PongNoFrameskip-v4':
        env = make_atari(env_id)
        env = wrap_deepmind(env, frame_stack=True)
        env = TorchWrapper(env)
        env = AtariPermuteWrapper(env)
    else:
        raise ValueError('{} is not a supported environment.'.format(env_id))

    return env
