import gym


def make_env(env_id):
    """
    Return an OpenAI Gym environment wrapped with appropriate wrappers. Throws error if env_id is not recognized.

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
    else:
        ValueError('{} is not a supported environment id.'.format(env_id))

    return env
