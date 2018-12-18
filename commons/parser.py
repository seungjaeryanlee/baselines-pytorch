import argparse


def get_args(description='', default_args=None):
    """
    Parse arguments with argument parser for and return hyperparameters as a
    dictionary.

    TODO This seems to be environment-dependent.

    Parameters
    ----------
    description: str
        Description for the argument parser. Defaults to an empty string.
    default_args: dict or None
        Defalt argument values in a dictionary.

    Returns
    -------
    args : dict
        Dictionary containing hyperparameter options specified by user or set
        by default.
    """
    if not default_args:
        default_args = {
            'ENV_ID': 'CartPole-v0',
            'SEED': 1,
            'NB_FRAMES': 10000,
            'BATCH_SIZE': 32,
            'DISCOUNT': 0.99,
            'TARGET_UPDATE_STEPS': 100,
            'LEARNING_RATE': 1e-3,
            'REPLAY_BUFFER_SIZE': 1000,
            'MIN_REPLAY_BUFFER_SIZE': 100,
            'EPSILON_START': 1,
            'EPSILON_END': 0.1,
            'EPSILON_DECAY_DURATION': 5000,
        }

    parser = argparse.ArgumentParser(description)
    parser.add_argument('-e', '--envid', action='store', dest='ENV_ID',
                        default=default_args['ENV_ID'], type=str)
    parser.add_argument('-s', '--seed', action='store', dest='SEED',
                        default=default_args['SEED'], type=int)
    parser.add_argument('-n', '--frames', action='store', dest='NB_FRAMES',
                        default=default_args['NB_FRAMES'], type=int)
    parser.add_argument('-b', '--batch', action='store', dest='BATCH_SIZE',
                        default=default_args['BATCH_SIZE'], type=int)
    parser.add_argument('-d', '--discount', action='store', dest='DISCOUNT',
                        default=default_args['DISCOUNT'], type=float)
    parser.add_argument('-u', '--update', action='store', dest='TARGET_UPDATE_STEPS',
                        default=default_args['TARGET_UPDATE_STEPS'], type=int)
    parser.add_argument('-l', '--lr', action='store', dest='LEARNING_RATE',
                        default=default_args['LEARNING_RATE'], type=float)
    parser.add_argument('--replay', action='store', dest='REPLAY_BUFFER_SIZE',
                        default=default_args['REPLAY_BUFFER_SIZE'], type=int)
    parser.add_argument('--min-replay', action='store', dest='MIN_REPLAY_BUFFER_SIZE',
                        default=default_args['MIN_REPLAY_BUFFER_SIZE'], type=int)
    parser.add_argument('--epsilon-start', action='store', dest='EPSILON_START',
                        default=default_args['EPSILON_START'], type=float)
    parser.add_argument('--epsilon-end', action='store', dest='EPSILON_END',
                        default=default_args['EPSILON_END'], type=float)
    parser.add_argument('--epsilon-decay-duration', action='store', dest='EPSILON_DECAY_DURATION',
                        default=default_args['EPSILON_DECAY_DURATION'], type=int)
    args = parser.parse_args()

    if args.REPLAY_BUFFER_SIZE < args.BATCH_SIZE:
        raise ValueError('Replay buffer size is below batch size: sampling is impossible')
    if args.REPLAY_BUFFER_SIZE < args.MIN_REPLAY_BUFFER_SIZE:
        raise ValueError('Replay buffer size is below minimum replay buffer size: training is impossible')

    return args
