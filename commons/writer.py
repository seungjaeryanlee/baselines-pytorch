from tensorboardX import SummaryWriter


def get_writer(name, args):
    """
    Create a tensorboardX writer with hyperparameters in title.

    Parameters
    ----------
    name : str
        Name of the agent.
    args
        Namespace that specifies environment, hyperparameters, and other information that determines the training.

    Returns
    -------
    writer
        A tensorboardX SummaryWriter.
    """
    writer = SummaryWriter('runs/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}'.format(
        args.ENV_ID,
        name,
        args.SEED,
        args.NB_FRAMES,
        args.BATCH_SIZE,
        args.DISCOUNT,
        args.TARGET_UPDATE_STEPS,
        args.LEARNING_RATE,
        args.REPLAY_BUFFER_SIZE,
        args.MIN_REPLAY_BUFFER_SIZE,
        args.EPSILON_START,
        args.EPSILON_END,
        args.EPSILON_DECAY_DURATION,
    ))

    return writer
