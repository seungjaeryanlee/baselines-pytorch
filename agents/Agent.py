class Agent:
    def __init__(self, env, args):
        """
        A Deep Q-Network (DQN) agent that can be trained with environments that
        have feature vectors as states and discrete values as actions.
        """
        pass
    
    def train(self, nb_frames=None):
        """
        Train the agent by interacting with the environment. The number of
        frames to train the agent can be specified either during initialization
        of the agent (through `args`) or as a parameter of this agent (through
        `nb_frames`).

        Parameters
        ----------
        nb_frames: int, optional
            Number of frames to train the agent. If not specified, use
            `args.NB_FRAMES`.
        """
        pass

    def save(self):
        """
        Save the parameters of the agent's neural networks and optimizers.
        """
        pass

    def test(self, nb_epsiodes=1, render=True):
        """
        Run the agent for `nb_epsiodes` with or without render.

        Parameters
        ----------
        nb_epsiodes: int, optional
            Number of episodes to test the agent. Defaults to 1 episode.
        render: bool, optional
            Render the environment. Defaults to True.
        """
        pass
