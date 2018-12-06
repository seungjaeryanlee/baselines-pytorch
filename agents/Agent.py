import random
import time

import torch
import torch.optim as optim

from commons import get_writer
from networks import DQN
from replays import UniformReplayBuffer


class Agent:
    def __init__(self, env, device, args):
        """
        A Deep Q-Network (DQN) agent that can be trained with environments that
        have feature vectors as states and discrete values as actions.
        """
        self.env = env
        self.device = device
        self.args = args

        self.writer = get_writer('DQN', args)
        self.current_net = DQN(env.observation_space.shape[0], env.action_space.n)
        self.target_net = DQN(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.current_net.parameters(), lr=args.LEARNING_RATE)
        self.replay_buffer = UniformReplayBuffer(args.REPLAY_BUFFER_SIZE)

        # TODO Implement Epsilon Decay
        self.get_epsilon_by_frame_idx = lambda frame_idx: 0.1

    def act(self, state, epsilon):
        """
        Return an action sampled from an epsilon-greedy policy.

        TODO Implement

        Parameters
        ----------
        state
            The state to compute the epsilon-greedy action of.
        epsilon : float
            Epsilon in epsilon-greedy policy: probability of choosing a random action.

        Returns
        -------
        action : int
            An integer representing a discrete action chosen by the agent.
        """
        if random.random() > epsilon:
            with torch.no_grad():
                q_value = self.current_net(state)
            action = q_value.max(1)[1].item()
        else:
            action = self.env.action_space.sample()
        return action

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
        nb_frames = nb_frames if nb_frames else self.args.NB_FRAMES

        episode_reward = 0
        episode_idx = 0
        loss = torch.FloatTensor([0])
        state = self.env.reset()
        for frame_idx in range(1, nb_frames + 1):
            # Interact and save to replay buffer
            epsilon = self.get_epsilon_by_frame_idx(frame_idx)
            action = self.act(state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.writer.add_scalar('data/rewards', reward.item(), frame_idx)
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward.item()

            if done:
                print('Frame {:5d}/{:5d}\tReturn {:3.2f}\tLoss {:2.4f}'.format(frame_idx + 1, nb_frames, episode_reward, loss.item()))
                self.writer.add_scalar('data/episode_rewards', episode_reward, episode_idx)
                state = self.env.reset()
                episode_reward = 0
                episode_idx += 1

            # Train DQN if the replay buffer is populated enough
            if len(self.replay_buffer) > self.args.MIN_REPLAY_BUFFER_SIZE:
                self.optimizer.zero_grad()
                replay_batch = self.replay_buffer.sample(self.args.BATCH_SIZE)
                loss = self._compute_loss(replay_batch)
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('data/losses', loss.item(), frame_idx)

            # Update Target DQN periodically
            if (frame_idx + 1) % self.args.TARGET_UPDATE_STEPS == 0:
                self._update_target()

        self.writer.close()

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

    def _compute_loss(self, batch):
        """
        Compute batch MSE loss between 1-step target Q and prediction Q.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A tuple of batches: (state_batch, action_batch, reward_batch, next_state_batch, done_batch).

        Returns
        -------
        loss : torch.FloatTensor
            MSE loss of target Q and prediction Q that can be backpropagated. Has shape torch.Size([1]).
        """
        state, action, reward, next_state, done = self.replay_buffer.sample(self.args.BATCH_SIZE)

        # TODO Send to device in self.replay.sample?
        state      = state.to(self.device)
        next_state = next_state.to(self.device)
        action     = action.to(self.device)
        reward     = reward.to(self.device)
        done       = done.to(self.device)

        # Predicted Q: Q_current(s, a)
        # q_values : torch.Size([self.args.BATCH_SIZE, self.env.action_space.n])
        # action   : torch.Size([self.args.BATCH_SIZE])
        # q_value  : torch.Size([self.args.BATCH_SIZE])
        q_values = self.current_net(state)
        q_value  = q_values.gather(1, action.unsqueeze(1)).squeeze()

        # Target Q: r + gamma * max_{a'} Q_target(s', a')
        # next_q_values    : torch.Size([self.args.BATCH_SIZE, self.env.action_space.n])
        # next_q_value     : torch.Size([self.args.BATCH_SIZE])
        # expected_q_value : torch.Size([self.args.BATCH_SIZE])
        with torch.no_grad():
            # Q_target(s', a')
            next_q_values = self.current_net(next_state)
            next_q_value  = next_q_values.max(dim=1)[0].squeeze()
            expected_q_value = reward + self.args.DISCOUNT * next_q_value * (1 - done)

        assert expected_q_value.shape == q_value.shape

        # Compute MSE Loss
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        return loss

    def _update_target(self):
        """
        Update weights of Target DQN with weights of current DQN.
        """
        self.target_net.load_state_dict(self.current_net.state_dict())
