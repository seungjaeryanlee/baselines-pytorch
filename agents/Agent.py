import random
import os
import time

import torch
import torch.optim as optim

from commons import get_writer, get_epsilon_decay_function
from networks import DQN, AtariDQN
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

        if len(env.observation_space.shape) == 1:  # Feature-type observations
            self.current_net = DQN(
                env.observation_space.shape[0], env.action_space.n)
            self.target_net = DQN(
                env.observation_space.shape[0], env.action_space.n)
        elif len(env.observation_space.shape) == 3:  # Image-type observations
            self.current_net = AtariDQN(
                env.observation_space.shape[0], env.action_space.n)
            self.target_net = AtariDQN(
                env.observation_space.shape[0], env.action_space.n)
        else:
            raise ValueError('Unsupported observation type: '
                             'check environment observation space.')
        self.optimizer = optim.Adam(
            self.current_net.parameters(), lr=args.LEARNING_RATE)
        self.replay_buffer = UniformReplayBuffer(args.REPLAY_BUFFER_SIZE)

        self.writer = get_writer('DQN', args)
        self.get_epsilon_by_frame_idx = get_epsilon_decay_function(
            args.EPSILON_START, args.EPSILON_END, args.EPSILON_DECAY_DURATION)

    def act(self, state, epsilon):
        """
        Return an action sampled from an epsilon-greedy policy.
        Parameters
        ----------
        state
            The state to compute the epsilon-greedy action of.
        epsilon : float
            Epsilon in epsilon-greedy policy: probability of choosing a random
            action.

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
            # Start timer
            t_start = time.time()

            # Interact and save to replay buffer
            epsilon = self.get_epsilon_by_frame_idx(frame_idx)
            action = self.act(state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.writer.add_scalar('data/rewards', reward.item(), frame_idx)
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward.item()

            if done:
                print('Frame {:5d}/{:5d}\tReturn {:3.2f}\tLoss {:2.4f}'.format(
                    frame_idx + 1, nb_frames, episode_reward, loss.item()))
                self.writer.add_scalar(
                    'data/episode_rewards', episode_reward, episode_idx)
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

            # End timer
            t_end = time.time()
            self.writer.add_scalar('data/time', t_end - t_start, frame_idx)

        self.writer.close()

    def save(self, PATH='best/'):
        """
        Save the parameters of the agent's neural networks and optimizers.
        """
        if not os.path.exists('saves/'):
            os.makedirs('saves/')
        if not os.path.exists('saves/' + PATH):
            # TODO Check if dangerous?
            os.makedirs('saves/' + PATH)
        torch.save(self.current_net.state_dict(), 'saves/' + PATH + 'dqn.pth')
        torch.save(self.optimizer.state_dict(), 'saves/' + PATH + 'optim.pth')
        print('[save] Successfully saved network and optimizer to '
              '{}. Note that save/ directory is ignored by git.'.format(PATH))

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
            A tuple of batches: (state_batch, action_batch, reward_batch,
            next_state_batch, done_batch).

        Returns
        -------
        loss : torch.FloatTensor
            MSE loss of target Q and prediction Q that can be backpropagated.
            Has shape torch.Size([1]).
        """
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.args.BATCH_SIZE)

        # TODO Send to device in self.replay.sample?
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        # Predicted Q: Q_current(s, a)
        # q_values : torch.Size([BATCH_SIZE, self.env.action_space.n])
        # action   : torch.Size([BATCH_SIZE])
        # q_value  : torch.Size([BATCH_SIZE])
        q_values = self.current_net(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze()

        # Target Q: r + gamma * max_{a'} Q_target(s', a')
        # next_q_values    : torch.Size([BATCH_SIZE, self.env.action_space.n])
        # next_q_value     : torch.Size([BATCH_SIZE])
        # expected_q_value : torch.Size([BATCH_SIZE])
        with torch.no_grad():
            # Q_target(s', a')
            next_q_values = self.current_net(next_state)
            next_q_value = next_q_values.max(dim=1)[0].squeeze()
            expected_q_value = reward + \
                self.args.DISCOUNT * next_q_value * (1 - done)

        assert expected_q_value.shape == q_value.shape

        # Compute MSE Loss
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        return loss

    def _update_target(self):
        """
        Update weights of Target DQN with weights of current DQN.
        """
        self.target_net.load_state_dict(self.current_net.state_dict())
