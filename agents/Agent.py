import time


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
        nb_frames = nb_frames if nb_frames else self.args.NB_FRAMES

        episode_reward = 0
        episode_idx = 0
        loss = 0
        state = self.env.reset()
        for frame_idx in range(1, nb_frames + 1):
            # Interact and save to replay buffer
            epsilon = epsilon_by_frame(frame_idx)
            action = self.act(state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.writer.add_scalar('data/rewards', reward, frame_idx)
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                print('Frame {:5d}/{:5d}\tReturn {:3.2f}\tLoss {:2.4f}'.format(frame_idx + 1, nb_frames, episode_reward, loss))
                self.writer.add_scalar('data/episode_rewards', episode_reward, episode_idx)
                state = self.env.reset()
                episode_reward = 0
                episode_idx += 1

            # Train DQN if the replay buffer is populated enough
            if len(replay_buffer) > self.args.MIN_REPLAY_BUFFER_SIZE:
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
