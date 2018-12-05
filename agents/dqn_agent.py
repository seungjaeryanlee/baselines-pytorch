import random

import torch


# GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    def __init__(self, env, current_net, target_net, replay, optimizer, device, discount):
        self.env = env
        self.current_net = current_net
        self.target_net = target_net
        self.replay = replay
        self.optimizer = optimizer
        self.device = device
        self.discount = discount

    def act(self, state, epsilon):
        """
        Choose action given state s and epsilon e. The epsilon denotes the
        possibility of random action for epsilon-greedy policy.
        """
        if random.random() > epsilon:
            with torch.no_grad():
                q_value = self.current_net(state)
            action = q_value.max(1)[1].item()
        else:
            action = self.env.action_space.sample()
        return action

    def train(self, batch_size):
        """
        Train the agent with one batch and return loss.
        """
        state, action, reward, next_state, done = self.replay.sample(batch_size)

        state      = state.to(self.device)
        next_state = next_state.unsqueeze(0).to(self.device)
        action     = torch.LongTensor([action]).to(self.device)
        reward     = reward.to(self.device)
        done       = done.to(self.device)

        # Predicted Q: Q_current(s, a)
        q_values = self.current_net(state).unsqueeze(0)
        q_value  = q_values.gather(2, action.unsqueeze(2)).squeeze()

        # Target Q: r + gamma * max_{a'} Q_target(s', a')
        with torch.no_grad():
            # Q_target(s', a')
            next_q_values = self.current_net(next_state)
            next_q_value  = next_q_values.max(dim=2)[0].squeeze()

            expected_q_value = reward + self.discount * next_q_value * (1 - done)

        assert expected_q_value.shape == q_value.shape

        # Compute MSE Loss
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update_target(self):
        self.target_net.load_state_dict(self.current_net.state_dict())
