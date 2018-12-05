import random

import torch


class NaiveDQNAgent:
    def __init__(self, env, net, optimizer, device, discount):
        self.env = env
        self.net = net
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
                q_value = self.net(state)
            action = q_value.max(1)[1].item()
        else:
            action = self.env.action_space.sample()

        return action

    def train(self, state, action, reward, next_state, done):
        """
        Train the agent with one batch and return loss.
        """
        state      = state.to(self.device)
        next_state = next_state.unsqueeze(0).to(self.device)
        action     = torch.LongTensor([action]).to(self.device)
        reward     = reward.to(self.device)
        done       = done.to(self.device)

        q_values = self.net(state)
        q_value  = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values    = self.net(next_state)
            next_q_value     = next_q_values.max(1)[0]
            expected_q_value = reward + self.discount * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
