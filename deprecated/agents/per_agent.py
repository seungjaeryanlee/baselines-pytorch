import random

import torch


# GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PERAgent:
    def __init__(self, env, current_net, target_net, replay, optimizer, discount):
        self.env = env
        self.current_net = current_net
        self.target_net = target_net
        self.replay = replay
        self.optimizer = optimizer
        self.discount = discount

    def act(self, state, epsilon):
        """
        Choose action given state s and epsilon e. The epsilon denotes the
        possibility of random action for epsilon-greedy policy.
        """
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_value = self.current_net(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.env.action_space.n)

        return action

    def train(self, batch_size, beta):
        """
        Train the agent with one batch and return loss.
        """
        state, action, reward, next_state, done, indices, weights = self.replay.sample(batch_size, beta)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.LongTensor(action).to(device)
        reward     = torch.FloatTensor(reward).to(device)
        done       = torch.FloatTensor(done).to(device)
        weights    = torch.FloatTensor(weights)

        q_values            = self.current_net(state)
        next_q_values       = self.current_net(next_state)
        next_q_state_values = self.target_net(next_state) 

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.discount * next_q_value * (1 - done)

        loss  = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss  = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.replay.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        return loss

    def update_target(self):
        self.target_net.load_state_dict(self.current_net.state_dict())
