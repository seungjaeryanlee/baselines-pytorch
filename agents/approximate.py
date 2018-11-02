import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ApproximateAgent:
    """
    A template class for deep reinforcement learning agents that uses neural
    networks to approximate Q-values with epsilon-greedy exploration.
    """
    def __init__(self, q_network, state_space, action_space, epsilon):
        self.q_network = q_network
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon

    def get_action(self, state, epsilon=None):
        """
        Get action chosen by the agent given a state.
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.random() < epsilon: # Random selection
            return np.random.choice(self.action_space.n)
        else: # Greedy selection
            state = torch.FloatTensor(state)
            q_values = self.q_network(state)
            action = torch.argmax(q_values).item()
            return action

    def get_state_value(self, state):
        """
        Get state value V(s) of given state s.
        """
        return self.q_network(torch.FloatTensor(state)).max().item()


class ApproximateQLearningAgent(ApproximateAgent):
    def __init__(self, q_network, optimizer, state_space, action_space, epsilon=0.1, gamma=0.999):
        """
        Agent that learns action values (Q) through Q-learning method. Assumes
        Box state space and Discrete action space.
        """
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma

        self.q_network = q_network
        self.optimizer = optimizer

    def learn(self, state, action, next_state, reward, done):
        """
        Train the agent with a given transition.
        """
        predicted_q_value = self.q_network(torch.FloatTensor(state))[action]
        with torch.no_grad():
            target_q_value = torch.FloatTensor([reward])
            if not done:
                target_q_value += self.gamma * self.q_network(torch.FloatTensor(next_state)).max().item()
        loss = F.smooth_l1_loss(predicted_q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
