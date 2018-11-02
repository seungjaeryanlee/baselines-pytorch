import numpy as np


def argmax_all(list_):
    """
    Return all argmax indices. Different from numpy.argmax where only 
    """
    return np.argwhere(list_ == np.amax(list_)).flatten()


class TabularAgent:
    def __init__(self, state_space, action_space):
        """
        A tabular, epsilon-greedy agent template class.
        """
        self.state_space = state_space
        self.action_space = action_space


class TabularQLearningAgent(TabularAgent):
    def __init__(self, state_space, action_space, lr=0.1, epsilon=0.1):
        """
        Agent that learns action values (Q) through Q-learning method. Assumes
        Discrete state space and action space.
        """
        TabularAgent.__init__(self, state_space, action_space)
        self.lr = lr
        self.epsilon = epsilon

        self.q_table = np.zeros((state_space.n, action_space.n))

    def get_action(self, state, epsilon=None):
        """
        Get action chosen by the agent given a state.
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.random() < epsilon: # Random selection
            return np.random.choice(self.action_space.n)
        else: # Greedy selection
            best_actions = argmax_all(self.q_table[state])
            return np.random.choice(best_actions)

    def get_state_value(self, state):
        """
        Get state value V(s) of given state s.
        """
        return np.max(self.q_table[state])

    def learn(self, state, action, next_state, reward, done):
        """
        Train the agent with a given transition.
        """
        if done:
            delta = reward - self.q_table[state][action]
        else:
            delta = reward + np.max(self.q_table[next_state]) - self.q_table[state][action]
        self.q_table[state][action] += self.lr * delta


class TabularBufferQAgent(TabularAgent):
    def __init__(self, state_space, action_space, replay_buffer,
                 lr=0.1, epsilon=0.1, batch_size=16):
        """
        Agent that learns action values (Q) through Q-learning method with
        experience replay. Assumes Discrete state space and action space.
        """
        TabularAgent.__init__(self, state_space, action_space)
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.q_table = np.zeros((state_space.n, action_space.n))

    def get_action(self, state, epsilon=None):
        """
        Get action chosen by the agent given a state.
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.random() < epsilon: # Random selection
            return np.random.choice(self.action_space.n)
        else: # Greedy selection
            best_actions = argmax_all(self.q_table[state])
            return np.random.choice(best_actions)

    def get_state_value(self, state):
        """
        Get state value V(s) of given state s.
        """
        return np.max(self.q_table[state])

    def learn(self, state, action, next_state, reward, done):
        """
        Train the agent by sampling from the replay buffer.

        TODO Improve efficiency through matrix operations.
        """
        transition = (state, action, next_state, reward, done)
        self.replay_buffer.append(transition)

        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        for state, action, next_state, reward, done in transitions:
            if done:
                delta = reward - self.q_table[state][action]
            else:
                delta = reward + np.max(self.q_table[next_state]) - self.q_table[state][action]
            self.q_table[state][action] += self.lr * delta


class TabularEmsembleQAgent(TabularAgent):
    def __init__(self, state_space, action_space, replay_buffers,
                 lr=0.1, epsilon=0.1):
        """
        Agent that learns action values (Q) through Q-learning method with
        ensemble experience replay. Assumes Discrete state space and action
        space.
        """
        TabularAgent.__init__(self, state_space, action_space)
        self.replay_buffers = replay_buffers
        self.lr = lr
        self.epsilon = epsilon

        self.q_table = np.zeros((state_space.n, action_space.n))

    def get_action(self, state, epsilon=None):
        """
        Get action chosen by the agent given a state.
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.random() < epsilon: # Random selection
            return np.random.choice(self.action_space.n)
        else: # Greedy selection
            best_actions = argmax_all(self.q_table[state])
            return np.random.choice(best_actions)

    def get_state_value(self, state):
        """
        Get state value V(s) of given state s.
        """
        return np.max(self.q_table[state])

    def learn(self, state, action, next_state, reward, done):
        """
        Train the agent by sampling from the replay buffer.

        TODO Improve efficiency through matrix operations.
        """
        transition = (state, action, next_state, reward, done)
        for replay_buffer, batch_size in self.replay_buffers:
            replay_buffer.append(transition)

            if len(replay_buffer) < batch_size:
                return

            transitions = replay_buffer.sample(batch_size)
            for state, action, next_state, reward, done in transitions:
                if done:
                    delta = reward - self.q_table[state][action]
                else:
                    delta = reward + np.max(self.q_table[next_state]) - self.q_table[state][action]
                self.q_table[state][action] += self.lr * delta
