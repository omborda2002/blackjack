import random
import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.action_space = [0, 1, 2]  # stand, hit, double

    def _convert_state(self, state):
        return tuple(state)

    def get_best_action(self, state):
        state = self._convert_state(state)
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.action_space)
        return int(np.argmax(self.q_table[state]))

    def choose_action(self, state, explore=True):
        state = self._convert_state(state)
        if explore and random.random() < self.epsilon:
            return random.choice(self.action_space)
        return self.get_best_action(state)

    def learn(self, state, action, reward, next_state, done):
        state = self._convert_state(state)
        next_state = self._convert_state(next_state)

        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.action_space)
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * len(self.action_space)

        q_old = self.q_table[state][action]
        q_max = max(self.q_table[next_state]) if not done else 0.0
        q_new = q_old + self.alpha * (reward + self.gamma * q_max - q_old)
        self.q_table[state][action] = q_new

        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)