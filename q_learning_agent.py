import random
import numpy as np

class QLearningAgent:
    def __init__(self, env, strategy="basic_strategy", alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.strategy = strategy
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}  # Q-values table
        self.action_space = [0, 1, 2]  # 0: stand, 1: hit, 2: double

        # Assign point-counting weights
        self.counting_weights = {
            2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 1, 8: 0, 9: 0, 10: -2, 11: -2
        } if "improved" in strategy else {
            2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, 11: -1
        }

    def _convert_state(self, state):
        return tuple(state)

    def get_best_action(self, state):
        # Greedy action from Q-table
        state = self._convert_state(state)
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.action_space)
        return int(np.argmax(self.q_table[state]))

    def choose_action(self, state, explore=True):
        # ε-greedy action
        state = self._convert_state(state)
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.action_space)
        if explore and random.random() < self.epsilon:
            return random.choice(self.action_space)
        return int(np.argmax(self.q_table[state]))

    def train(self, env, episodes=50000):
        for episode in range(1, episodes + 1):
            state = env.reset()
            running_count = 0
            done = False

            while not done:
                player_sum, dealer_card, usable_ace = state

                # Construct state tuple depending on strategy
                if "point_count" in self.strategy:
                    true_count = running_count // 1
                    full_state = (player_sum, dealer_card, usable_ace, true_count)
                else:
                    full_state = (player_sum, dealer_card, usable_ace)

                action = self.choose_action(full_state, explore=True)
                next_state, reward, done, card_seen = env.step(action)

                # Update count using cards seen
                for card in card_seen:
                    running_count += self.counting_weights.get(card, 0)

                # Next state processing
                if "point_count" in self.strategy:
                    true_count_next = running_count // 1
                    next_full_state = (next_state[0], next_state[1], next_state[2], true_count_next)
                else:
                    next_full_state = (next_state[0], next_state[1], next_state[2])

                fs, nfs = self._convert_state(full_state), self._convert_state(next_full_state)
                if fs not in self.q_table:
                    self.q_table[fs] = [0.0] * len(self.action_space)
                if nfs not in self.q_table:
                    self.q_table[nfs] = [0.0] * len(self.action_space)

                # Q-learning update
                best_next = max(self.q_table[nfs])
                td_target = reward + self.gamma * best_next
                self.q_table[fs][action] += self.alpha * (td_target - self.q_table[fs][action])

                state = next_state

            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
