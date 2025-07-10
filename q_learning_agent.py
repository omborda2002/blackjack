import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05):
        self.q_table = defaultdict(lambda: [0.0 for _ in actions])
        self.actions = actions
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state):
        # Get valid actions based on game state
        valid_actions = self._get_valid_actions(state)
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            return self._best_action(state, valid_actions)
    
    def _get_valid_actions(self, state):
        """Get valid actions based on current state"""
        valid_actions = [0, 1]  # Can always stand or hit
        
        # Check if we can double (based on state context)
        if len(state) >= 3:  # Has enough info to determine if doubling is valid
            player_sum = state[0]
            # Assume we can double if player sum is reasonable for doubling
            if 9 <= player_sum <= 11:
                valid_actions.append(2)
        
        return valid_actions

    def _best_action(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = self.actions
            
        values = self.q_table[state]
        best_value = float('-inf')
        best_action = valid_actions[0]
        
        for action in valid_actions:
            if action < len(values):
                if values[action] > best_value:
                    best_value = values[action]
                    best_action = action
        
        return best_action

    def update(self, state, action, reward, next_state):
        if action not in self.actions:
            return  # Skip invalid actions
            
        action_idx = self.actions.index(action)
        valid_next_actions = self._get_valid_actions(next_state)
        best_next_action = self._best_action(next_state, valid_next_actions)
        
        if best_next_action in self.actions:
            best_next_value = self.q_table[next_state][self.actions.index(best_next_action)]
        else:
            best_next_value = 0
        
        # Q-learning update rule
        td_target = reward + self.gamma * best_next_value
        td_delta = td_target - self.q_table[state][action_idx]
        self.q_table[state][action_idx] += self.alpha * td_delta

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            table = pickle.load(f)
            self.q_table = defaultdict(lambda: [0.0 for _ in self.actions], table)
