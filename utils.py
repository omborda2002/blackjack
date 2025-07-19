import os
import csv
from datetime import datetime
from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from termcolor import cprint

# === Logger Initialization ===
def init_logger(log_dir: str = "logs", scenario: str = "default") -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{scenario}_{timestamp}.csv"
    path = os.path.join(log_dir, filename)

    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "total_reward", "steps", "win"])
    return path

# === Episode Logger ===
def log_episode(log_path: str, episode: int, total_reward: int, steps: int, win: bool) -> None:
    with open(log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, total_reward, steps, int(win)])

# === Summary Statistics ===
def summarize_logs(log_path: str) -> Dict[str, float]:
    total = 0
    wins = 0
    rewards = 0
    count = 0

    with open(log_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            total += 1
            rewards += float(row["total_reward"])
            wins += int(row["win"])
            count += 1

    return {
        "episodes": count,
        "avg_reward": rewards / count if count else 0,
        "win_rate": wins / count if count else 0
    }

# === Plotting Function (Optional Use) ===
def plot_logs(log_path: str):
    df = pd.read_csv(log_path)
    df["rolling_win_rate"] = df["win"].rolling(window=100, min_periods=1).mean()

    plt.figure()
    plt.plot(df["episode"], df["total_reward"], label="Total Reward")
    plt.plot(df["episode"], df["rolling_win_rate"] * 100, label="Win Rate (100 avg)")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Learning Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Training and Logging ===
def log_results(strategy_name: str, agent, env, num_episodes: int = 50000, max_steps: int = 100):
    log_path = init_logger(scenario=strategy_name)
    action_space = agent.action_space
    weights = agent.counting_weights
    gamma, alpha = agent.gamma, agent.alpha

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        win = False
        running_count = 0

        while not done and steps < max_steps:
            player_sum, dealer_card, usable_ace = state
            if "point_count" in agent.strategy:
                true_count = running_count // 1
                full_state = (player_sum, dealer_card, usable_ace, true_count)
            else:
                full_state = (player_sum, dealer_card, usable_ace)

            action = agent.choose_action(full_state, explore=True)
            next_state, reward, done, card_seen = env.step(action)

            # Update running count
            for card in card_seen:
                running_count += weights.get(card, 0)

            if "point_count" in agent.strategy:
                next_true_count = running_count // 1
                next_full_state = (next_state[0], next_state[1], next_state[2], next_true_count)
            else:
                next_full_state = (next_state[0], next_state[1], next_state[2])

            fs, nfs = tuple(full_state), tuple(next_full_state)
            if fs not in agent.q_table:
                agent.q_table[fs] = [0.0] * len(action_space)
            if nfs not in agent.q_table:
                agent.q_table[nfs] = [0.0] * len(action_space)

            best_next = max(agent.q_table[nfs])
            td_target = reward + gamma * best_next
            agent.q_table[fs][action] += alpha * (td_target - agent.q_table[fs][action])

            state = next_state
            total_reward += reward
            steps += 1
            win = win or reward > 0

        log_episode(log_path, episode, total_reward, steps, win)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        if episode % 5000 == 0:
            cprint(f"[{strategy_name}] Ep {episode:6d} | Avg R: {total_reward/steps:.4f} | "
                   f"Eps: {agent.epsilon:.4f} | Q-States: {len(agent.q_table)}", "green")

    return log_path

# === Evaluation (No Exploration) ===
def evaluate_agent(agent, env, eval_episodes=10000, max_steps=100):
    total_reward = 0
    wins = 0

    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.choose_action(state, explore=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            steps += 1

        total_reward += reward
        if reward > 0:
            wins += 1

    avg_reward = total_reward / eval_episodes
    win_rate = (wins / eval_episodes) * 100
    profit_per_hour = avg_reward * 100  # assume 100 hands/hour
    return avg_reward, win_rate, profit_per_hour

# === Q-table Persistence ===
def save_q_table(agent, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(agent.q_table, f)

def load_q_table(agent, path):
    import pickle
    with open(path, 'rb') as f:
        agent.q_table = pickle.load(f)