import os
import csv
from datetime import datetime
from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd

def init_logger(log_dir: str = "logs", scenario: str = "default") -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{scenario}_{timestamp}.csv"
    path = os.path.join(log_dir, filename)
    
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "total_reward", "steps", "win"])
    return path

def log_episode(log_path: str, episode: int, total_reward: int, steps: int, win: bool) -> None:
    with open(log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, total_reward, steps, int(win)])

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
