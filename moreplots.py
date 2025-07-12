import pandas as pd
import matplotlib.pyplot as plt
import os

log_dir = "logs"
log_files = [f for f in os.listdir(log_dir) if f.endswith(".csv") and not f.endswith("_summary.csv")]

reward_curves = {}
winrate_curves = {}

for file in log_files:
    strategy_name = file.split("_")[0]
    df = pd.read_csv(os.path.join(log_dir, file))
    df["rolling_reward"] = df["total_reward"].rolling(window=100, min_periods=1).mean()
    df["rolling_win"] = df["win"].rolling(window=100, min_periods=1).mean() * 100
    reward_curves[strategy_name] = df[["episode", "rolling_reward"]]
    winrate_curves[strategy_name] = df[["episode", "rolling_win"]]

plt.figure(figsize=(10, 5))
for strategy, data in reward_curves.items():
    plt.plot(data["episode"], data["rolling_reward"], label=strategy)
plt.title("Rolling Average Total Reward (window=100)")
plt.xlabel("Episode")
plt.ylabel("Avg Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
for strategy, data in winrate_curves.items():
    plt.plot(data["episode"], data["rolling_win"], label=strategy)
plt.title("Rolling Win Rate (%) (window=100)")
plt.xlabel("Episode")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
