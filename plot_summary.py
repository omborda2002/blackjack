import pandas as pd
import matplotlib.pyplot as plt
import os

log_dir = "logs"
summary_files = [f for f in os.listdir(log_dir) if f.endswith("_summary.csv")]

summary_data = []
for file in summary_files:
    df = pd.read_csv(os.path.join(log_dir, file))
    summary_data.append(df)

combined = pd.concat(summary_data, ignore_index=True)

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.bar(combined['strategy'], combined['avg_reward'])
plt.title("Average Reward per Strategy")
plt.ylabel("Avg Reward")

plt.subplot(3, 1, 2)
plt.bar(combined['strategy'], combined['win_rate'])
plt.title("Win Rate (%) per Strategy")
plt.ylabel("Win Rate")

plt.subplot(3, 1, 3)
plt.bar(combined['strategy'], combined['profit'])
plt.title("Profit per Hour ($) per Strategy")
plt.ylabel("Profit")
plt.xlabel("Strategy")

plt.tight_layout()
plt.show()
