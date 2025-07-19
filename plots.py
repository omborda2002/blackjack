import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

# === Settings ===
LOG_DIR = "logs"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# === Load CSVs ===
file_paths = sorted(glob(os.path.join(LOG_DIR, "*.csv")))
strategy_dfs = []

for path in file_paths:
    name = os.path.basename(path).split("_2025")[0]  # Extract strategy name
    df = pd.read_csv(path)
    df["strategy"] = name
    df["cumulative_reward"] = df["total_reward"].cumsum()
    strategy_dfs.append(df)

# === Combine and Plot ===
df_all = pd.concat(strategy_dfs, ignore_index=True)

# --- Cumulative Reward Plot ---
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_all, x="episode", y="cumulative_reward", hue="strategy")
plt.title("Cumulative Reward vs. Episode")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.legend(title="Strategy")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cumulative_reward_plot.png"), dpi=300)
plt.show()

# --- Win Rate Rolling Plot (optional) ---
df_all["rolling_winrate"] = df_all.groupby("strategy")["win"].transform(lambda x: x.rolling(100, min_periods=1).mean() * 100)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_all, x="episode", y="rolling_winrate", hue="strategy")
plt.title("Rolling Win Rate vs. Episode (window=100)")
plt.xlabel("Episode")
plt.ylabel("Win Rate (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rolling_winrate_plot.png"), dpi=300)
plt.show()

# === Summary Table ===
summary = df_all.groupby("strategy").agg({
    "total_reward": "mean",
    "win": "mean"
}).reset_index()
summary["Win Rate (%)"] = summary["win"] * 100
summary["Profit/hr (€)"] = summary["total_reward"] * 100
summary = summary.drop(columns="win")

summary.to_csv(os.path.join(OUT_DIR, "strategy_summary.csv"), index=False)
print("✅ Plots and CSV exported to /figures")