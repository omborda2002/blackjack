import argparse
import os
from termcolor import cprint
from blackjack_env import BlackjackEnv
from q_learning_agent import QLearningAgent
from utils import evaluate_agent, log_results, save_q_table

def train_agent(strategy_name, env_kwargs, num_episodes=50000, eval_episodes=10000):
    cprint(f"▶ Training Started: {strategy_name}", "cyan", attrs=["bold"])
    env = BlackjackEnv(**env_kwargs)  # Create environment with specific rule variations
    agent = QLearningAgent(env, strategy=strategy_name)  # Initialize agent with strategy

    # Train agent and log results
    log_path = log_results(strategy_name, agent, env, num_episodes)
    save_q_table(agent, f"logs/{strategy_name}_q_table.pkl")
    cprint(f"✅ Completed: {strategy_name}", "green")

    # Evaluate agent performance (greedy policy)
    agent.epsilon = 0.0
    cprint(f"🔍 Evaluating {strategy_name} ({eval_episodes:,} episodes)...", "blue")
    avg_reward, win_rate, profit_per_hour = evaluate_agent(agent, env, eval_episodes)

    # Print evaluation summary
    print("📊 Evaluation Results:")
    print(f"Avg Reward: {avg_reward:.3f} | Win Rate: {win_rate:.2f}% | Profit/hr: {profit_per_hour:.2f} €")

def run_all_scenarios():
    # List of strategies and corresponding environment settings
    scenarios = [
        ("basic_strategy", {}),
        ("point_count", {}),
        ("variation_hits_soft_17", {"dealer_hits_soft_17": True}),
        ("variation_double_after_split", {"double_after_split": True}),
        ("improved_point_count", {})
    ]

    for strategy_name, env_kwargs in scenarios:
        train_agent(strategy_name, env_kwargs)

if __name__ == "__main__":
    run_all_scenarios()