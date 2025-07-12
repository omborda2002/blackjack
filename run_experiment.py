import argparse
import os
from termcolor import cprint
from blackjack_env import BlackjackEnv
from q_learning_agent import QLearningAgent
from utils import evaluate_agent, log_results, save_q_table


def train_agent(strategy_name, env_kwargs, num_episodes=50000, eval_episodes=10000):
    cprint(f"▶ Training Started: {strategy_name}", "cyan", attrs=["bold"])
    env = BlackjackEnv(**env_kwargs)
    agent = QLearningAgent(env)
    log_path = log_results(strategy_name, agent, env, num_episodes)

    save_q_table(agent, f"logs/{strategy_name}_q_table.pkl")

    cprint(f"✅ Completed: {strategy_name}", "green")
    cprint(f"🔍 Evaluating {strategy_name} ({eval_episodes:,} episodes)...", "blue")
    avg_reward, win_rate, profit_per_hour = evaluate_agent(agent, env, eval_episodes)

    print("📊 Evaluation Results:")
    with open(f"logs/{strategy_name}_summary.csv", "w") as f:
        f.write("strategy,avg_reward,win_rate,profit,q_table_size\n")
        f.write(f"{strategy_name},{avg_reward:.4f},{win_rate:.2f},{profit_per_hour:.2f},{len(agent.q_table)}\n")


    print(f"    Average Reward: {avg_reward:.4f}")
    print(f"    Win Rate: {win_rate:.2f}%")
    print(f"    Expected Profit/Hour: ${profit_per_hour:.2f}")
    print(f"    Q-Table Size: {len(agent.q_table)} states")
    print(f"    Final Epsilon: {agent.epsilon:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="all",
                        help="Strategy to train: basic_strategy, counting_strategy, tough_casino, or all")
    parser.add_argument("--episodes", type=int, default=50000, help="Number of training episodes")
    parser.add_argument("--eval_episodes", type=int, default=10000, help="Number of evaluation episodes")
    args = parser.parse_args()

    strategies = {
        "basic_strategy": {
            "use_counting": False,
        },
        "counting_strategy": {
            "use_counting": True,
            "use_bet_scaling": True,
        },
        "tough_casino": {
            "use_counting": True,
            "toughest": True,
        },
    }

    if args.strategy == "all":
        selected = strategies.items()
    elif args.strategy in strategies:
        selected = [(args.strategy, strategies[args.strategy])]
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    for strategy_name, env_kwargs in selected:
        train_agent(strategy_name, env_kwargs, args.episodes, args.eval_episodes)


if __name__ == "__main__":
    main()
