# Run Experiment
from blackjack_env import BlackjackEnv
from q_learning_agent import QLearningAgent
from utils import init_logger, log_episode, summarize_logs, plot_logs

try:
    from termcolor import colored
except ImportError:
    def colored(text, color):
        return text

def train_agent(
    num_episodes=20000,
    scenario_name="basic_strategy",
    log_dir="logs",
    use_counting=False,
    dealer_hits_soft_17=True,
    blackjack_payout=(1, 1),
    use_true_count=False,
    decks=1
):
    env = BlackjackEnv(
        use_counting=use_counting,
        dealer_hits_soft_17=dealer_hits_soft_17,
        blackjack_payout=blackjack_payout,
        use_true_count=use_true_count,
        decks=decks
    )
    agent = QLearningAgent(
        actions=env.action_space,
        alpha=0.05,           # Lower learning rate for stability
        gamma=0.95,           # Higher discount factor
        epsilon=1.0,          # Start with full exploration
        epsilon_decay=0.9999, # Slower decay for more exploration
        epsilon_min=0.01      # Lower minimum for less randomness
    )
    log_path = init_logger(log_dir=log_dir, scenario=scenario_name)

    print(colored(f"\n▶ Training Started: {scenario_name}", "blue"))
    
    # Track performance over time
    performance_window = []
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        log_episode(log_path, episode, total_reward, steps, total_reward > 0)
        
        # Track recent performance
        performance_window.append(total_reward)
        if len(performance_window) > 1000:
            performance_window.pop(0)

        if episode % 5000 == 0:
            recent_avg = sum(performance_window) / len(performance_window)
            print(f"[{scenario_name}] Ep {episode:6d} | Recent Avg: {recent_avg:+.4f} | Eps: {agent.epsilon:.4f} | Q-States: {len(agent.q_table)}")
        elif episode % 1000 == 0:
            print(f"[{scenario_name}] Ep {episode:6d} | Reward: {total_reward:+4.1f} | Steps: {steps} | Eps: {agent.epsilon:.4f}")

    summary = summarize_logs(log_path)
    print(colored(f"\n✅ Completed: {scenario_name}", "green"))
    print(f"[Summary] Episodes: {summary['episodes']} | Avg Reward: {summary['avg_reward']:.3f} | Win Rate: {summary['win_rate']:.2%}")

    # Save trained model
    agent.save(f"{log_dir}/{scenario_name}_qtable.pkl")
    
    # Evaluation phase with no exploration
    print(f"🔍 Evaluating {scenario_name} (10,000 episodes)...")
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # No exploration during evaluation
    
    eval_rewards = []
    eval_wins = 0
    
    for eval_episode in range(10000):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
        if episode_reward > 0:
            eval_wins += 1
    
    agent.epsilon = original_epsilon  # Restore epsilon
    
    avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
    eval_win_rate = eval_wins / len(eval_rewards)
    
    print(f"📊 Evaluation Results:")
    print(f"    Average Reward: {avg_eval_reward:.4f}")
    print(f"    Win Rate: {eval_win_rate:.2%}")
    print(f"    Expected Profit/Hour: ${avg_eval_reward * 100:.2f}")
    
    plot_logs(log_path)


if __name__ == "__main__":
    # Increased episodes for better convergence
    print("Starting BlackJack RL Training with Enhanced Parameters")
    
    # Basic Strategy (50k episodes)
    train_agent(num_episodes=50000, scenario_name="basic_strategy")
    
    # Card Counting Strategy (75k episodes)
    train_agent(num_episodes=75000, scenario_name="counting_strategy", use_counting=True)
    
    # Rule Variation 1: Dealer stands on soft 17, better payout
    train_agent(
        num_episodes=50000,
        scenario_name="variant_strategy",
        use_counting=True,
        dealer_hits_soft_17=False,
        blackjack_payout=(3, 2)
    )
    
    # True Count Strategy with multiple decks
    train_agent(
        num_episodes=75000,
        scenario_name="true_count_strategy",
        use_counting=True,
        use_true_count=True,
        decks=6,
        dealer_hits_soft_17=False,
        blackjack_payout=(3, 2)
    )
    
    # Enhanced Strategy with all optimizations
    train_agent(
        num_episodes=100000,
        scenario_name="enhanced_strategy",
        use_counting=True,
        use_true_count=True,
        decks=6,
        dealer_hits_soft_17=False,
        blackjack_payout=(3, 2)
    )
