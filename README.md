# BlackJack Reinforcement Learning Project

This project implements a reinforcement learning-based BlackJack player as part of Portfolio Exam 3 for the course "Reasoning and Decision Making under Uncertainty" at THWS.

## Overview

The implementation includes:

1. **Basic Strategy Learning** - Q-learning agent that learns optimal BlackJack basic strategy
2. **Complete Point-Count System** - Advanced agent with Hi-Lo card counting capabilities  
3. **Rule Variations Analysis** - Testing different BlackJack rule sets and their impact
4. **Enhanced Profit System** - Optimized system for maximum expected profit

## Project Structure

```
blackjack_rl/
├── blackjack_env.py        # BlackJack environment with rule variations
├── q_learning_agent.py     # Q-learning agents (basic + card counting)
├── run_experiment.py       # Main experiment runner
├── utils.py                # Utilities for logging and visualization
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── logs/                  # Experiment results and logs
```

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- JSON (built-in)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run All Experiments
```bash
python run_experiment.py
```

### Run Specific Experiment
```bash
python run_experiment.py --experiment basic      # Basic strategy only
python run_experiment.py --experiment counting   # Card counting only
python run_experiment.py --experiment variations # Rule variations only
python run_experiment.py --experiment enhanced   # Enhanced system only
```

### Quick Test Run
```bash
python run_experiment.py --quick
```

## Experiments

### 1. Basic Strategy Learning
- **Objective**: Learn optimal basic BlackJack strategy using Q-learning
- **Agent**: Standard Q-learning with epsilon-greedy exploration
- **Training**: 50,000 episodes
- **Evaluation**: 10,000 episodes
- **Expected Results**: Win rate ~43-47%, slight negative expected value

### 2. Complete Point-Count System
- **Objective**: Implement Hi-Lo card counting with Q-learning
- **Agent**: Enhanced Q-learning with true count in state space
- **Features**: 
  - Running count tracking
  - True count calculation
  - Betting strategy based on count
- **Training**: 75,000 episodes
- **Expected Results**: Positive expected value with proper betting

### 3. Rule Variations Analysis
Tests two rule variations and their impact on strategy:

#### Variation 1: Surrender Allowed
- **Rule**: Player can surrender hand for 50% loss
- **Impact**: Should improve player's expected value slightly
- **Strategy Changes**: Surrender on hard 16 vs dealer 9,10,A

#### Variation 2: No Double After Split
- **Rule**: Cannot double down after splitting pairs
- **Impact**: Reduces player's expected value
- **Strategy Changes**: More conservative splitting strategy

### 4. Enhanced Profit System
- **Objective**: Maximize expected profit through optimized parameters
- **Features**:
  - Optimized deck composition (4 decks, 80% penetration)
  - Fine-tuned learning parameters
  - Extended training (100,000 episodes)
  - Advanced betting strategy

## Results and Analysis

After running experiments, results are saved in the `logs/` directory with timestamp:

- **Training Progress Plots**: Learning curves, epsilon decay, win rates
- **Strategy Comparison**: Performance comparison across all methods
- **Rule Variation Analysis**: Impact of different rules on performance
- **Results Summary**: Comprehensive JSON and text reports
- **Model Files**: Trained Q-tables for each agent

## Key Metrics

For each strategy, the following metrics are calculated:

- **Win Rate**: Percentage of hands won
- **Average Reward**: Expected reward per hand
- **Expected Profit**: Estimated profit per hour of play
- **Q-table Size**: Number of unique states learned

## Implementation Details

### State Representation
- **Basic Agent**: (player_sum, dealer_up_card, usable_ace)
- **Card Counting Agent**: (player_sum, dealer_up_card, usable_ace, true_count, action_flags)

### Action Space
- 0: Stand
- 1: Hit  
- 2: Double Down
- 3: Split (when applicable)
- 4: Surrender (when applicable)

### Card Counting System
- **Hi-Lo System**: +1 for low cards (2-6), -1 for high cards (10,J,Q,K,A), 0 for neutral (7,8,9)
- **True Count**: Running count divided by estimated decks remaining
- **Betting Strategy**: Bet size scales with positive true count

## Expected Performance

Based on theoretical BlackJack analysis:

1. **Basic Strategy**: ~-0.5% house edge (win rate ~43-47%)
2. **Card Counting**: ~+0.5% to +1.5% player edge with proper betting
3. **Rule Variations**: 
   - Surrender: Improves by ~0.1%
   - No DAS: Reduces by ~0.1-0.2%
4. **Enhanced System**: Target +1% to +2% player edge

## Academic Context

This implementation addresses the Portfolio Exam 3 requirements:

1. ✅ **Basic Strategy Learning** - Q-learning implementation
2. ✅ **Complete Point-Count System** - Hi-Lo card counting integration
3. ✅ **Rule Variations** - Surrender and Double-After-Split analysis
4. ✅ **Profit Optimization** - Enhanced system with improved parameters
5. ✅ **No External RL Frameworks** - Pure Python implementation
6. ✅ **Comprehensive Logging** - Detailed results and analysis

## Usage Notes

- The system is optimized for research, not human play
- Card counting is legal but may be restricted in casinos
- Results are based on simulation and may vary with different random seeds
- For academic purposes only

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
- Thorp, E. O. (1966). Beat the Dealer
- Basic Strategy charts and optimal play analysis

## Author

Student Implementation for THWS Portfolio Exam 3  
Course: Reasoning and Decision Making under Uncertainty  
Summer 2025