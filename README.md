# Enhanced Blackjack Reinforcement Learning System

## Overview

This enhanced blackjack RL system implements multiple difficulty levels, from player-favorable basic strategy to the most challenging casino conditions possible. The system is designed to evaluate how different rule variations affect optimal play and expected returns.

## Key Features

### 🎯 Multiple Difficulty Levels
- **Basic Strategy**: Standard favorable rules for baseline comparison
- **Counting Strategy**: Card counting with reasonable casino rules
- **Tough Casino**: Moderately unfavorable rules
- **Toughest Casino**: Extremely unfavorable rules with multiple disadvantages
- **Nightmare Casino**: The ultimate challenge with maximum house edge

### 🃏 Advanced Casino Rules Implementation
- **Continuous Shuffling Machine**: Completely neutralizes card counting
- **Dealer Wins Ties**: Pushes become losses for the player
- **Restricted Doubling**: Only allowed on 9, 10, 11
- **Reduced Blackjack Payouts**: From 3:2 down to even money (1:1)
- **House Edge Boost**: Additional mathematical disadvantage
- **Multi-deck Shoes**: Up to 8 decks with varying shuffle penetration

### 🧠 Intelligent Agent Features
- **True Count Calculation**: Accurate card counting implementation
- **Bet Scaling**: Dynamic betting based on count (when not defeated by CSM)
- **State Space Optimization**: Efficient Q-table representation
- **Adaptive Learning**: Epsilon-greedy with decay

## Strategy Comparison

| Strategy | Decks | BJ Payout | Dealer S17 | Special Rules | Expected Difficulty |
|----------|-------|-----------|------------|---------------|-------------------|
| Basic | 1 | 3:2 | ✅ | None | Easy |
| Counting | 6 | 3:2 | ✅ | Card counting | Moderate |
| Tough | 8 | 6:5 | ❌ | Dealer hits soft 17 | Hard |
| Toughest | 8 | 6:5 | ❌ | CSM, Dealer wins ties, Limited doubling | Very Hard |
| Nightmare | 8 | 1:1 | ❌ | All toughest rules + Even money BJ | Extreme |

## Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Usage
```bash
# Run all strategies
python run_experiment.py

# Run specific strategy
python run_experiment.py --strategy toughest_casino

# Custom training parameters
python run_experiment.py --episodes 100000 --eval-episodes 20000
```

### Advanced Configuration
```python
# Example: Custom ultra-tough casino
env = BlackjackEnv(
    use_counting=True,
    use_true_count=True,
    use_bet_scaling=True,
    toughest=True,
    dealer_hits_soft_17=True,
    blackjack_payout=(1, 1),  # Even money
    decks=8
)
```

## What Makes "Toughest Casino" Actually Tough

The original implementation had several issues that made it less challenging:

### ❌ Original Issues:
1. **Blackjack payout contradiction**: Set to (1,1) then overridden to (6,5)
2. **Limited rule enforcement**: Many tough rules weren't actually implemented
3. **Ineffective card counting countermeasures**: Simple deck management
4. **Missing key disadvantages**: No tie-breaking rules, limited house edge manipulation

### ✅ Enhanced Implementation:
1. **Continuous Shuffling Machine**: Shuffles every 10 cards, completely defeating card counting
2. **Dealer Wins Ties**: All pushes become losses (-100% on ties)
3. **Restricted Doubling**: Only on 9, 10, 11 (industry worst practice)
4. **Reduced Blackjack Payouts**: Down to 6:5 or even 1:1 (even money)
5. **House Edge Boost**: Mathematical 2% additional disadvantage on losses
6. **Bet Scaling Limits**: Maximum 2x bet ratio instead of 5x
7. **Blackjack Frequency Reduction**: Some face cards randomly become 9s
8. **Late Shuffle Point**: 75% penetration hurts counting effectiveness

## Expected Results

With the enhanced tough rules, you should see:

```
📊 Expected Performance Ranking (worst to best):
1. nightmare_casino:  -0.15 to -0.20 avg reward
2. toughest_casino:   -0.12 to -0.18 avg reward  
3. tough_casino:      -0.08 to -0.12 avg reward
4. counting_strategy: -0.02 to +0.02 avg reward
5. basic_strategy:    -0.01 to +0.01 avg reward
```

## Research Applications

This system is perfect for academic research on:
- **Rule Variation Impact**: How individual rule changes affect optimal strategy
- **Card Counting Effectiveness**: Measuring countermeasure success
- **Reinforcement Learning Robustness**: Agent adaptation to adverse conditions
- **Casino Game Theory**: Mathematical analysis of house edge modifications

## File Structure

```
blackjack_rl/
├── blackjack_env.py        # Enhanced environment with tough rules
├── q_learning_agent.py     # Q-learning implementation
├── run_experiment.py       # Enhanced experiment runner
├── utils.py               # Logging and evaluation utilities
├── requirements.txt       # Dependencies
└── logs/                 # Training logs and results
```

## Advanced Features

### 🔬 Research Mode
```python
# Enable detailed logging for research
env = BlackjackEnv(toughest=True, reward_shaping=True)
agent = QLearningAgent(env, alpha=0.05, epsilon_decay=0.9995)
```

### 📊 Analysis Tools
- **Q-table size tracking**: Monitor state space exploration
- **Win rate analysis**: Detailed performance metrics
- **Profit/hour calculation**: Real-world applicability
- **Rule impact measurement**: Isolate individual rule effects

## Contributing

When adding new rules or modifications:
1. Ensure mathematical correctness of house edge calculations
2. Test against known basic strategy benchmarks
3. Document rule interactions and precedence
4. Add appropriate evaluation metrics

## License

This implementation is for educational and research purposes. Please check local gambling regulations before any commercial use.

---

*"The house always wins... but how much depends on the rules."*