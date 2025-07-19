# Blackjack Reinforcement Learning Agent

This project implements a self-learning Blackjack agent using **Q-Learning** and explores various strategies including **card counting** and **rule variations**. The agent learns optimal play policies through simulation and logs performance metrics for analysis.

---

## 📁 Project Structure

```
blackjack_rl/
├── blackjack_env.py              # Blackjack game environment
├── q_learning_agent.py          # Q-learning agent with optional counting
├── run_experiment.py            # Runs all strategy experiments
├── utils.py                     # Logging, evaluation, saving/loading Q-tables
├── plots.py                     # Generates reward/win rate plots and summary CSV
├── requirements.txt             # Dependencies list
├── README.md                    # This file
├── figures/                     # Output plots + summary CSV
│   ├── cumulative_reward_plot.png
│   ├── rolling_winrate_plot.png
│   └── strategy_summary.csv
├── logs/                        # Training logs and Q-tables
│   ├── *.csv
│   ├── *_q_table.pkl
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train All Strategies

```bash
python run_experiment.py
```

This will train and evaluate five strategies:

* Basic Strategy
* Point Count (Hi-Lo)
* Dealer Hits on Soft 17
* Double After Split Allowed
* Improved Point Count (custom weights)

### 3. Generate Plots & Summary Table

```bash
python plots.py
```

This will create:

* `figures/cumulative_reward_plot.png`
* `figures/rolling_winrate_plot.png`
* `figures/strategy_summary.csv`

---

## 📊 Evaluation Metrics

After each training run, the agent is evaluated on:

* **Average Reward per Game**
* **Win Rate (%)**
* **Profit per Hour (€)** (assuming 100 hands/hour)

These metrics appear both in console output and the summary CSV.

---

## 🧠 Strategy Modes Explained

| Strategy                       | Description                                         |
| ------------------------------ | --------------------------------------------------- |
| `basic_strategy`               | Pure state-based Q-learning (no memory)             |
| `point_count`                  | Adds Hi-Lo running count as a state feature         |
| `variation_hits_soft_17`       | Dealer hits on soft 17 rule tested                  |
| `variation_double_after_split` | Double-down allowed after splits                    |
| `improved_point_count`         | Uses custom weights (e.g. +2 for 5,6; -2 for 10,11) |

---

## 💾 Reproducibility

* Logs are saved as `.csv` in `logs/`
* Q-tables are saved as `.pkl` files per strategy
* All results are deterministic based on random seed unless modified

---

## 📌 Notes

* All agents use **epsilon-greedy** exploration with decay
* Bets are scaled based on the **true count** if enabled in the environment
* No external RL libraries were used (per exam rules)

---

## 👨‍🏫 Author

**Om Sanjaybhai Borda**
Technical University of Applied Sciences Würzburg-Schweinfurt
Email: [om.borda@study.thws.de](mailto:om.borda@study.thws.de)

---

## 📜 License

This project is part of the Portfolio Exam 3 for *Reasoning and Decision Making under Uncertainty (Summer 2025)* and is submitted to Prof. Dr. Frank Deinzer.
