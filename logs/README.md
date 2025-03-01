# Training Logs Directory

This directory contains logs and charts from training runs of the Golf Card Game AI.

## Directory Structure

Each training run creates a timestamped subdirectory with the following structure:

```
logs/
├── training_run_YYYYMMDD_HHMMSS/
│   ├── training_YYYYMMDD_HHMMSS.log    # Log file with training information
│   ├── training_curves_ep1000.png      # Training curves at episode 1000
│   ├── training_curves_ep2000.png      # Training curves at episode 2000
│   ├── ...
│   ├── training_curves_ep20000.png     # Final training curves
│   ├── training_metrics.csv            # CSV file with training metrics
│   └── eval_metrics.csv                # CSV file with evaluation metrics
```

## Metrics Tracked

The training process tracks and visualizes the following metrics:

1. **Episode Rewards**: Rewards received during training
2. **Training Loss**: Loss values during training
3. **Evaluation Rewards**: Average rewards during evaluation
4. **Win Rate vs Random Opponent**: Win rate against a random opponent
5. **Loss Rate vs Random Opponent**: Loss rate against a random opponent
6. **Max Turns Rate**: Rate of games reaching the maximum number of turns
7. **Average Golf Score**: Average golf score (lower is better)
8. **Score Difference**: Difference between agent's score and random opponent's score
9. **Learning Rate Adjustments**: Changes in learning rate during training

## Exploration Strategy

The agent uses a piecewise linear epsilon decay strategy for exploration:

1. **Warmup Phase**: For the first N episodes (controlled by `--epsilon-warmup`), epsilon remains at the starting value (typically 1.0) to encourage thorough exploration of the state space.
2. **Decay Phase**: After the warmup period, epsilon decays linearly from the starting value to the end value over a specified number of episodes (controlled by `--epsilon-decay-episodes`).

This strategy ensures that the agent explores thoroughly in the early stages of training before gradually focusing on exploitation of learned strategies. The linear decay makes it easier to understand and control the exploration schedule compared to exponential decay rates.

For example, with default settings:

- Epsilon starts at 1.0
- Remains at 1.0 for the first 5,000 episodes (warmup)
- Then linearly decays to 0.01 over the next 10,000 episodes
- Reaches minimum value of 0.01 after episode 15,000

## Using the Logs

The logs and charts can be used to:

1. Monitor training progress
2. Identify issues with training (e.g., high loss values)
3. Compare different training runs
4. Determine the best model for deployment

The CSV files can be used for further analysis in tools like Excel, Python, or R.
