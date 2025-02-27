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

## Using the Logs

The logs and charts can be used to:

1. Monitor training progress
2. Identify issues with training (e.g., high loss values)
3. Compare different training runs
4. Determine the best model for deployment

The CSV files can be used for further analysis in tools like Excel, Python, or R.
