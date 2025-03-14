import os
import argparse
import logging
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Dict
import time
import copy
from tqdm import tqdm, trange

from golf_game import GolfGame
from deep_cfr_agent import DeepCFRAgent

def get_run_dir():
    """Generate a unique run directory name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('runs', f'cfr_run_{timestamp}')
    return run_dir

def setup_logging(log_level, run_dir):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create logs directory within run directory
    logs_dir = os.path.join(run_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a log file
    log_file = os.path.join(logs_dir, "training.log")
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("golf-ai-cfr")
    logger.info(f"Logging to {log_file}")
    return logger, logs_dir

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Deep CFR agent to play Golf')
    
    # Get default run directory
    default_run_dir = get_run_dir()
    
    parser.add_argument('--run-dir', type=str, default=default_run_dir,
                        help='Directory for this training run (default: auto-generated with timestamp)')
    parser.add_argument('--iterations', type=int, default=50000, help='Number of CFR iterations')
    parser.add_argument('--traversals', type=int, default=250, help='Number of traversals per iteration')
    parser.add_argument('--advantage-steps', type=int, default=200, help='Number of training steps for advantage network')
    parser.add_argument('--strategy-steps', type=int, default=200, help='Number of training steps for strategy network')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--hidden-size', type=int, default=256, help='Hidden size of the neural network')
    parser.add_argument('--embedding-dim', type=int, default=8, help='Dimension of card embeddings')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration parameter')
    parser.add_argument('--eval-interval', type=int, default=10, help='Evaluation interval (iterations)')
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='Early stopping patience for training')
    parser.add_argument('--early-stopping-threshold', type=float, default=0.000001, help='Threshold for early stopping')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load model from')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    # Optimization arguments
    parser.add_argument('--mixed-precision', action='store_true', 
                        help='Enable mixed precision training for faster computation')
    parser.add_argument('--pin-memory', action='store_true', 
                        help='Use pinned memory for faster CPU-GPU transfers')
    
    return parser.parse_args()

def evaluate_agent(agent, num_episodes=100, logger=None):
    """Evaluate the agent against a random opponent."""
    env = GolfGame(num_players=2)
    
    rewards = []
    wins = 0
    losses = 0
    max_turns_reached = 0
    agent_scores = []
    opponent_scores = []
    
    # Use tqdm for evaluation progress
    for episode in tqdm(range(num_episodes), desc="Evaluating", leave=False):
        state = env.reset()
        done = False
        
        while not done:
            # Agent's turn
            if env.current_player == 0:
                valid_actions = env._get_valid_actions()
                action = agent.select_action(state, valid_actions, player=0, training=False)
                state, _, done, info = env.step(action)  # Ignore immediate reward
            # Random opponent's turn
            else:
                valid_actions = env._get_valid_actions()
                action = DeepCFRAgent.random_action(valid_actions)
                state, _, done, info = env.step(action)  # Ignore immediate reward
        
        # Calculate final scores
        agent_score = env._calculate_score(0)
        opponent_score = env._calculate_score(1)
        agent_scores.append(agent_score)
        opponent_scores.append(opponent_score)
        
        # Determine winner and calculate final payoff
        if agent_score < opponent_score:
            wins += 1
            rewards.append(1.0)  # Agent wins
        elif agent_score > opponent_score:
            losses += 1
            rewards.append(-1.0)  # Agent loses
        else:
            rewards.append(0.0)  # Tie
        
        # Check if max turns reached
        if env.turn_count >= env.max_turns:
            max_turns_reached += 1
    
    # Calculate statistics
    avg_reward = np.mean(rewards)
    win_rate = wins / num_episodes
    loss_rate = losses / num_episodes
    max_turns_rate = max_turns_reached / num_episodes
    avg_agent_score = np.mean(agent_scores)
    avg_opponent_score = np.mean(opponent_scores)
    score_diff = avg_opponent_score - avg_agent_score
    
    if logger:
        logger.info(f"Evaluation results - Win rate: {win_rate:.2f}, Loss rate: {loss_rate:.2f}, Tie rate: {1-win_rate-loss_rate:.2f}")
        logger.info(f"Average Golf Scores - Agent: {avg_agent_score:.2f}, Opponent: {avg_opponent_score:.2f}, Diff: {score_diff:.2f}")
        logger.info(f"Max turns reached in {max_turns_rate:.2f} of games")
    
    return avg_reward, win_rate, loss_rate, max_turns_rate, avg_agent_score, score_diff

def train(args, logger, logs_dir):
    """Train the Deep CFR agent."""
    logger.info(f"Starting training with args: {args}")
    
    # Create model save directory within run directory
    models_dir = os.path.join(os.path.dirname(logs_dir), 'models')
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Saving models to {models_dir}")
    
    # Create charts directory within run directory
    charts_dir = os.path.join(os.path.dirname(logs_dir), 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    logger.info(f"Saving charts to {charts_dir}")
    
    # Initialize environment and agent
    env = GolfGame(num_players=2)
    state_size = 29  # 14 card indices + 15 binary features (including turn progress)
    action_size = 9   # Number of possible actions
    logger.info(f"Environment initialized with state_size={state_size}, action_size={action_size}")
    
    # Initialize Deep CFR agent
    agent = DeepCFRAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        num_card_ranks=13,  # Number of possible card ranks (A-K)
        learning_rate=args.lr,
        batch_size=args.batch_size,
        cfr_iterations=args.iterations,
        traversals_per_iter=args.traversals,
        advantage_train_steps=args.advantage_steps,
        strategy_train_steps=args.strategy_steps,
        epsilon=args.epsilon,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold
    )
    
    # Apply optimization settings
    if args.mixed_precision and agent.device.type == 'cuda':
        agent.use_amp = True
        logger.info("Enabled mixed precision training for faster computation")
    
    # Log device information
    device_info = f"Using device: {agent.device}"
    if agent.device.type == 'cuda':
        device_info += f" ({torch.cuda.get_device_name(0)})"
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA detected - Enabled cuDNN benchmarking for faster training")
        
        # Set PyTorch to use TensorFloat-32 (TF32) on Ampere GPUs
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Ampere GPU detected - Using TensorFloat-32 for faster training")
            
        # Use pinned memory for faster CPU-GPU transfers
        if args.pin_memory:
            logger.info("Using pinned memory for faster CPU-GPU transfers")
    logger.info(device_info)
    
    # Load model if specified
    if args.load_model and os.path.exists(args.load_model):
        agent.load(args.load_model)
        logger.info(f"Loaded model from {args.load_model}")
    
    # Metrics tracking
    advantage_losses_0 = []
    advantage_losses_1 = []
    strategy_losses = []
    
    # Evaluation metrics
    eval_rewards = []
    eval_win_rates = []
    eval_loss_rates = []
    eval_max_turns_rates = []
    eval_avg_scores = []
    eval_score_diffs = []
    
    # Best model tracking
    best_eval_win_rate = float('-inf')
    
    # EMA calculation function
    def calculate_ema(data, alpha=0.1):
        """Calculate Exponential Moving Average with the given alpha."""
        ema = np.zeros_like(data, dtype=float)
        if len(data) > 0:
            ema[0] = data[0]
            for i in range(1, len(data)):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    # Function to save charts
    def save_charts(iteration):
        plt.figure(figsize=(15, 15))
        
        # Row 1: Training losses
        plt.subplot(3, 2, 1)
        if advantage_losses_0:
            plt.plot(advantage_losses_0, alpha=0.5, label='Advantage Loss (P0)')
            plt.plot(calculate_ema(advantage_losses_0), color='red', linewidth=2, label='EMA (P0)')
        if advantage_losses_1:
            plt.plot(advantage_losses_1, alpha=0.5, label='Advantage Loss (P1)')
            plt.plot(calculate_ema(advantage_losses_1), color='blue', linewidth=2, label='EMA (P1)')
        plt.title('Advantage Network Losses')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 2, 2)
        if strategy_losses:
            plt.plot(strategy_losses, alpha=0.5, label='Strategy Loss')
            plt.plot(calculate_ema(strategy_losses), color='green', linewidth=2, label='EMA')
        plt.title('Strategy Network Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Row 2: Evaluation metrics
        plt.subplot(3, 2, 3)
        if eval_win_rates:
            plt.plot(eval_win_rates, 'g-', label='Win Rate')
            plt.plot(eval_loss_rates, 'r-', label='Loss Rate')
            plt.plot([1 - w - l for w, l in zip(eval_win_rates, eval_loss_rates)], 'b-', label='Tie Rate')
        plt.title('Win/Loss/Tie Rates')
        plt.xlabel('Evaluation')
        plt.ylabel('Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 2, 4)
        if eval_avg_scores:
            plt.plot(eval_avg_scores, 'b-', label='Agent Score')
            plt.plot([s - d for s, d in zip(eval_avg_scores, eval_score_diffs)], 'r-', label='Opponent Score')
        plt.title('Average Golf Scores')
        plt.xlabel('Evaluation')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Row 3: Additional metrics
        plt.subplot(3, 2, 5)
        if eval_rewards:
            plt.plot(eval_rewards, 'g-', label='Reward')
            plt.plot(calculate_ema(eval_rewards), color='red', linewidth=2, label='EMA')
        plt.title('Average Reward')
        plt.xlabel('Evaluation')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 2, 6)
        if eval_max_turns_rates:
            plt.plot(eval_max_turns_rates, 'r-', label='Max Turns Rate')
        plt.title('Max Turns Reached Rate')
        plt.xlabel('Evaluation')
        plt.ylabel('Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f'deep_cfr_training_charts_{iteration}.png'))
        plt.close()
    
    # Main training loop
    start_time = time.time()
    
    # Create a tqdm progress bar for iterations
    pbar = trange(1, args.iterations + 1, desc="Training CFR", leave=True)
    
    for iteration in pbar:
        # Train the agent for one iteration
        losses = agent.train(env)
        
        # Track losses
        if losses['advantage_loss_0'] is not None:
            advantage_losses_0.append(losses['advantage_loss_0'])
        if losses['advantage_loss_1'] is not None:
            advantage_losses_1.append(losses['advantage_loss_1'])
        if losses['strategy_loss'] is not None:
            strategy_losses.append(losses['strategy_loss'])
        
        # Update progress bar with loss information
        pbar.set_postfix({
            'Adv0': f"{losses['advantage_loss_0']:.4f}" if losses['advantage_loss_0'] is not None else "N/A",
            'Adv1': f"{losses['advantage_loss_1']:.4f}" if losses['advantage_loss_1'] is not None else "N/A",
            'Strat': f"{losses['strategy_loss']:.4f}" if losses['strategy_loss'] is not None else "N/A"
        })
        
        # Log progress less frequently to avoid cluttering the console
        if iteration % 10 == 0:
            elapsed_time = time.time() - start_time
            iterations_per_second = iteration / elapsed_time if elapsed_time > 0 else 0
            estimated_time_remaining = (args.iterations - iteration) / iterations_per_second if iterations_per_second > 0 else 0
            
            logger.info(f"Iteration {iteration}/{args.iterations} - "
                       f"Adv Loss P0: {losses['advantage_loss_0']:.6f}, "
                       f"Adv Loss P1: {losses['advantage_loss_1']:.6f}, "
                       f"Strat Loss: {losses['strategy_loss']:.6f}, "
                       f"Speed: {iterations_per_second:.2f} it/s, "
                       f"ETA: {estimated_time_remaining/60:.1f} min")
        
        # Evaluate the agent
        if iteration % args.eval_interval == 0:
            logger.info(f"Evaluating agent at iteration {iteration}...")
            avg_reward, win_rate, loss_rate, max_turns_rate, avg_agent_score, score_diff = evaluate_agent(
                agent, num_episodes=100, logger=logger
            )
            
            # Update progress bar with evaluation results
            pbar.set_postfix({
                'WinRate': f"{win_rate:.2f}",
                'Adv0': f"{losses['advantage_loss_0']:.4f}" if losses['advantage_loss_0'] is not None else "N/A",
                'Strat': f"{losses['strategy_loss']:.4f}" if losses['strategy_loss'] is not None else "N/A"
            })
            
            # Track evaluation metrics
            eval_rewards.append(avg_reward)
            eval_win_rates.append(win_rate)
            eval_loss_rates.append(loss_rate)
            eval_max_turns_rates.append(max_turns_rate)
            eval_avg_scores.append(avg_agent_score)
            eval_score_diffs.append(score_diff)
            
            # Save best model
            if win_rate > best_eval_win_rate:
                best_eval_win_rate = win_rate
                best_model_path = os.path.join(models_dir, f'deep_cfr_best.pt')
                agent.save(best_model_path)
                logger.info(f"New best model saved with win rate: {win_rate:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(models_dir, f'deep_cfr_iteration_{iteration}.pt')
            agent.save(checkpoint_path)
            logger.info(f"Checkpoint saved at iteration {iteration}")
            
            # Save charts
            save_charts(iteration)
    
    # Final evaluation with tqdm
    logger.info("Final evaluation...")
    avg_reward, win_rate, loss_rate, max_turns_rate, avg_agent_score, score_diff = evaluate_agent(
        agent, num_episodes=200, logger=logger
    )
    
    # Save final model
    final_model_path = os.path.join(models_dir, 'deep_cfr_final.pt')
    agent.save(final_model_path)
    logger.info(f"Final model saved with win rate: {win_rate:.4f}")
    
    # Save final charts
    save_charts(args.iterations)
    
    # Training complete
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.2f} minutes")
    logger.info(f"Best win rate: {best_eval_win_rate:.4f}")
    logger.info(f"Final win rate: {win_rate:.4f}")

if __name__ == "__main__":
    args = parse_args()
    
    # Create run directory structure
    os.makedirs(args.run_dir, exist_ok=True)
    logger, logs_dir = setup_logging(args.log_level, args.run_dir)
    
    try:
        logger.info(f"Starting Deep CFR training in run directory: {args.run_dir}")
        train(args, logger, logs_dir)
        logger.info("Training complete")
    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")
        raise 