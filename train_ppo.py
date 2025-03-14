import os
import numpy as np
import torch
import argparse
import time
import logging
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
from golf_game import GolfGame
from golf_game_v2 import GameInfo
from ppo_agent import PPOAgent
from training_utils import get_run_dir, setup_logging, setup_run_directories, calculate_ema

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a PPO agent for Golf card game')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=50000, help='Number of episodes to train')
    parser.add_argument('--update_interval', type=int, default=2048, help='Number of steps between PPO updates')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs per update')
    parser.add_argument('--hidden_size', type=int, default=512, help='Size of hidden layers')
    parser.add_argument('--actor_lr', type=float, default=0.0003, help='Learning rate for actor')
    parser.add_argument('--critic_lr', type=float, default=0.0003, help='Learning rate for critic')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--policy_clip', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Gradient clipping')
    
    # Evaluation parameters
    parser.add_argument('--eval_interval', type=int, default=100, help='Episodes between evaluations')
    parser.add_argument('--eval_episodes', type=int, default=100, help='Number of episodes for evaluation')
    
    # Saving parameters
    parser.add_argument('--save_interval', type=int, default=5000, help='Episodes between saving model')
    parser.add_argument('--run_dir', type=str, default=None, help='Directory for this training run (default: auto-generated with timestamp)')
    parser.add_argument('--model_name', type=str, default='ppo_golf', help='Base name for model files')
    
    # Visualization parameters
    parser.add_argument('--plot_interval', type=int, default=100, help='Episodes between plotting')
    
    # Logging parameters
    parser.add_argument('--log_interval', type=int, default=100, help='Episodes between logging')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level')
    
    # Load model
    parser.add_argument('--load_model', type=str, default=None, help='Path to load model from')
    
    return parser.parse_args()

def evaluate_agent(agent, env, num_episodes=100):
    """Evaluate agent performance."""
    total_rewards = []
    win_rate = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            valid_actions = env._get_valid_actions()
            action, _, _ = agent.select_action(state, valid_actions, training=False)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        
        # Check if agent won (player 0 has lower score than player 1)
        # Get scores from the info object if available
        if hasattr(info, 'scores') and info.scores and len(info.scores) >= 2:
            player0_score = info.scores[0]
            player1_score = info.scores[1]
        else:
            # Fall back to calculating scores from the environment
            player0_score = env._calculate_score(0)
            player1_score = env._calculate_score(1)
            
        if player0_score < player1_score:
            win_rate += 1
    
    avg_reward = np.mean(total_rewards)
    win_rate = win_rate / num_episodes
    
    return avg_reward, win_rate

def plot_learning_curve(x, scores, win_rates, losses=None, avg_scores=None, figure_file=None):
    """Plot and save learning curves."""
    fig = plt.figure(figsize=(12, 12))
    
    # Plot 1: Evaluation Rewards
    plt.subplot(3, 1, 1)
    plt.plot(x, scores)
    plt.title('Evaluation Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot 2: Win Rates
    plt.subplot(3, 1, 2)
    plt.plot(x, win_rates)
    plt.title('Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate %')
    
    # Plot 3: Training Loss and Average Scores if available
    plt.subplot(3, 1, 3)
    if losses is not None and len(losses) > 0:
        episodes = range(1, len(losses) + 1)
        plt.plot(episodes, losses, 'b-', alpha=0.3, label='Loss')
        
        # Calculate and plot EMA of losses
        if len(losses) > 10:
            ema_losses = calculate_ema(losses)
            plt.plot(episodes, ema_losses, 'b-', label='Loss (EMA)')
        
        plt.ylabel('Loss', color='b')
        plt.legend(loc='upper left')
    
    # Add average scores on the same subplot if available
    if avg_scores is not None and len(avg_scores) > 0:
        if losses is not None:
            plt.twinx()
        
        episodes = range(1, len(avg_scores) + 1)
        plt.plot(episodes, avg_scores, 'r-', alpha=0.3, label='Avg Score')
        
        # Calculate and plot EMA of average scores
        if len(avg_scores) > 10:
            ema_scores = calculate_ema(avg_scores)
            plt.plot(episodes, ema_scores, 'r-', label='Avg Score (EMA)')
        
        plt.ylabel('Average Score', color='r')
        plt.legend(loc='upper right')
    
    plt.title('Training Metrics')
    plt.xlabel('Episode')
    
    plt.tight_layout()
    if figure_file:
        plt.savefig(figure_file)
    
    plt.close(fig)

def main():
    """Main training loop."""
    args = parse_args()
    
    # Set up run directory
    if args.run_dir is None:
        args.run_dir = get_run_dir(algorithm_name='ppo')
    
    # Setup logging
    logger, logs_dir = setup_logging(args.log_level, args.run_dir)
    
    # Create necessary directories
    dirs = setup_run_directories(args.run_dir)
    models_dir = dirs.models_dir
    charts_dir = dirs.charts_dir
    
    logger.info(f"Starting training with args: {args}")
    logger.info(f"Saving models to {models_dir}")
    logger.info(f"Saving charts to {charts_dir}")
    
    # Initialize environment and agent
    env = GolfGame()
    
    agent = PPOAgent(
        state_size=28,
        action_size=9,
        hidden_size=args.hidden_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        policy_clip=args.policy_clip,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        update_interval=args.update_interval
    )
    
    # Load model if specified
    if args.load_model:
        agent.load(args.load_model)
        logger.info(f"Loaded model from {args.load_model}")
    
    # Training metrics
    best_eval_reward = float('-inf')
    best_win_rate = 0.0
    episode_rewards = []
    eval_rewards = []
    win_rates = []
    eval_episodes = []
    losses = []
    episode_losses = []
    
    # For tracking recent performance
    recent_rewards = deque(maxlen=100)
    
    # For timing
    start_time = time.time()
    
    # Main training loop
    for episode in tqdm(range(1, args.num_episodes + 1)):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Episode loop
        for i in range(1000):
            if done:
                break
            valid_actions = env._get_valid_actions()
            action, prob, val = agent.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, prob, val, reward, done, valid_actions)
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            
            # Learn if enough steps have been accumulated
            if agent.total_steps % agent.update_interval == 0:
                loss = agent.learn()
                if loss is not None:
                    episode_losses.append(loss)
        
        # Track rewards
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        
        # Record average loss for this episode if any
        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            losses.append(avg_loss)
            episode_losses = []
        elif len(losses) > 0:
            # If no learning happened this episode, repeat the last loss
            losses.append(losses[-1])
        else:
            # First episode with no learning
            losses.append(0.0)
        
        # Evaluate agent
        if episode % args.eval_interval == 0:
            eval_reward, win_rate = evaluate_agent(agent, env, args.eval_episodes)
            eval_rewards.append(eval_reward)
            win_rates.append(win_rate)
            eval_episodes.append(episode)
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                model_path = os.path.join(models_dir, f"{args.model_name}_best_reward.pt")
                agent.save(model_path)
                logger.info(f"New best model (reward: {best_eval_reward:.4f}) saved to {model_path}")
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                model_path = os.path.join(models_dir, f"{args.model_name}_best_winrate.pt")
                agent.save(model_path)
                logger.info(f"New best model (win rate: {best_win_rate:.4f}) saved to {model_path}")
        
        # Save model periodically
        if episode % args.save_interval == 0:
            model_path = os.path.join(models_dir, f"{args.model_name}_episode_{episode}.pt")
            agent.save(model_path)
            logger.info(f"Saved model checkpoint to {model_path}")
        
        # Plot learning curve
        if episode % args.plot_interval == 0 and eval_episodes:
            plot_file = os.path.join(charts_dir, f"{args.model_name}_learning_curve_ep{episode}.png")
            plot_learning_curve(
                eval_episodes, 
                eval_rewards, 
                win_rates,
                losses=losses,
                avg_scores=episode_rewards,
                figure_file=plot_file
            )
            logger.debug(f"Saved learning curve to {plot_file}")
        
        # Log progress
        if episode % args.log_interval == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            elapsed_time = time.time() - start_time
            logger.info(f"Episode: {episode}, Avg Reward (last 100): {avg_reward:.4f}, Steps: {agent.total_steps}, Time: {elapsed_time:.2f}s")
            if eval_episodes:
                logger.info(f"Last Eval - Reward: {eval_rewards[-1]:.4f}, Win Rate: {win_rates[-1]:.4f}")
    
    # Final save
    final_model_path = os.path.join(models_dir, f"{args.model_name}_final.pt")
    agent.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Final evaluation
    final_reward, final_win_rate = evaluate_agent(agent, env, args.eval_episodes * 2)
    logger.info(f"Final Evaluation - Reward: {final_reward:.4f}, Win Rate: {final_win_rate:.4f}")
    
    # Final plot
    if eval_episodes:
        final_plot_file = os.path.join(charts_dir, f"{args.model_name}_final_curve.png")
        plot_learning_curve(
            eval_episodes, 
            eval_rewards, 
            win_rates,
            losses=losses,
            avg_scores=episode_rewards,
            figure_file=final_plot_file
        )
        logger.info(f"Saved final learning curve to {final_plot_file}")
    
    return final_plot_file

if __name__ == "__main__":
    main() 