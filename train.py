import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import logging
from golf_game import GolfGame
from agent import DQNAgent
import random

# Setup logging
def setup_logging(log_level):
    """Set up logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("golf-ai")

# Logger will be properly configured in main

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN agent to play Golf')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size of the neural network')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.97, help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Starting epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.1, help='Minimum epsilon for exploration')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Decay rate for epsilon')
    parser.add_argument('--target-update', type=int, default=100, help='Target network update frequency')
    parser.add_argument('--eval-interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load model from')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    return parser.parse_args()

def evaluate_agent(agent, num_episodes=100, logger=None):
    """Evaluate the agent's performance against a random opponent."""
    if logger:
        logger.debug(f"Evaluating agent over {num_episodes} episodes")
    env = GolfGame(num_players=2)  # Two players: agent and opponent
    
    if logger:
        logger.debug(f"Using observation size: {agent.state_size}, action size: {agent.action_size}")
    
    total_rewards = []
    win_count = 0
    loss_count = 0
    max_turns_count = 0
    
    # Track actual golf scores (lower is better)
    agent_scores = []
    opponent_scores = []
    average_score_diff = 0  # How much better the agent is than opponents
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        agent_player = 0  # Agent is always player 0
        
        if logger and episode % 10 == 0:
            logger.debug(f"Starting evaluation episode {episode + 1}/{num_episodes}")

        while not done:
            # Determine whose turn it is
            current_player = env.current_player
            
            if current_player == agent_player:
                # Agent's turn
                valid_actions = env._get_valid_actions()
                action = agent.select_action(state, valid_actions, training=False)
            else:
                # Opponent's turn (random policy)
                valid_actions = env._get_valid_actions()
                action = random.choice(valid_actions)
            
            next_state, reward, done, info = env.step(action)
            
            # Only update state when it's the agent's turn
            if current_player == agent_player:
                episode_reward += reward
                state = next_state
            
            # Check game result when done
            if done:
                if "max_turns_reached" in info and info["max_turns_reached"]:
                    max_turns_count += 1
                    if logger:
                        logger.warning(f"Episode {episode + 1} reached max turns limit!")
                
                # Get final scores
                scores = info.get("scores", [0, 0])
                
                # Record actual golf scores
                agent_score = scores[0]
                opponent_score = scores[1]
                agent_scores.append(agent_score)
                opponent_scores.append(opponent_score)
                
                # Determine win/loss based on scores (agent is player 0)
                if agent_score < opponent_score:  # Agent won
                    win_count += 1
                    if logger:
                        logger.info(f"Episode {episode + 1} finished with a win! Score: {agent_score} vs {opponent_score}")
                elif agent_score > opponent_score:  # Agent lost
                    loss_count += 1
                    if logger:
                        logger.info(f"Episode {episode + 1} finished with a loss! Score: {agent_score} vs {opponent_score}")
                else:  # Tie
                    if logger:
                        logger.info(f"Episode {episode + 1} finished with a tie! Score: {agent_score} vs {opponent_score}")

        total_rewards.append(episode_reward)
        if logger and episode % 25 == 0:
            logger.debug(f"Evaluation episode {episode}/{num_episodes}, reward: {episode_reward:.2f}")
    
    avg_reward = sum(total_rewards) / num_episodes
    win_rate = win_count / num_episodes
    loss_rate = loss_count / num_episodes
    max_turns_rate = max_turns_count / num_episodes
    
    # Calculate average golf scores (lower is better)
    avg_agent_score = sum(agent_scores) / len(agent_scores) if agent_scores else 0
    avg_opponent_score = sum(opponent_scores) / len(opponent_scores) if opponent_scores else 0
    score_diff = avg_opponent_score - avg_agent_score  # Positive means agent is better
    
    if logger:
        logger.debug(f"Evaluation complete - Avg reward: {avg_reward:.2f}")
        logger.info(f"Evaluation results - Win rate: {win_rate:.2f}, Loss rate: {loss_rate:.2f}, Tie rate: {1-win_rate-loss_rate:.2f}")
        logger.info(f"Average Golf Scores - Agent: {avg_agent_score:.2f}, Opponent: {avg_opponent_score:.2f}, Diff: {score_diff:.2f}")
        logger.info(f"Max turns reached in {max_turns_rate:.2f} of games")
        
    return avg_reward, win_rate, loss_rate, max_turns_rate, avg_agent_score, score_diff

def train(args, logger):
    """Train the DQN agent."""
    logger.info(f"Starting training with args: {args}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"Saving models to {args.save_dir}")
    
    # Initialize environment and agent
    env = GolfGame(num_players=2)  # Two players for self-play: agent and opponent
    state_size = 60  # Size of the observation space (simplified - removed suits)
    action_size = 9   # Number of possible actions (removed 'knock' action)
    logger.info(f"Environment initialized with state_size={state_size}, action_size={action_size}")
    
    # Initialize agent with Prioritized Experience Replay
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=args.hidden_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        target_update=args.target_update,
        # PER parameters
        per_alpha=0.6,       # How much prioritization to use (0 = none, 1 = full)
        per_beta=0.4,        # Start value for importance sampling (0 = no correction, 1 = full)
        per_beta_increment=0.0001  # Beta increment per learning step
    )
    
    logger.info("Using Prioritized Experience Replay with win episode marking")
    
    # Load model if specified
    if args.load_model and os.path.exists(args.load_model):
        agent.load(args.load_model)
        logger.info(f"Loaded model from {args.load_model}")
    
    # Training loop
    rewards = []
    losses = []
    
    # Evaluation metrics
    eval_rewards = []
    eval_win_rates = [] 
    eval_loss_rates = []
    eval_max_turns_rates = []
    eval_avg_scores = []
    eval_score_diffs = []
    best_eval_reward = float('-inf')  # Best score difference (renamed for backward compatibility)
    
    # Additional metrics for PER
    win_count = 0
    episode_steps = []
    max_turns_reached_count = 0
    
    # Metrics for learning stability
    high_loss_count = 0
    consecutive_high_losses = 0
    learning_rate_adjustments = 0
    lr_history = [args.lr]  # Start with initial learning rate
    lr_history_episodes = [0]  # Episodes where LR changed
    
    logger.info("Starting training loop")
    for episode in tqdm(range(args.episodes)):
        if episode % 10 == 0:
            logger.debug(f"Starting episode {episode+1}/{args.episodes}")
        state = env.reset()
        done = False
        episode_reward = 0
        episode_losses = []
        steps = 0
        max_turns_reached = False
        
        # Self-play loop
        agent_player = 0  # Agent always starts as player 0
        
        while not done:
            current_player = env.current_player
            valid_actions = env._get_valid_actions()
            
            # Determine which player's turn it is
            if current_player == agent_player:
                # Main agent's turn
                action = agent.select_action(state, valid_actions)
                next_state, reward, done, info = env.step(action)
                
                # Only store experiences and learn for the main agent
                agent.remember(state, action, reward, next_state, done)
                
                # Learn from experiences
                loss = agent.learn()
                if loss is not None:
                    # Check for potential learning instability
                    if loss > 100:
                        high_loss_count += 1
                        consecutive_high_losses += 1
                        logger.warning(f"High loss detected: {loss:.2f} at step {steps}, episode {episode+1}")
                        
                        # Automatic learning rate adjustment
                        if consecutive_high_losses >= 5:
                            # Reduce learning rate
                            for param_group in agent.optimizer.param_groups:
                                param_group['lr'] *= 0.5
                                new_lr = param_group['lr']
                            
                            # Record learning rate change
                            learning_rate_adjustments += 1
                            lr_history.append(new_lr)
                            lr_history_episodes.append(episode)
                            
                            logger.warning(f"Adjusted learning rate to {new_lr:.6f} due to instability")
                            consecutive_high_losses = 0
                    else:
                        consecutive_high_losses = 0  # Reset counter when loss is normal
                        
                    episode_losses.append(loss)
                    
                # Update state for the main agent
                state = next_state
                episode_reward += reward
            else:
                # Opponent's turn - use the same agent but with higher exploration
                # This introduces diversity while still using learned knowledge
                opponent_action = agent.select_action(state, valid_actions, training=True)
                _, _, done, info = env.step(opponent_action)
                # No learning from opponent's experiences
            
            steps += 1
            
            # Check if max turns was reached
            if done and "max_turns_reached" in info and info["max_turns_reached"]:
                max_turns_reached = True
        
        # Record metrics
        rewards.append(episode_reward)
        episode_steps.append(steps)
        
        # Track wins and max turns
        if episode_reward > 0:
            win_count += 1
        if max_turns_reached:
            max_turns_reached_count += 1
            
        # Record loss
        avg_loss = 0.0
        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            losses.append(avg_loss)
            
        # Ensure epsilon decays even if learning doesn't happen often enough
        # Use a more gradual decay formula to stretch it over more episodes
        # This will decay from 1.0 to 0.1 over approximately 5000 episodes
        # Formula: epsilon = epsilon_end + (epsilon_start - epsilon_end) * exp(-decay_rate * episode)
        decay_rate = 0.0005  # Adjusted for ~5000 episodes decay
        episode_fraction = (episode + 1) / args.episodes
        agent.epsilon = max(
            agent.epsilon_end,
            agent.epsilon_end + (args.epsilon_start - agent.epsilon_end) * np.exp(-decay_rate * (episode + 1))
        )
        
        # Log progress periodically
        if (episode + 1) % 100 == 0:
            win_rate = win_count / (episode + 1)
            max_turns_rate = max_turns_reached_count / (episode + 1)
            avg_steps = sum(episode_steps) / len(episode_steps)
            
            loss_info = f", Avg Loss={avg_loss:.4f}" if episode_losses else ""
            logger.info(f"Episode {episode+1}: Reward={episode_reward:.2f}{loss_info}")
            
            logger.info(f"Epsilon={agent.epsilon:.4f} (Current)")
            # Show learning stability metrics
            stability_info = ""
            if high_loss_count > 0:
                stability_info = f", LR Adjustments={learning_rate_adjustments}"
                current_lr = agent.optimizer.param_groups[0]['lr']
                if current_lr != args.lr:
                    stability_info += f", Current LR={current_lr:.6f}"
            
            logger.info(f"Stats: Win Rate={win_rate:.2f}, Avg Steps={avg_steps:.1f}, Max Turns Rate={max_turns_rate:.2f}")
            logger.info(f"Stability: High Losses={high_loss_count}{stability_info}")
        
        # Evaluate agent periodically
        if (episode + 1) % args.eval_interval == 0:
            eval_results = evaluate_agent(agent, logger=logger)
            avg_reward, win_rate, loss_rate, max_turns_rate, avg_score, score_diff = eval_results
            
            # Store metrics for plotting
            eval_rewards.append(avg_reward)
            eval_win_rates.append(win_rate)
            eval_loss_rates.append(loss_rate)
            eval_max_turns_rates.append(max_turns_rate)
            eval_avg_scores.append(avg_score)
            eval_score_diffs.append(score_diff)
            
            logger.info(f"Episode {episode+1}/{args.episodes}, Epsilon: {agent.epsilon:.4f}")
            
            # NEW: Save model based on score difference (primary) or win rate (secondary)
            # A better model is one that achieves lower golf scores compared to random opponent
            is_better = False
            reason = ""
            
            # First, check if this is the first evaluation
            if best_eval_reward == float('-inf'):
                is_better = True
                reason = "first evaluation"
            # Next, prioritize significant score improvements
            elif score_diff > best_eval_reward + 1.0:  # At least 1 point better
                is_better = True
                reason = f"score diff improved from {best_eval_reward:.2f} to {score_diff:.2f}"
            # For smaller improvements, also consider win rate as tiebreaker
            elif score_diff > best_eval_reward and win_rate >= 0.5:
                is_better = True
                reason = f"score diff improved slightly to {score_diff:.2f} with good win rate {win_rate:.2f}"
            
            if is_better:
                best_eval_reward = score_diff
                agent.save(os.path.join(args.save_dir, 'best_model.pth'))
                logger.info(f"Saved best model with score diff {best_eval_reward:.2f} ({reason})")
            
            # Save checkpoint
            agent.save(os.path.join(args.save_dir, f'checkpoint_{episode+1}.pth'))
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    agent.save(final_model_path)
    logger.info(f"Training complete. Final model saved to {final_model_path}")
    
    # Plot training curves
    plt.figure(figsize=(15, 15))  # Increased height for more plots
    
    plt.subplot(3, 3, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(3, 3, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.subplot(3, 3, 3)
    plt.plot(range(0, len(eval_rewards) * args.eval_interval, args.eval_interval), eval_rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.subplot(3, 3, 4)
    plt.plot(range(0, len(eval_win_rates) * args.eval_interval, args.eval_interval), eval_win_rates)
    plt.title('Win Rate vs Random Opponent')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    
    plt.subplot(3, 3, 5)
    plt.plot(range(0, len(eval_loss_rates) * args.eval_interval, args.eval_interval), eval_loss_rates)
    plt.title('Loss Rate vs Random Opponent')
    plt.xlabel('Episode')
    plt.ylabel('Loss Rate')
    
    plt.subplot(3, 3, 6)
    plt.plot(range(0, len(eval_max_turns_rates) * args.eval_interval, args.eval_interval), eval_max_turns_rates)
    plt.title('Max Turns Rate')
    plt.xlabel('Episode')
    plt.ylabel('Rate of Games Reaching Max Turns')
    
    # New plots for Golf-specific metrics
    plt.subplot(3, 3, 7)
    eval_episodes = range(0, len(eval_avg_scores) * args.eval_interval, args.eval_interval)
    plt.plot(eval_episodes, eval_avg_scores, label='Agent')
    # Add a baseline of a random agent (based on opponent's average score)
    random_scores = []
    for score_diff, agent_score in zip(eval_score_diffs, eval_avg_scores):
        random_scores.append(agent_score + score_diff)  # Reconstruct opponent score
    plt.plot(eval_episodes, random_scores, label='Random')
    plt.title('Average Golf Score (Lower is Better)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    
    plt.subplot(3, 3, 8)
    plt.plot(eval_episodes, eval_score_diffs)
    plt.title('Score Difference (Agent vs Random)')
    plt.xlabel('Episode')
    plt.ylabel('Score Diff (Positive = Better)')
    
    plt.subplot(3, 3, 9)
    # If we have learning rate adjustments, plot agent's current learning rate
    if 'lr_history' in locals():
        plt.plot(lr_history_episodes, lr_history)
        plt.title('Learning Rate Adjustments')
        plt.xlabel('Episode')
        plt.ylabel('Learning Rate')
    else:
        # Alternative plot if no LR history: show high loss events
        if high_loss_count > 0:
            plt.bar(['High Losses'], [high_loss_count])
            plt.title('Learning Stability')
            plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'))
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logging(args.log_level)
    try:
        logger.info("Starting Golf Card Game AI training")
        train(args, logger)
    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")
        raise 