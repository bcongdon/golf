import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import logging
from golf_game_v2 import GolfGame, GameConfig, Action, Card, GameInfo
from agent import DQNAgent
import random
from collections import deque
import datetime
from reflex_agent import ReflexAgent
from training_utils import get_run_dir, setup_logging, setup_run_directories
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN agent to play Golf')
    
    # Get default run directory
    default_run_dir = get_run_dir(algorithm_name='dqn')
    
    parser.add_argument('--run-dir', type=str, default=default_run_dir,
                        help='Directory for this training run (default: auto-generated with timestamp)')
    parser.add_argument('--episodes', type=int, default=100000, help='Number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size of the neural network')
    parser.add_argument('--embedding-dim', type=int, default=8, help='Dimension of card embeddings')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.975, help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Starting epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.05, help='Minimum epsilon for exploration')
    parser.add_argument('--epsilon-decay-episodes', type=int, default=90000, help='Number of episodes to decay epsilon from start to end')
    parser.add_argument('--epsilon-warmup', type=int, default=10000, help='Number of episodes to keep epsilon at start value')
    parser.add_argument('--target-update', type=int, default=500, help='Target network update frequency')
    parser.add_argument('--eval-interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load model from')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--num-workers', type=int, default=0, 
                        help='Number of worker processes for environment (0=auto)')
    
    # New optimization arguments
    parser.add_argument('--mixed-precision', action='store_true', 
                        help='Enable mixed precision training for faster computation')
    parser.add_argument('--pin-memory', action='store_true', 
                        help='Use pinned memory for faster CPU-GPU transfers')
    parser.add_argument('--learn-every', type=int, default=4, 
                        help='Learn every N steps to reduce overhead')
    parser.add_argument('--use-huber-loss', action='store_true', 
                        help='Use Huber loss instead of MSE for better stability')
    parser.add_argument('--use-per', action='store_true',
                        help='Use Prioritized Experience Replay (PER)')
    parser.add_argument('--per-alpha', type=float, default=0.7,
                        help='Alpha parameter for PER (0 = uniform, 1 = full prioritization)')
    parser.add_argument('--per-beta', type=float, default=0.5,
                        help='Initial beta parameter for PER importance sampling')
    parser.add_argument('--per-beta-increment', type=float, default=0.0001,
                        help='Increment for beta parameter over time')
    parser.add_argument('--segment-tree', action='store_true', 
                        help='Use segment tree for more efficient sampling in replay buffer')
    parser.add_argument('--optimize-memory', action='store_true', 
                        help='Optimize memory usage with float16 for states')
    
    return parser.parse_args()

def evaluate_agent(agent, num_episodes=100, logger=None):
    """Evaluate the agent's performance against a random opponent and reflex agent."""
    if logger:
        logger.debug(f"Evaluating agent over {num_episodes} episodes")
    config = GameConfig(num_players=2)
    env = GolfGame(config)
    
    # Create reflex agent
    reflex_agent_1 = ReflexAgent(player_id=1)
    
    # Metrics for random opponent
    random_rewards = []
    random_wins = 0
    random_losses = 0
    random_ties = 0
    random_scores = []  # All scores
    random_scores_no_ties = []  # Scores excluding ties
    random_opponent_scores = []  # All opponent scores
    random_opponent_scores_no_ties = []  # Opponent scores excluding ties
    
    # Metrics for reflex opponent
    reflex_rewards = []
    reflex_wins = 0
    reflex_losses = 0
    reflex_ties = 0
    reflex_scores = []  # All scores
    reflex_scores_no_ties = []  # Scores excluding ties
    reflex_opponent_scores = []  # All opponent scores
    reflex_opponent_scores_no_ties = []  # Opponent scores excluding ties
    
    # Track Q-values during evaluation
    q_values_list = []
    
    # Set agent to evaluation mode
    agent.q_network.eval()
    
    # Evaluate against both random and reflex opponents
    for opponent_type in ['random', 'reflex']:
        for episode in range(num_episodes // 2):  # Split episodes between opponents
            state = env.reset()
            done = False
            episode_reward = 0
            agent_player = 0  # Agent is always player 0
            
            while not done:
                current_player = env.current_player
                valid_actions = env._get_valid_actions()
                
                if current_player == agent_player:
                    # Agent's turn
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                        q_vals = agent.q_network(state_tensor).cpu().numpy()[0]
                        valid_q = [q_vals[action] for action in valid_actions]
                        q_values_list.extend(valid_q)
                        action = agent.select_action(state, valid_actions, training=False)
                else:
                    # Opponent's turn
                    if opponent_type == 'reflex':
                        action = reflex_agent_1.select_action(state, valid_actions)
                    else:  # random
                        action = agent.random_action(valid_actions)
                
                next_state, reward, done, info = env.step(action)
                
                if current_player == agent_player:
                    episode_reward += reward
                
                state = next_state
            
            # Process episode results
            if isinstance(info, dict) and "scores" in info:
                scores = info["scores"]
            elif hasattr(info, "scores") and info.scores:
                scores = info.scores
            else:
                scores = [0, 0]
            
            agent_score = scores[agent_player]
            opponent_idx = 1 if agent_player == 0 else 0
            opponent_score = scores[opponent_idx]
            
            if opponent_type == 'random':
                random_rewards.append(episode_reward)
                random_scores.append(agent_score)
                random_opponent_scores.append(opponent_score)
                
                if agent_score < opponent_score:
                    random_wins += 1
                    random_scores_no_ties.append(agent_score)
                    random_opponent_scores_no_ties.append(opponent_score)
                elif agent_score > opponent_score:
                    random_losses += 1
                    random_scores_no_ties.append(agent_score)
                    random_opponent_scores_no_ties.append(opponent_score)
                else:
                    random_ties += 1
            else:  # reflex
                reflex_rewards.append(episode_reward)
                reflex_scores.append(agent_score)
                reflex_opponent_scores.append(opponent_score)
                
                if agent_score < opponent_score:
                    reflex_wins += 1
                    reflex_scores_no_ties.append(agent_score)
                    reflex_opponent_scores_no_ties.append(opponent_score)
                elif agent_score > opponent_score:
                    reflex_losses += 1
                    reflex_scores_no_ties.append(agent_score)
                    reflex_opponent_scores_no_ties.append(opponent_score)
                else:
                    reflex_ties += 1
            
            # Check if max turns was reached
            if done:
                # Handle both dict and GameInfo objects
                if isinstance(info, dict) and "max_turns_reached" in info and info["max_turns_reached"]:
                    max_turns_reached = True
                elif hasattr(info, "max_turns_reached") and info.max_turns_reached:
                    max_turns_reached = True
        
        # Record metrics
        rewards.append(episode_reward)
        episode_steps.append(steps)
        
        # Track wins and max turns
        if isinstance(info, dict) and "scores" in info and len(info["scores"]) > 1 and info["scores"][0] > info["scores"][1]:
            win_count += 1
        elif hasattr(info, "scores") and info.scores and len(info.scores) > 1 and info.scores[0] > info.scores[1]:
            win_count += 1
        if max_turns_reached:
            max_turns_reached_count += 1
            
    # Calculate metrics
    # Use non-tie scores for average score calculations
    random_avg_score = np.mean(random_scores_no_ties) if random_scores_no_ties else 0
    random_avg_opponent_score = np.mean(random_opponent_scores_no_ties) if random_opponent_scores_no_ties else 0
    
    reflex_avg_score = np.mean(reflex_scores_no_ties) if reflex_scores_no_ties else 0
    reflex_avg_opponent_score = np.mean(reflex_opponent_scores_no_ties) if reflex_opponent_scores_no_ties else 0
    
    random_metrics = {
        'avg_reward': np.mean(random_rewards),
        'win_rate': random_wins / (num_episodes / 2),
        'loss_rate': random_losses / (num_episodes / 2),
        'tie_rate': random_ties / (num_episodes / 2),
        'avg_score': random_avg_score,
        'score_diff': random_avg_opponent_score - random_avg_score
    }
    
    reflex_metrics = {
        'avg_reward': np.mean(reflex_rewards),
        'win_rate': reflex_wins / (num_episodes / 2),
        'loss_rate': reflex_losses / (num_episodes / 2),
        'tie_rate': reflex_ties / (num_episodes / 2),
        'avg_score': reflex_avg_score,
        'score_diff': reflex_avg_opponent_score - reflex_avg_score
    }
    
    if logger:
        logger.info(f"vs Random - Win Rate: {random_metrics['win_rate']:.2f}, "
                   f"Tie Rate: {random_metrics['tie_rate']:.2f}, "
                   f"Avg Score (excl. ties): {random_metrics['avg_score']:.2f}, "
                   f"Score Diff: {random_metrics['score_diff']:.2f}")
        logger.info(f"vs Reflex - Win Rate: {reflex_metrics['win_rate']:.2f}, "
                   f"Tie Rate: {reflex_metrics['tie_rate']:.2f}, "
                   f"Avg Score (excl. ties): {reflex_metrics['avg_score']:.2f}, "
                   f"Score Diff: {reflex_metrics['score_diff']:.2f}")
    
    # Calculate Q-value statistics
    q_stats = {}
    if q_values_list:
        q_stats = {
            'mean': np.mean(q_values_list),
            'min': np.min(q_values_list),
            'max': np.max(q_values_list)
        }
    
    # Return metrics for tracking
    return (
        random_metrics['avg_reward'],
        random_metrics['win_rate'],
        random_metrics['loss_rate'],
        random_metrics['score_diff'],
        reflex_metrics['avg_reward'],
        reflex_metrics['win_rate'],
        reflex_metrics['loss_rate'],
        reflex_metrics['score_diff'],
        q_stats
    )

def train(args, logger, logs_dir):
    """Train the DQN agent."""
    logger.info(f"Starting training with args: {args}")
    
    # Get all run directories
    run_dir = os.path.dirname(logs_dir)
    dirs = setup_run_directories(run_dir)
    models_dir = dirs.models_dir
    charts_dir = dirs.charts_dir
    
    logger.info(f"Saving models to {models_dir}")
    logger.info(f"Saving charts to {charts_dir}")
    
    # Initialize metrics lists
    rewards = []
    losses = []
    epsilon_history = []
    q_value_avg = []
    q_value_min = []
    q_value_max = []
    eval_rewards = []
    eval_win_rates = []
    eval_loss_rates = []
    eval_max_turns_rates = []
    eval_avg_scores = []
    eval_score_diffs = []
    # Add reflex agent metrics
    reflex_rewards = []
    reflex_win_rates = []
    reflex_score_diffs = []
    best_eval_reward = float('-inf')
    
    # Initialize environment with GameConfig
    config = GameConfig(
        num_players=2,
        grid_rows=2,
        grid_cols=3,
        initial_revealed=2,
        max_turns=100,
        normalize_rewards=True,
        copies_per_rank=4
    )
    env = GolfGame(config)  # Two players for self-play: agent and opponent
    
    # Calculate state size based on game configuration
    cards_per_player = config.grid_rows * config.grid_cols
    state_size = (2 * cards_per_player  # Player hands
                 + 2  # Discard and drawn card
                 + 2 * cards_per_player  # Revealed flags
                 + 3)  # Drawn from discard, final round flags, and turn progress
    action_size = len(Action)  # Number of possible actions
    logger.info(f"Environment initialized with state_size={state_size}, action_size={action_size}")
    
    # Initialize two reflex agents, one for each player position
    reflex_agent_0 = ReflexAgent(player_id=0)  # For when reflex agent plays as player 0
    reflex_agent_1 = ReflexAgent(player_id=1)  # For when reflex agent plays as player 1
    logger.info("Initialized ReflexAgents for both player positions")
    
    # Initialize agent with Prioritized Experience Replay
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        num_card_ranks=13,  # Number of possible card ranks (A-K)
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        epsilon_warmup_episodes=args.epsilon_warmup,
        batch_size=args.batch_size,
        target_update=args.target_update,
        # PER parameters
        use_per=args.use_per,  # Whether to use prioritized experience replay
        per_alpha=args.per_alpha,  # How much prioritization to use (0 = none, 1 = full)
        per_beta=args.per_beta,  # Start value for importance sampling (0 = no correction, 1 = full)
        per_beta_increment=args.per_beta_increment  # Beta increment per learning step
    )
    
    # Apply optimization settings
    if args.mixed_precision and agent.device.type == 'cuda':
        agent.use_amp = True
        agent.scaler = torch.cuda.amp.GradScaler()
        logger.info("Enabled mixed precision training for faster computation")
    
    # Log replay buffer information
    if args.use_per:
        logger.info(f"Using Prioritized Experience Replay (PER) with alpha={args.per_alpha}, beta={args.per_beta}")
    else:
        logger.info("Using standard (uniform) experience replay")
    
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
            # This is handled in the agent's learn method
    logger.info(device_info)
    
    logger.info("Using Prioritized Experience Replay with win episode marking")
    
    # Load model if specified
    if args.load_model and os.path.exists(args.load_model):
        agent.load(args.load_model)
        logger.info(f"Loaded model from {args.load_model}")
    
    
    # Random opponent probability (will decrease over time)
    random_opponent_prob = 0.7  # Start with high probability of random opponent
    random_opponent_decay = 0.9999  # Decay factor
    
    # Metrics tracking
    episode_steps = []
    q_value_avg = []
    q_value_min = []
    q_value_max = []
    
    # Add tracking for reflex agent performance
    reflex_agent_games = 0
    reflex_agent_wins = 0
    reflex_agent_losses = 0
    reflex_agent_score_diff = 0
    
    # Win count tracking
    win_count = 0
    max_turns_reached_count = 0
    
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
    def save_charts(episode):
        plt.figure(figsize=(15, 20))  # Increased height for more plots
        
        # Row 1: Basic training metrics
        plt.subplot(4, 3, 1)
        plt.plot(rewards, alpha=0.5, label='Episode Rewards')
        
        # Calculate and plot EMA of rewards
        if len(rewards) > 0:
            rewards_ema = list(calculate_ema(rewards, alpha=0.1))
            plt.plot(rewards_ema, color='red', linewidth=2, label='EMA (α=0.1)')
            
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        # Plot EMA of training loss (replacing the regular training loss plot)
        plt.subplot(4, 3, 2)
        if len(losses) > 0:
            plt.plot(losses, alpha=0.3, color='lightblue', label='Raw Loss')
            loss_ema = list(calculate_ema(losses, alpha=0.1))
            plt.plot(loss_ema, color='blue', linewidth=2, label='EMA (α=0.1)')
            plt.title('Training Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.legend()
        
        # Add epsilon plot
        plt.subplot(4, 3, 3)
        plt.plot(epsilon_history)
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        # Row 2: Evaluation metrics
        if eval_rewards:
            plt.subplot(4, 3, 4)
            plt.plot(range(0, len(eval_rewards) * args.eval_interval, args.eval_interval), eval_rewards)
            plt.title('Evaluation Rewards vs Random')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            
            plt.subplot(4, 3, 5)
            plt.plot(range(0, len(eval_win_rates) * args.eval_interval, args.eval_interval), eval_win_rates)
            plt.title('Win Rate vs Random Opponent')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate')
            
            # Add win rate vs reflex agent (replacing loss rate vs random)
            if reflex_win_rates and len(reflex_win_rates) > 0:
                plt.subplot(4, 3, 6)
                plt.plot(range(0, len(reflex_win_rates) * args.eval_interval, args.eval_interval), reflex_win_rates)
                plt.title('Win Rate vs Reflex Opponent')
                plt.xlabel('Episode')
                plt.ylabel('Win Rate')
        
        # Row 3: More evaluation metrics
        if eval_rewards:
            # Add evaluation rewards for reflex agent (replacing max turns rate)
            if reflex_rewards and len(reflex_rewards) > 0:
                plt.subplot(4, 3, 7)
                plt.plot(range(0, len(reflex_rewards) * args.eval_interval, args.eval_interval), reflex_rewards)
                plt.title('Evaluation Rewards vs Reflex')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
            
            # Golf-specific metrics
            plt.subplot(4, 3, 8)
            if eval_avg_scores and len(eval_avg_scores) > 0:
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
            
            plt.subplot(4, 3, 9)
            if eval_score_diffs and len(eval_score_diffs) > 0:
                eval_episodes = range(0, len(eval_score_diffs) * args.eval_interval, args.eval_interval)
                plt.plot(eval_episodes, eval_score_diffs)
            plt.title('Score Difference (Agent vs Random)')
            plt.xlabel('Episode')
            plt.ylabel('Score Diff (Positive = Better)')
        
        # Row 4: Q-value metrics and learning rate
        if q_value_avg:
            plt.subplot(4, 3, 10)
            plt.plot(q_value_avg, label='Average')
            plt.plot(q_value_min, label='Min')
            plt.plot(q_value_max, label='Max')
            plt.title('Q-Value Statistics')
            plt.xlabel('Episode')
            plt.ylabel('Q-Value')
            plt.legend()
        
        # Add reflex score difference if available
        if reflex_score_diffs and len(reflex_score_diffs) > 0:
            plt.subplot(4, 3, 11)
            eval_episodes = range(0, len(reflex_score_diffs) * args.eval_interval, args.eval_interval)
            plt.plot(eval_episodes, reflex_score_diffs)
            plt.title('Score Difference (Agent vs Reflex)')
            plt.xlabel('Episode')
            plt.ylabel('Score Diff (Positive = Better)')
        
        plt.tight_layout()
        
        # Save to both model directory and logs directory
        plt.savefig(os.path.join(models_dir, 'training_curves.png'))
        plt.savefig(os.path.join(logs_dir, f'training_curves_ep{episode+1}.png'))
        plt.close()  # Close the figure to free memory
        
        # Save metrics as CSV for later analysis
        if eval_rewards:
            eval_data = {
                'Episode': range(0, len(eval_rewards) * args.eval_interval, args.eval_interval),
                'Reward_vs_Random': eval_rewards,
                'WinRate_vs_Random': eval_win_rates,
                'AvgScore': eval_avg_scores,
                'ScoreDiff_vs_Random': eval_score_diffs
            }
            
            # Add reflex metrics if available
            if reflex_rewards and len(reflex_rewards) > 0:
                eval_data['Reward_vs_Reflex'] = reflex_rewards
            if reflex_win_rates and len(reflex_win_rates) > 0:
                eval_data['WinRate_vs_Reflex'] = reflex_win_rates
            if reflex_score_diffs and len(reflex_score_diffs) > 0:
                eval_data['ScoreDiff_vs_Reflex'] = reflex_score_diffs
                
            pd.DataFrame(eval_data).to_csv(os.path.join(logs_dir, 'eval_metrics.csv'), index=False)
        
        # Save training metrics
        train_data = {
            'Episode': range(len(rewards)),
            'Reward': rewards,
            'Epsilon': epsilon_history if len(epsilon_history) == len(rewards) else epsilon_history + [None] * (len(rewards) - len(epsilon_history)),
            'Q_Avg': q_value_avg + [None] * (len(rewards) - len(q_value_avg)),
            'Q_Min': q_value_min + [None] * (len(rewards) - len(q_value_min)),
            'Q_Max': q_value_max + [None] * (len(rewards) - len(q_value_max))
        }
        if losses:
            train_data['Loss_EMA'] = list(calculate_ema(losses, alpha=0.1)) + [None] * (len(rewards) - len(losses))
        pd.DataFrame(train_data).to_csv(os.path.join(logs_dir, 'training_metrics.csv'), index=False)
    
    # Determine if we should use multi-processing for environment steps
    # This can help reduce CPU bottlenecks when GPU is fast
    use_multiprocessing = False
    if agent.device.type == 'cuda':
        # Only use multiprocessing if we have a GPU and multiple CPU cores
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        # Use num_workers if specified, otherwise auto-detect
        if args.num_workers > 0:
            use_multiprocessing = True
            num_workers = args.num_workers
            logger.info(f"Using {num_workers} worker processes as specified")
        elif cpu_count >= 4:  # At least 4 cores for multiprocessing to be beneficial
            use_multiprocessing = True
            num_workers = min(4, cpu_count - 1)  # Leave one core for main process
            logger.info(f"Enabling environment multiprocessing with {num_workers} worker processes")
        
        if use_multiprocessing:
            # Set up multiprocessing environment
            try:
                import torch.multiprocessing as mp
                mp.set_start_method('spawn', force=True)
                logger.info("Using 'spawn' multiprocessing start method for better CUDA compatibility")
            except RuntimeError:
                logger.warning("Could not set multiprocessing start method to 'spawn'")
                
            # Set PyTorch threads for better performance with multiprocessing
            torch.set_num_threads(1)
            logger.info("Set PyTorch threads to 1 per process for better multiprocessing performance")
    
    # Function to select opponent action (to avoid code duplication)
    def select_opponent_action(state, valid_actions):
        # Determine opponent type based on training phase (probabilistic approach)
        # As epsilon decreases, we transition from random → reflex → self-play
        random_prob = agent.epsilon  # Highest when epsilon is high (early training)
        reflex_prob = 4 * agent.epsilon * (1 - agent.epsilon)  # Peaks in middle of training
        selfplay_prob = (1 - agent.epsilon) ** 2  # Highest when epsilon is low (late training)
        
        # Normalize probabilities to sum to 1
        total_prob = random_prob + reflex_prob + selfplay_prob
        random_prob /= total_prob
        reflex_prob /= total_prob
        selfplay_prob /= total_prob
        
        # Choose opponent type based on probabilities
        choice = random.random()
        
        if choice < random_prob:
            # Random opponent
            return agent.random_action(valid_actions)
        elif choice < random_prob + reflex_prob:
            # Reflex agent opponent
            reflex_agent = reflex_agent_0 if env.current_player == 0 else reflex_agent_1
            return reflex_agent.select_action(state, valid_actions)
        else:
            # Self-play (with some exploration)
            original_epsilon = agent.epsilon
            agent.epsilon = max(0.1, agent.epsilon)  # At least 10% exploration
            
            # Flip state perspective for player 1, since the agent was trained as player 0
            if env.current_player == 1:
                flipped_state = env.flip_state_perspective(state)
                action = agent.select_action(flipped_state, valid_actions, training=True)
            else:
                action = agent.select_action(state, valid_actions, training=True)
                
            agent.epsilon = original_epsilon  # Restore original epsilon
            return action
    
    # Pre-allocate tensors for batched learning
    if agent.device.type == 'cuda':
        # Pre-compile the network for faster execution
        logger.info("Pre-compiling neural networks for faster execution...")
        dummy_state = torch.zeros((1, state_size), device=agent.device)
        with torch.no_grad():
            _ = agent.q_network(dummy_state)
            _ = agent.target_network(dummy_state)
        logger.info("Neural networks pre-compiled")
    
    # Adjust learning frequency based on device
    if agent.device.type == 'cuda':
        # On GPU, we can afford to learn less frequently but with larger batches
        learn_every = args.learn_every  # Use command line argument
        # Increase effective batch size for better GPU utilization
        effective_batch_size = args.batch_size * 2
        logger.info(f"GPU detected: Learning every {learn_every} steps with effective batch size {effective_batch_size}")
    elif agent.device.type == 'mps':
        # On MPS, use a moderate learning frequency
        learn_every = args.learn_every
        effective_batch_size = args.batch_size
        logger.info(f"MPS detected: Learning every {learn_every} steps with batch size {effective_batch_size}")
    else:
        # On CPU, learn more frequently with smaller batches
        learn_every = args.learn_every
        effective_batch_size = args.batch_size
        logger.info(f"CPU mode: Learning every {learn_every} steps with batch size {effective_batch_size}")
    
    # Pre-allocate memory for experiences
    experiences_batch = []
    experiences_capacity = learn_every * 2  # Double capacity to avoid frequent resizing
    
    # Function to sample Q-values for monitoring
    def sample_q_values(agent, env, num_samples=10):
        """Sample Q-values from random hands to monitor network behavior."""
        q_values = []
        
        # Create a temporary environment for sampling
        config = GameConfig(num_players=2)
        temp_env = GolfGame(config)
        
        for _ in range(num_samples):
            # Reset to get a fresh environment
            state = temp_env.reset()
            
            # Randomly reveal some cards to create diverse states
            # This simulates different stages of the game
            num_revealed = random.randint(1, 5)  # Reveal between 1 and 5 cards
            
            # Randomly reveal cards for the current player
            revealed_indices = random.sample(range(6), num_revealed)
            for idx in revealed_indices:
                temp_env.revealed_cards[temp_env.current_player].add(idx)
            
            # Randomly reveal some opponent cards
            opponent = (temp_env.current_player + 1) % temp_env.config.num_players
            opponent_revealed = random.randint(0, 4)  # Reveal between 0 and 4 opponent cards
            if opponent_revealed > 0:
                opponent_indices = random.sample(range(6), opponent_revealed)
                for idx in opponent_indices:
                    temp_env.revealed_cards[opponent].add(idx)
            
            # Randomly set drawn card (50% chance)
            if random.random() < 0.5:
                if temp_env.deck:
                    temp_env.drawn_card = temp_env.deck.pop()
                    temp_env.drawn_from_discard = False
            
            # Randomly set discard pile (75% chance)
            if random.random() < 0.75 and temp_env.deck:
                temp_env.discard_pile = [temp_env.deck.pop()]
            
            # Get the observation after modifications
            state = temp_env._get_observation()
            
            # Get valid actions for this state
            valid_actions = temp_env._get_valid_actions()
            
            if not valid_actions:  # Skip if no valid actions
                continue
                
            # Get Q-values for all actions
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_vals = agent.q_network(state_tensor).cpu().numpy()[0]
            
            # Only consider valid actions
            valid_q = [q_vals[action] for action in valid_actions]
            
            if valid_q:  # Only append if we have valid actions
                q_values.extend(valid_q)
        
        if not q_values:  # Safety check
            return 0, 0, 0
            
        return np.mean(q_values), np.min(q_values), np.max(q_values)
    
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
        
        # Clear experiences batch at the start of each episode
        experiences_batch = []
        
        while not done:
            current_player = env.current_player
            valid_actions = env._get_valid_actions()
            
            # Determine which player's turn it is
            if current_player == agent_player:
                # Main agent's turn
                action = agent.select_action(state, valid_actions)
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                experiences_batch.append((state, action, reward, next_state, done))
                
                # Only learn periodically to reduce overhead
                if len(experiences_batch) >= learn_every or done:
                    # Add all experiences to replay buffer in a batch
                    for exp in experiences_batch:
                        s, a, r, ns, d = exp
                        agent.remember(s, a, r, ns, d)
                    
                    # Learn if we have enough data
                    if len(agent.memory) >= effective_batch_size:
                        loss = agent.learn()
                        
                        if loss is not None:         
                            episode_losses.append(loss)
                    
                    # Clear batch after learning
                    experiences_batch = []
                    
                # Update state for the main agent
                state = next_state
                episode_reward += reward
            else:
                # Opponent's turn
                opponent_action = select_opponent_action(state, valid_actions)
                
                # Take the action
                next_state, _, done, info = env.step(opponent_action)
                # No learning from opponent's experiences
                
                # Update state if needed
                if not done:
                    state = next_state
            
            steps += 1
            
            # Check if max turns was reached
            if done:
                # Handle both dict and GameInfo objects
                if isinstance(info, dict) and "max_turns_reached" in info and info["max_turns_reached"]:
                    max_turns_reached = True
                elif hasattr(info, "max_turns_reached") and info.max_turns_reached:
                    max_turns_reached = True
        
        # Record metrics
        rewards.append(episode_reward)
        episode_steps.append(steps)
        
        # Track wins and max turns
        if isinstance(info, dict) and "scores" in info and len(info["scores"]) > 1 and info["scores"][0] > info["scores"][1]:
            win_count += 1
        elif hasattr(info, "scores") and info.scores and len(info.scores) > 1 and info.scores[0] > info.scores[1]:
            win_count += 1
        if max_turns_reached:
            max_turns_reached_count += 1
            
        # Record loss
        avg_loss = 0.0
        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            losses.append(avg_loss)
            
        # Update epsilon using piecewise linear decay strategy
        agent.update_epsilon()
            
        # Record epsilon value
        epsilon_history.append(agent.epsilon)
        
        # Sample Q-values periodically (every 100 episodes)
        if episode % 100 == 0:
            # Create a temporary environment for sampling to avoid affecting training
            temp_env = GolfGame()
            avg_q, min_q, max_q = sample_q_values(agent, temp_env)
            q_value_avg.append(avg_q)
            q_value_min.append(min_q)
            q_value_max.append(max_q)
            logger.debug(f"Q-value stats - Avg: {avg_q:.4f}, Min: {min_q:.4f}, Max: {max_q:.4f}")
        
        # Log progress periodically
        if (episode + 1) % 100 == 0:
            win_rate = win_count / (episode + 1)
            max_turns_rate = max_turns_reached_count / (episode + 1)
            avg_steps = sum(episode_steps) / len(episode_steps)
            
            loss_info = f", Avg Loss={avg_loss:.4f}" if episode_losses else ""
            logger.info(f"Episode {episode+1}: Reward={episode_reward:.2f}{loss_info}")
            
            logger.info(f"Epsilon={agent.epsilon:.4f} (Current)")
            
            logger.info(f"Stats: Win Rate={win_rate:.2f}, Avg Steps={avg_steps:.1f}, Max Turns Rate={max_turns_rate:.2f}")
            
            # Save charts every 1000 episodes
            if (episode + 1) % 1000 == 0:
                logger.info(f"Saving training charts at episode {episode+1}")
                save_charts(episode)
            
            # Add reflex agent specific logging
            if reflex_agent_games > 0:
                reflex_win_rate = reflex_agent_wins / reflex_agent_games
                reflex_loss_rate = reflex_agent_losses / reflex_agent_games
                avg_score_diff = reflex_agent_score_diff / reflex_agent_games
                logger.info(f"vs ReflexAgent: Games={reflex_agent_games}, "
                          f"Win Rate={reflex_win_rate:.2f}, Loss Rate={reflex_loss_rate:.2f}, "
                          f"Avg Score Diff={avg_score_diff:.2f}")
        
        # Evaluate agent periodically
        if (episode + 1) % args.eval_interval == 0:
            eval_results = evaluate_agent(agent, logger=logger)
            (random_avg_reward, random_win_rate, random_loss_rate, random_score_diff,
             reflex_avg_reward, reflex_win_rate, reflex_loss_rate, reflex_score_diff,
             q_stats) = eval_results
            
            # Store metrics for plotting
            eval_rewards.append(random_avg_reward)  # Using random opponent metrics as primary
            eval_win_rates.append(random_win_rate)
            eval_loss_rates.append(random_loss_rate)
            eval_max_turns_rates.append(0.0)  # Not tracked in new evaluation
            eval_avg_scores.append(-random_score_diff)  # Convert score diff to actual score
            eval_score_diffs.append(random_score_diff)
            
            # Store reflex agent metrics
            reflex_rewards.append(reflex_avg_reward)
            reflex_win_rates.append(reflex_win_rate)
            reflex_score_diffs.append(reflex_score_diff)
            
            # Update Q-value tracking if available
            if q_stats:
                q_value_avg.append(q_stats['mean'])
                q_value_min.append(q_stats['min'])
                q_value_max.append(q_stats['max'])
            
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
            elif random_score_diff > best_eval_reward + 1.0:  # At least 1 point better
                is_better = True
                reason = f"score diff improved from {best_eval_reward:.2f} to {random_score_diff:.2f}"
            # For smaller improvements, also consider win rate as tiebreaker
            elif random_score_diff > best_eval_reward and random_win_rate >= 0.5:
                is_better = True
                reason = f"score diff improved slightly to {random_score_diff:.2f} with good win rate {random_win_rate:.2f}"
            
            if is_better:
                best_eval_reward = random_score_diff
                agent.save(os.path.join(models_dir, 'best_model.pth'))
                logger.info(f"Saved best model with score diff {best_eval_reward:.2f} ({reason})")
            
            # Save checkpoint
            agent.save(os.path.join(models_dir, f'checkpoint_{episode+1}.pth'))
            
            # Save charts after each evaluation
            logger.info(f"Saving training charts after evaluation at episode {episode+1}")
            save_charts(episode)
    
    # Save final model
    final_model_path = os.path.join(models_dir, 'final_model.pth')
    agent.save(final_model_path)
    logger.info(f"Training complete. Final model saved to {final_model_path}")
    
    # Save final charts
    save_charts(args.episodes - 1)
    
    # Return final charts path for display
    return os.path.join(charts_dir, f'training_curves_ep{args.episodes}.png')

def main():
    """Main entry point for training."""
    args = parse_args()
    
    # Create directories
    os.makedirs(args.run_dir, exist_ok=True)
    
    # Setup logging and get the directory structure
    logger, logs_dir = setup_logging(args.log_level, args.run_dir)
    
    # Set up additional directories
    dirs = setup_run_directories(args.run_dir)
    
    # Train the agent and get the final chart path
    final_chart_path = train(args, logger, logs_dir)
    
    print(f"Training completed. Final chart saved to: {final_chart_path}")
    
    return final_chart_path

if __name__ == "__main__":
    main() 