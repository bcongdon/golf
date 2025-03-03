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
from collections import deque

# Setup logging
def setup_logging(log_level):
    """Set up logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamped log file
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("golf-ai")
    logger.info(f"Logging to {log_file}")
    return logger, logs_dir

# Logger will be properly configured in main

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN agent to play Golf')
    parser.add_argument('--episodes', type=int, default=300000, help='Number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size of the neural network')
    parser.add_argument('--embedding-dim', type=int, default=8, help='Dimension of card embeddings')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.975, help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Starting epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.05, help='Minimum epsilon for exploration')
    parser.add_argument('--epsilon-decay-episodes', type=int, default=150000, help='Number of episodes to decay epsilon from start to end')
    parser.add_argument('--epsilon-warmup', type=int, default=100000, help='Number of episodes to keep epsilon at start value')
    parser.add_argument('--target-update', type=int, default=1000, help='Target network update frequency')
    parser.add_argument('--eval-interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--logs-dir', type=str, default='logs', help='Directory to save logs and charts')
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
    parser.add_argument('--segment-tree', action='store_true', 
                        help='Use segment tree for more efficient sampling in replay buffer')
    parser.add_argument('--optimize-memory', action='store_true', 
                        help='Optimize memory usage with float16 for states')
    
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
    
    # Track Q-values during evaluation
    q_values_list = []
    
    # Set agent to evaluation mode
    agent.q_network.eval()
    
    # Pre-allocate arrays for better performance
    rewards_array = np.zeros(num_episodes, dtype=np.float32)
    wins_array = np.zeros(num_episodes, dtype=np.int32)
    losses_array = np.zeros(num_episodes, dtype=np.int32)
    max_turns_array = np.zeros(num_episodes, dtype=np.int32)
    agent_scores_array = np.zeros(num_episodes, dtype=np.float32)
    opponent_scores_array = np.zeros(num_episodes, dtype=np.float32)
    
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
                # Use torch.no_grad() to avoid tracking gradients during evaluation
                with torch.no_grad():
                    # Get Q-values for all actions
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    q_vals = agent.q_network(state_tensor).cpu().numpy()[0]
                    
                    # Record Q-values for valid actions
                    valid_q = [q_vals[action] for action in valid_actions]
                    q_values_list.extend(valid_q)
                    
                    action = agent.select_action(state, valid_actions, training=False)
            else:
                # Opponent's turn (random policy)
                valid_actions = env._get_valid_actions()
                # Use the common random policy implementation
                action = agent.random_action(valid_actions)
            
            next_state, reward, done, info = env.step(action)
            
            # Only update state when it's the agent's turn
            if current_player == agent_player:
                episode_reward += reward
                state = next_state
            
            # Check game result when done
            if done:
                if "max_turns_reached" in info and info["max_turns_reached"]:
                    max_turns_array[episode] = 1
                    if logger:
                        logger.warning(f"Episode {episode + 1} reached max turns limit!")
                
                # Get final scores
                scores = info.get("scores", [0, 0])
                
                # Record actual golf scores
                agent_score = scores[0]
                opponent_score = scores[1]
                agent_scores_array[episode] = agent_score
                opponent_scores_array[episode] = opponent_score
                
                # Determine win/loss based on scores (agent is player 0)
                if agent_score < opponent_score:  # Agent won
                    wins_array[episode] = 1
                    if logger:
                        logger.info(f"Episode {episode + 1} finished with a win! Score: {agent_score} vs {opponent_score}")
                elif agent_score > opponent_score:  # Agent lost
                    losses_array[episode] = 1
                    if logger:
                        logger.info(f"Episode {episode + 1} finished with a loss! Score: {agent_score} vs {opponent_score}")
                else:  # Tie
                    if logger:
                        logger.info(f"Episode {episode + 1} finished with a tie! Score: {agent_score} vs {opponent_score}")

        rewards_array[episode] = episode_reward
        if logger and episode % 25 == 0:
            logger.debug(f"Evaluation episode {episode}/{num_episodes}, reward: {episode_reward:.2f}")
    
    # Calculate metrics using vectorized operations
    avg_reward = rewards_array.mean()
    win_rate = wins_array.mean()
    loss_rate = losses_array.mean()
    max_turns_rate = max_turns_array.mean()
    
    # Calculate average golf scores (lower is better)
    avg_agent_score = agent_scores_array.mean()
    avg_opponent_score = opponent_scores_array.mean()
    score_diff = avg_opponent_score - avg_agent_score  # Positive means agent is better
    
    if logger:
        logger.debug(f"Evaluation complete - Avg reward: {avg_reward:.2f}")
        logger.info(f"Evaluation results - Win rate: {win_rate:.2f}, Loss rate: {loss_rate:.2f}, Tie rate: {1-win_rate-loss_rate:.2f}")
        logger.info(f"Average Golf Scores - Agent: {avg_agent_score:.2f}, Opponent: {avg_opponent_score:.2f}, Diff: {score_diff:.2f}")
        logger.info(f"Max turns reached in {max_turns_rate:.2f} of games")
    
    # Set agent back to training mode
    agent.q_network.train()
    
    # Calculate Q-value statistics
    q_stats = {}
    if q_values_list:
        q_stats = {
            'mean': np.mean(q_values_list),
            'min': np.min(q_values_list),
            'max': np.max(q_values_list),
            'std': np.std(q_values_list)
        }
        if logger:
            logger.info(f"Evaluation Q-values - Mean: {q_stats['mean']:.4f}, Min: {q_stats['min']:.4f}, Max: {q_stats['max']:.4f}, Std: {q_stats['std']:.4f}")
        
    return avg_reward, win_rate, loss_rate, max_turns_rate, avg_agent_score, score_diff, q_stats

def train(args, logger, logs_dir):
    """Train the DQN agent."""
    logger.info(f"Starting training with args: {args}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"Saving models to {args.save_dir}")
    
    # Initialize environment and agent
    env = GolfGame(num_players=2)  # Two players for self-play: agent and opponent
    state_size = 28  # New size: 14 card indices + 14 binary features
    action_size = 9   # Number of possible actions (removed 'knock' action)
    logger.info(f"Environment initialized with state_size={state_size}, action_size={action_size}")
    
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
        per_alpha=0.4,       # How much prioritization to use (0 = none, 1 = full)
        per_beta=0.4,        # Start value for importance sampling (0 = no correction, 1 = full)
        per_beta_increment=0.000001  # Beta increment per learning step
    )
    
    # Apply optimization settings
    if args.mixed_precision and agent.device.type == 'cuda':
        agent.use_amp = True
        agent.scaler = torch.cuda.amp.GradScaler()
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
            # This is handled in the agent's learn method
    logger.info(device_info)
    
    logger.info("Using Prioritized Experience Replay with win episode marking")
    
    # Load model if specified
    if args.load_model and os.path.exists(args.load_model):
        agent.load(args.load_model)
        logger.info(f"Loaded model from {args.load_model}")
    
    # Initialize opponent pool for self-play
    opponent_pool = []  # Will store snapshots of the agent at different training stages
    opponent_pool_size = 5  # Maximum number of past versions to keep
    opponent_update_frequency = 2000  # Add to pool every N episodes
    
    # Random opponent probability (will decrease over time)
    random_opponent_prob = 0.7  # Start with high probability of random opponent
    random_opponent_decay = 0.9999  # Decay factor
    
    # Metrics tracking
    rewards = []
    losses = []
    episode_steps = []
    epsilon_history = []
    q_value_avg = []
    q_value_min = []
    q_value_max = []
    
    # Evaluation metrics
    eval_rewards = []
    eval_win_rates = []
    eval_loss_rates = []
    eval_max_turns_rates = []
    eval_avg_scores = []
    eval_score_diffs = []
    eval_draw_rates = []
    
    # Stability tracking
    high_loss_count = 0
    consecutive_high_losses = 0
    
    # Best model tracking
    best_eval_reward = float('-inf')
    
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
            rewards_ema = calculate_ema(rewards, alpha=0.1)
            plt.plot(rewards_ema, color='red', linewidth=2, label='EMA (α=0.1)')
            
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        plt.subplot(4, 3, 2)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
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
            plt.title('Evaluation Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            
            plt.subplot(4, 3, 5)
            plt.plot(range(0, len(eval_win_rates) * args.eval_interval, args.eval_interval), eval_win_rates)
            plt.title('Win Rate vs Random Opponent')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate')
            
            plt.subplot(4, 3, 6)
            plt.plot(range(0, len(eval_loss_rates) * args.eval_interval, args.eval_interval), eval_loss_rates)
            plt.title('Loss Rate vs Random Opponent')
            plt.xlabel('Episode')
            plt.ylabel('Loss Rate')
        
        # Row 3: More evaluation metrics
        if eval_rewards:
            plt.subplot(4, 3, 7)
            plt.plot(range(0, len(eval_max_turns_rates) * args.eval_interval, args.eval_interval), eval_max_turns_rates)
            plt.title('Max Turns Rate')
            plt.xlabel('Episode')
            plt.ylabel('Rate of Games Reaching Max Turns')
            
            # Golf-specific metrics
            plt.subplot(4, 3, 8)
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
        
        plt.subplot(4, 3, 11)
        # Plot EMA of training loss
        if len(losses) > 0:
            loss_ema = calculate_ema(losses, alpha=0.1)
            plt.plot(losses, alpha=0.3, label='Loss')
            plt.plot(loss_ema, color='blue', linewidth=2, label='EMA (α=0.1)')
            plt.title('Training Loss with EMA')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.legend()
        
        plt.tight_layout()
        
        # Save to both model directory and logs directory
        plt.savefig(os.path.join(args.save_dir, 'training_curves.png'))
        plt.savefig(os.path.join(logs_dir, f'training_curves_ep{episode+1}.png'))
        plt.close()  # Close the figure to free memory
        
        # Save metrics as CSV for later analysis
        import pandas as pd
        if eval_rewards:
            eval_data = {
                'Episode': range(0, len(eval_rewards) * args.eval_interval, args.eval_interval),
                'Reward': eval_rewards,
                'WinRate': eval_win_rates,
                'LossRate': eval_loss_rates,
                'MaxTurnsRate': eval_max_turns_rates,
                'AvgScore': eval_avg_scores,
                'ScoreDiff': eval_score_diffs
            }
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
            train_data['Loss'] = losses + [None] * (len(rewards) - len(losses))
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
        if random.random() < random_opponent_prob:
            # Use random opponent with common random policy
            return agent.random_action(valid_actions)
        elif opponent_pool and random.random() < 0.7:  # 70% chance to use pool when available
            # Use an agent from the opponent pool
            opponent_idx = random.randint(0, len(opponent_pool) - 1)
            opponent_agent = opponent_pool[opponent_idx]
            # Set to evaluation mode for inference
            opponent_agent.q_network.eval()
            with torch.no_grad():
                return opponent_agent.select_action(state, valid_actions, training=False)
        else:
            # Use current agent with high exploration
            # Temporarily increase epsilon for more exploration
            original_epsilon = agent.epsilon
            agent.epsilon = max(0.3, agent.epsilon)  # At least 30% exploration
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
        temp_env = GolfGame(num_players=2)
        
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
            opponent = (temp_env.current_player + 1) % temp_env.num_players
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
                            # Check for potential learning instability
                            if loss > 100:
                                high_loss_count += 1
                                consecutive_high_losses += 1
                                logger.warning(f"High loss detected: {loss:.2f} at step {steps}, episode {episode+1}")
                                
                                # Automatic learning rate adjustment code removed
                                if consecutive_high_losses >= 5:
                                    # Reset consecutive high losses counter
                                    consecutive_high_losses = 0
                                    logger.warning(f"High loss streak ended")
                                    
                                    # Save a checkpoint for debugging
                                    checkpoint_path = os.path.join(args.save_dir, f'checkpoint_high_loss_ep{episode+1}.pt')
                                    agent.save(checkpoint_path)
                                    logger.info(f"Saved checkpoint due to high losses: {checkpoint_path}")
                            else:
                                consecutive_high_losses = 0  # Reset counter when loss is normal
                                
                            episode_losses.append(loss)
                    
                    # Clear batch after learning
                    experiences_batch = []
                    
                # Update state for the main agent
                state = next_state
                episode_reward += reward
            else:
                # Opponent's turn - use a mix of strategies
                opponent_action = select_opponent_action(state, valid_actions)
                
                # Take the action
                next_state, _, done, info = env.step(opponent_action)
                # No learning from opponent's experiences
                
                # Update state if needed
                if not done:
                    state = next_state
            
            steps += 1
            
            # Check if max turns was reached
            if done and "max_turns_reached" in info and info["max_turns_reached"]:
                max_turns_reached = True
        
        # Add current agent to opponent pool periodically
        if (episode + 1) % opponent_update_frequency == 0:
            # Create a deep copy of the current agent
            import copy
            opponent_snapshot = copy.deepcopy(agent)
            
            # Add to pool, maintaining maximum size
            opponent_pool.append(opponent_snapshot)
            if len(opponent_pool) > opponent_pool_size:
                # Remove oldest snapshot
                opponent_pool.pop(0)
            
            logger.info(f"Added agent snapshot to opponent pool (size: {len(opponent_pool)})")
            
            # Decay random opponent probability
            random_opponent_prob *= random_opponent_decay
            logger.info(f"Random opponent probability decayed to {random_opponent_prob:.4f}")
        
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
            
        # Update epsilon using piecewise linear decay strategy
        agent.update_epsilon()
            
        # Record epsilon value
        epsilon_history.append(agent.epsilon)
        
        # Sample Q-values periodically (every 100 episodes)
        if episode % 100 == 0:
            # Create a temporary environment for sampling to avoid affecting training
            temp_env = GolfGame(num_players=2)
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
        
        # Evaluate agent periodically
        if (episode + 1) % args.eval_interval == 0:
            eval_results = evaluate_agent(agent, logger=logger)
            avg_reward, win_rate, loss_rate, max_turns_rate, avg_score, score_diff, q_stats = eval_results
            
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
            
            # Save charts after each evaluation
            logger.info(f"Saving training charts after evaluation at episode {episode+1}")
            save_charts(episode)
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    agent.save(final_model_path)
    logger.info(f"Training complete. Final model saved to {final_model_path}")
    
    # Save final charts
    save_charts(args.episodes - 1)
    
    # Return final charts path for display
    return os.path.join(logs_dir, f'training_curves_ep{args.episodes}.png')

if __name__ == "__main__":
    args = parse_args()
    logger, logs_dir = setup_logging(args.log_level)
    
    # Override logs_dir if specified in args
    if args.logs_dir != 'logs':
        logs_dir = args.logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        logger.info(f"Using custom logs directory: {logs_dir}")
    
    try:
        logger.info("Starting Golf Card Game AI training")
        final_chart_path = train(args, logger, logs_dir)
        logger.info(f"Training complete. Final charts saved to {final_chart_path}")
    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")
        raise 