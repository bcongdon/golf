import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import sys

# Check if required packages are installed
required_packages = {
    "wandb": "wandb>=0.15.0",
    "stable_baselines3": "stable-baselines3>=2.0.0",
    "pettingzoo": "pettingzoo>=1.23.0",
    "supersuit": "supersuit>=3.8.1",
    "gymnasium": "gymnasium>=0.28.1"
}

missing_packages = []
for package, requirement in required_packages.items():
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(requirement)

if missing_packages:
    print("Missing required packages. Please install:")
    print("\n".join(f"  {pkg}" for pkg in missing_packages))
    print("\nYou can install all requirements with:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Import all other dependencies
import wandb
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym

from golf_pettingzoo_env import env as golf_env
from golf_game_v2 import GameConfig

# Custom Network Architecture
class GolfLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor with card embeddings and LSTM for sequential decision making.
    
    Features:
    - Card embeddings to represent card values
    - LSTM to maintain information about game state over time
    - Separate paths for revealed and unrevealed cards
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, 
                 embedding_dim: int = 8, lstm_hidden_size: int = 128, num_card_ranks: int = 13):
        super(GolfLSTMFeatureExtractor, self).__init__(observation_space, features_dim)
        
        self.observation_dim = observation_space.shape[0]
        self.embedding_dim = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        
        # Card embedding layer (card ranks -> embedding)
        # +1 for unknown/masked cards and +1 for None (no card)
        self.card_embedding = nn.Embedding(num_card_ranks + 2, embedding_dim)
        
        # Calculate sizes
        # We extract the first 14 elements as card indices (6 player + 6 opponent + discard + drawn)
        self.num_card_positions = 14
        self.non_card_features = self.observation_dim - self.num_card_positions
        
        # Combine embedded cards and other features
        embedded_size = self.num_card_positions * embedding_dim
        total_input_size = embedded_size + self.non_card_features
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=total_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Final feature extraction
        self.features_extractor = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # LSTM hidden state
        self.hidden = None
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def reset_hidden_states(self, batch_size=1):
        """Reset hidden states for LSTM."""
        device = next(self.parameters()).device
        self.hidden = (
            torch.zeros(1, batch_size, self.lstm_hidden_size).to(device),
            torch.zeros(1, batch_size, self.lstm_hidden_size).to(device)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract card indices and non-card features
        batch_size = observations.shape[0]
        card_indices = observations[:, :self.num_card_positions].long()
        non_card_features = observations[:, self.num_card_positions:]
        
        # Apply embedding to card indices
        embedded_cards = self.card_embedding(card_indices)
        
        # Reshape embedded cards: (batch_size, num_cards, embedding_dim) -> (batch_size, num_cards * embedding_dim)
        embedded_cards = embedded_cards.view(batch_size, -1)
        
        # Concatenate embedded cards with non-card features
        combined_features = torch.cat([embedded_cards, non_card_features], dim=1)
        
        # Add sequence dimension for LSTM (batch_size, 1, features)
        combined_features = combined_features.unsqueeze(1)
        
        # Initialize hidden state if not already done
        if self.hidden is None or self.hidden[0].shape[1] != batch_size:
            self.reset_hidden_states(batch_size)
            
        # Apply LSTM
        lstm_out, self.hidden = self.lstm(combined_features, self.hidden)
        
        # Extract output and apply final layers
        lstm_out = lstm_out.squeeze(1)  # Remove sequence dimension
        features = self.features_extractor(lstm_out)
        
        return features


# Custom PPO policy with LSTM
class LSTMActorCriticPolicy(ActorCriticPolicy):
    """
    Actor critic policy with LSTM for memory.
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # Pass modified network architecture
        kwargs.update({
            "features_extractor_class": GolfLSTMFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "embedding_dim": 8,
                "lstm_hidden_size": 128
            }
        })
        super(LSTMActorCriticPolicy, self).__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs
        )
    
    def _build_mlp_extractor(self):
        """
        Create the policy and value networks.
        Using simple MLP networks with separate outputs for actions and value.
        """
        # Define output dimensions for policy and value networks
        self.latent_dim_pi = 64
        self.latent_dim_vf = 64
        
        # Create a custom ModuleDict that includes the required attributes
        self.mlp_extractor = nn.ModuleDict({
            "policy": nn.Sequential(
                nn.Linear(self.features_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.latent_dim_pi),
                nn.ReLU()
            ),
            "value": nn.Sequential(
                nn.Linear(self.features_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.latent_dim_vf),
                nn.ReLU()
            )
        })
        
        # Add attributes expected by Stable Baselines3
        self.mlp_extractor.latent_dim_pi = self.latent_dim_pi
        self.mlp_extractor.latent_dim_vf = self.latent_dim_vf
        
    def forward(self, obs, deterministic=False):
        """
        Forward pass in all the networks.
        """
        # Reset LSTM hidden states at the beginning of episode
        if not hasattr(self, '_last_obs') or self._last_obs is None or obs.shape[0] != self._last_obs.shape[0]:
            self.reset_lstm_states(obs.shape[0])
        
        # Extract features using the feature extractor
        features = self.extract_features(obs)
        
        # Apply MLP extractor to get latent policy and value vectors
        latent_pi = self.mlp_extractor["policy"](features)
        latent_vf = self.mlp_extractor["value"](features)
        
        # Get distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Sample actions
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        
        # Get state value
        values = self.value_net(latent_vf)
        
        # Store current observation for next call
        self._last_obs = obs.clone()
        
        # Return actions, values, and log probs
        return actions, values, log_probs
    
    def reset_lstm_states(self, batch_size=1):
        """Reset LSTM hidden states."""
        try:
            if hasattr(self, 'features_extractor') and hasattr(self.features_extractor, 'reset_hidden_states'):
                self.features_extractor.reset_hidden_states(batch_size)
            
            if hasattr(self, '_last_obs'):
                self._last_obs = None
            else:
                # Initialize _last_obs attribute if it doesn't exist
                self._last_obs = None
        except Exception as e:
            print(f"Warning: Error resetting LSTM states: {e}")
            # Try to ensure we at least have _last_obs initialized
            self._last_obs = None


# Custom wandb callback
class WandbCallback(BaseCallback):
    """
    Callback for logging to Weights and Biases.
    """
    def __init__(self, verbose=0, log_freq=100):
        super(WandbCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.win_rates = deque(maxlen=100)
        self.last_time_steps = 0
    
    def _on_step(self) -> bool:
        """
        Log metrics to wandb every log_freq steps.
        """
        # Log training stats from the logger if available
        if self.n_calls % self.log_freq == 0:
            # Get training metrics from the model's logger
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                for key, value in self.model.logger.name_to_value.items():
                    wandb.log({f"train/{key}": value}, step=self.num_timesteps)
            
            # Log episode stats from internal storage 
            if self.episode_rewards:
                wandb.log({
                    "metrics/mean_reward": np.mean(self.episode_rewards),
                    "metrics/mean_episode_length": np.mean(self.episode_lengths),
                    "metrics/win_rate": np.mean(self.win_rates) if self.win_rates else 0
                }, step=self.num_timesteps)
            
            # Try to extract additional info from the environment if available
            try:
                if hasattr(self.training_env, 'get_attr'):
                    # For vectorized environments
                    infos = self.training_env.get_attr('infos')
                    if infos and len(infos) > 0:
                        info = infos[0]
                        wandb.log({f"env/{k}": v for k, v in info.items() if not isinstance(v, dict)}, 
                                  step=self.num_timesteps)
                elif hasattr(self.training_env, 'aec_env') and hasattr(self.training_env.aec_env, 'infos'):
                    # For our PettingZoo wrapper
                    for agent, info in self.training_env.aec_env.infos.items():
                        if agent == "player_0" and info:
                            wandb.log({f"env/{k}": v for k, v in info.items() if not isinstance(v, dict)}, 
                                      step=self.num_timesteps)
            except:
                pass  # Silently fail if we can't extract environment info
        
        return True
    
    def update_episode_stats(self, rewards, lengths, wins=None):
        """
        Update episode statistics.
        
        Args:
            rewards: List of episode rewards
            lengths: List of episode lengths
            wins: List of win indicators (1 for win, 0 for loss)
        """
        if isinstance(rewards, list):
            self.episode_rewards.extend(rewards)
        else:
            self.episode_rewards.append(rewards)
            
        if isinstance(lengths, list):
            self.episode_lengths.extend(lengths)
        else:
            self.episode_lengths.append(lengths)
            
        if wins is not None:
            if isinstance(wins, list):
                self.win_rates.extend(wins)
            else:
                self.win_rates.append(wins)


# Custom evaluation callback that handles LSTM reset
class LSTMEvalCallback(EvalCallback):
    """
    Evaluation callback for LSTM networks.
    Ensures LSTM states are reset between evaluation episodes.
    """
    def __init__(
        self,
        eval_env,
        eval_freq=10000,
        n_eval_episodes=5,
        log_path=None,
        best_model_save_path=None,
        deterministic=True,
        callback_on_new_best=None,
        callback_after_eval=None,
        verbose=1,
        warn=True,
    ):
        """
        Initialize the evaluation callback.
        
        Args:
            eval_env: The environment used for evaluation
            eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
            n_eval_episodes: The number of episodes to test the agent
            log_path: Path to a folder where the evaluations will be saved
            best_model_save_path: Path to a folder where the best model will be saved
            deterministic: Whether to use deterministic or stochastic actions
            callback_on_new_best: Called when there is a new best model
            callback_after_eval: Called after every evaluation
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
            warn: Whether to output additional warnings
        """
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            verbose=verbose,
            warn=warn
        )

    def _on_step(self) -> bool:
        """
        Evaluate the LSTM policy and reset hidden states between episodes.
        
        This version is designed to work directly with our AEC environment wrapper.
        """
        # Check if it's time to evaluate
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Collect evaluation metrics
            episode_rewards, episode_lengths, episode_wins = [], [], []
            
            for i in range(self.n_eval_episodes):
                # Reset LSTM states before each episode
                if hasattr(self.model.policy, 'reset_lstm_states'):
                    self.model.policy.reset_lstm_states()
                
                # Reset the evaluation environment
                try:
                    obs, info = self.eval_env.reset(seed=self.n_calls + i)
                except Exception as e:
                    print(f"Error resetting eval environment: {e}")
                    continue
                
                # Run episode
                done = False
                episode_reward = 0
                episode_length = 0
                
                while not done:
                    # Get action with policy
                    action, _ = self.model.predict(
                        obs, deterministic=self.deterministic
                    )
                    
                    # Check if action is valid
                    valid_actions = info.get("valid_actions", [])
                    if valid_actions and action not in valid_actions:
                        # If invalid, choose randomly from valid actions
                        action = np.random.choice(valid_actions)
                    
                    # Step environment
                    try:
                        obs, reward, terminated, truncated, info = self.eval_env.step(action)
                        done = terminated or truncated
                        
                        # Update metrics
                        episode_reward += reward
                        episode_length += 1
                    except Exception as e:
                        print(f"Error during evaluation step: {e}")
                        break
                
                # Record metrics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Check if player 0 won by comparing scores
                won = 0
                try:
                    if "scores" in info and len(info["scores"]) >= 2:
                        if info["scores"][0] < info["scores"][1]:
                            won = 1
                except Exception as e:
                    print(f"Error determining win: {e}")
                
                episode_wins.append(won)
                
                # Log episode result
                if self.verbose > 0:
                    print(f"Eval episode {i+1}/{self.n_eval_episodes}: "
                          f"reward={episode_reward:.2f}, length={episode_length}, won={bool(won)}")
            
            # Calculate mean and std metrics
            if episode_rewards:
                mean_reward = np.mean(episode_rewards)
                mean_length = np.mean(episode_lengths)
                win_rate = np.mean(episode_wins)
                
                # Store metrics for best model tracking
                self.last_mean_reward = mean_reward
                
                # Log to tensorboard if available
                if self.log_path is not None:
                    try:
                        from stable_baselines3.common.logger import Figure
                        import matplotlib.pyplot as plt
                        
                        # Create reward plot
                        fig = plt.figure(figsize=(6, 4))
                        plt.hist(episode_rewards, bins=10)
                        plt.title(f"Episode Rewards (mean: {mean_reward:.2f})")
                        plt.xlabel("Reward")
                        plt.ylabel("Count")
                        
                        # Log figure to tensorboard
                        self.logger.record("eval/figure", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
                        plt.close(fig)
                    except ImportError:
                        pass  # No matplotlib, skip figure logging
                
                # Update rewards and scores
                if len(self.evaluations_timesteps) == 0 or self.n_calls > self.evaluations_timesteps[-1]:
                    self.evaluations_timesteps.append(self.n_calls)
                    self.evaluations_rewards.append(episode_rewards)
                    if hasattr(self, 'evaluations_lengths'):
                        self.evaluations_lengths.append(episode_lengths)
                    
                    # Save best model
                    if mean_reward > self.best_mean_reward:
                        if self.verbose > 0:
                            print(f"New best mean reward: {mean_reward:.2f} vs {self.best_mean_reward:.2f}")
                        
                        self.best_mean_reward = mean_reward
                        
                        if self.best_model_save_path is not None:
                            self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                            if self.verbose > 0:
                                print(f"Saved best model to {os.path.join(self.best_model_save_path, 'best_model')}")
                        
                        # Trigger callback if needed
                        if self.callback_on_new_best is not None:
                            self.callback_on_new_best.on_step()
                
                # Log to wandb if running
                try:
                    if wandb.run is not None:
                        wandb.log({
                            "eval/mean_reward": mean_reward,
                            "eval/mean_length": mean_length,
                            "eval/win_rate": win_rate
                        }, step=self.num_timesteps)
                except:
                    pass  # Skip wandb logging if it fails
                
                # Log to console
                if self.verbose > 0:
                    print(f"Evaluation results at timestep {self.num_timesteps}:")
                    print(f"  Mean reward: {mean_reward:.2f} ± {np.std(episode_rewards):.2f}")
                    print(f"  Mean length: {mean_length:.2f} ± {np.std(episode_lengths):.2f}")
                    print(f"  Win rate: {win_rate:.2f}")
                    
                # Trigger callback if needed
                if self.callback_after_eval is not None:
                    self.callback_after_eval.on_step()
        
        return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a PPO+LSTM agent for Golf card game')
    
    # Environment parameters
    parser.add_argument('--num_players', type=int, default=2, help='Number of players')
    parser.add_argument('--grid_rows', type=int, default=2, help='Number of rows in card grid')
    parser.add_argument('--grid_cols', type=int, default=3, help='Number of columns in card grid')
    parser.add_argument('--initial_revealed', type=int, default=2, help='Initial revealed cards per player')
    parser.add_argument('--max_turns', type=int, default=200, help='Maximum number of turns')
    parser.add_argument('--normalize_rewards', action='store_true', help='Normalize rewards')
    
    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=2000000, help='Total timesteps to train')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n_steps', type=int, default=1024, help='Number of steps per update')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--clip_range', type=float, default=0.2, help='PPO clip range')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm')
    
    # Evaluation parameters
    parser.add_argument('--eval_freq', type=int, default=20000, help='Evaluation frequency in timesteps')
    parser.add_argument('--n_eval_episodes', type=int, default=10, help='Number of episodes for evaluation')
    
    # Logging and saving
    parser.add_argument('--log_freq', type=int, default=1000, help='Logging frequency in timesteps')
    parser.add_argument('--save_freq', type=int, default=50000, help='Model saving frequency in timesteps')
    parser.add_argument('--model_dir', type=str, default='models/ppo_lstm', help='Directory to save models')
    parser.add_argument('--wandb_project', type=str, default='golf-ppo-lstm', help='Wandb project name')
    parser.add_argument('--log_dir', type=str, default='logs/ppo_lstm', help='Directory to save logs')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--run_name', type=str, default=None, help='Name for this run')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level (0: no output, 1: info, 2: debug)')
    
    return parser.parse_args()


def make_env(args, seed=None):
    """
    Create the Golf environment using the PettingZoo AEC environment directly.
    This doesn't use any parallelism wrappers, just wraps the AEC environment to
    make it compatible with standard gym/SB3 interfaces.
    """
    # Create game config
    config = GameConfig(
        num_players=args.num_players,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        initial_revealed=args.initial_revealed,
        max_turns=args.max_turns,
        normalize_rewards=args.normalize_rewards
    )
    
    try:
        # Create a simple wrapper that adapts the PettingZoo AEC environment for SB3
        # without using any parallelism
        class AECGymWrapper(gym.Env):
            """Wrapper for PettingZoo AEC environment that follows gym interface."""
            
            def __init__(self, config, render_mode=None):
                # Create the AEC environment
                self.aec_env = golf_env(config=config, render_mode=render_mode)
                
                # Get observation and action spaces for player_0
                self.observation_space = self.aec_env.observation_spaces["player_0"]
                self.action_space = self.aec_env.action_spaces["player_0"]
                
                # Initialize state
                self.current_obs = None
                self.player0_agent = "player_0"
            
            def reset(self, seed=None, options=None):
                """Reset the environment and return initial observation."""
                # Reset the AEC environment
                self.aec_env.reset(seed=seed)
                
                # Get observation for player_0
                self.current_obs = self.aec_env.observe(self.player0_agent)
                
                # Get valid actions
                valid_actions = self.aec_env.get_valid_actions(self.player0_agent)
                
                return self.current_obs, {"valid_actions": valid_actions}
            
            def step(self, action):
                """Take a step for player_0 and auto-play for other agents."""
                total_reward = 0
                terminated = False
                truncated = False
                info = {}
                
                # Run until it's player_0's turn again or the game ends
                for agent in self.aec_env.agent_iter():
                    # Get current state
                    obs, reward, term, trunc, agent_info = self.aec_env.last()
                    
                    # Update info
                    if agent_info:
                        info.update(agent_info)
                    
                    # Check if game is over
                    if term or trunc:
                        terminated = term
                        truncated = trunc
                        break
                    
                    # Take action based on agent
                    if agent == self.player0_agent:
                        # This is our learning agent - use the provided action
                        self.aec_env.step(action)
                        total_reward += reward
                    else:
                        # Random opponent
                        valid_actions = self.aec_env.get_valid_actions(agent)
                        if valid_actions:
                            random_action = np.random.choice(valid_actions)
                            self.aec_env.step(random_action)
                        else:
                            # No valid actions, just use a safe default
                            self.aec_env.step(0)
                            
                    # If we're done with player_0's turn, we can exit
                    if agent == self.player0_agent:
                        break
                
                # Get observation for next step
                if not (terminated or truncated):
                    self.current_obs = self.aec_env.observe(self.player0_agent)
                    info["valid_actions"] = self.aec_env.get_valid_actions(self.player0_agent)
                else:
                    # Get final scores if the game is done
                    if hasattr(self.aec_env, 'infos'):
                        scores = {
                            a: self.aec_env.infos[a].get("score", float('inf'))
                            for a in self.aec_env.agents
                        }
                        info["scores"] = [scores.get("player_0", float('inf')), 
                                         scores.get("player_1", float('inf'))]
                
                return self.current_obs, total_reward, terminated, truncated, info
            
            def render(self):
                """Render the environment."""
                return self.aec_env.render()
            
            def close(self):
                """Close the environment."""
                return self.aec_env.close()
        
        # Create the wrapped environment
        env = AECGymWrapper(config, render_mode="ansi")
        
        # Apply episode statistics recording
        from gymnasium.wrappers import RecordEpisodeStatistics
        env = RecordEpisodeStatistics(env)
        
        # Set seed if provided
        if seed is not None:
            # Note: seed is passed during reset
            env.reset(seed=seed)
        
        return env
        
    except Exception as e:
        print(f"Error creating environment: {e}")
        raise


def setup_directories(args):
    """Set up directories for logs and models."""
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create best model directory
    best_model_path = os.path.join(args.model_dir, 'best')
    os.makedirs(best_model_path, exist_ok=True)
    
    return best_model_path


def train(args):
    """Train a PPO+LSTM agent."""
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Set up directories
    best_model_path = setup_directories(args)
    
    # Check if tensorboard is available
    tensorboard_log = None
    try:
        import tensorboard
        tensorboard_log = args.log_dir
        print("Tensorboard logging enabled")
    except ImportError:
        print("Tensorboard not found. Disabling tensorboard logging.")
        # Set environment variable to disable tensorboard warning
        os.environ['STABLE_BASELINES_TENSORBOARD_DISABLED'] = 'True'
    
    # Initialize wandb
    run_name = args.run_name if args.run_name else f"ppo_lstm_{int(time.time())}"
    try:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=run_name,
            dir=args.log_dir
        )
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        print("Training will continue without wandb logging.")
    
    # Create environments
    try:
        # Training environment
        env = make_env(args, seed=args.seed)
        print("Created training environment successfully")

        # Evaluation environment
        eval_env = make_env(args, seed=args.seed+1)
        print("Created evaluation environment successfully")
    except Exception as e:
        print(f"Fatal error creating environment: {e}")
        raise
    
    # Create model
    try:
        model = PPO(
            policy=LSTMActorCriticPolicy,
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            verbose=args.verbose,
            tensorboard_log=tensorboard_log
        )
        
        # Load pre-trained model if specified
        if args.load_model:
            model = PPO.load(args.load_model, env=env)
            print(f"Loaded model from {args.load_model}")
    except Exception as e:
        print(f"Error creating model: {e}")
        raise
    
    # Create callbacks
    callback_list = []
    
    # WandbCallback for logging metrics
    try:
        wandb_callback = WandbCallback(verbose=args.verbose, log_freq=args.log_freq)
        callback_list.append(wandb_callback)
    except Exception as e:
        print(f"Warning: Failed to create wandb callback: {e}")
    
    # Evaluation callback
    try:
        eval_callback = LSTMEvalCallback(
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            log_path=args.log_dir,
            best_model_save_path=best_model_path,
            deterministic=True,
            verbose=args.verbose
        )
        callback_list.append(eval_callback)
    except Exception as e:
        print(f"Warning: Failed to create eval callback: {e}")
    
    # Save callback to save model periodically
    try:
        class SaveCallback(BaseCallback):
            def __init__(self, save_freq, save_path, verbose=0):
                super(SaveCallback, self).__init__(verbose)
                self.save_freq = save_freq
                self.save_path = save_path
            
            def _on_step(self):
                if self.n_calls % self.save_freq == 0:
                    path = os.path.join(self.save_path, f'model_{self.n_calls}')
                    self.model.save(path)
                    if self.verbose > 0:
                        print(f"Saved model to {path}")
                return True
        
        save_callback = SaveCallback(
            save_freq=args.save_freq,
            save_path=args.model_dir,
            verbose=args.verbose
        )
        callback_list.append(save_callback)
    except Exception as e:
        print(f"Warning: Failed to create save callback: {e}")
    
    # Train model with error handling
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_list if callback_list else None
        )
        
        # Save final model
        final_model_path = os.path.join(args.model_dir, 'final_model')
        model.save(final_model_path)
        print(f"Saved final model to {final_model_path}")
    except Exception as e:
        print(f"Error during training: {e}")
        # Try to save model even if training failed
        try:
            final_model_path = os.path.join(args.model_dir, 'interrupted_model')
            model.save(final_model_path)
            print(f"Saved interrupted model to {final_model_path}")
        except:
            print("Failed to save interrupted model")
        raise
    finally:
        # Close wandb
        if wandb.run is not None:
            wandb.finish()
        
        # Close environments
        env.close()
        eval_env.close()
    
    return model


def evaluate(model, args, num_episodes=10):
    """Evaluate a trained model."""
    # Create environment
    env = make_env(args, seed=args.seed+1000)
    
    # Initialize metrics
    rewards = []
    lengths = []
    win_count = 0
    
    for i in range(num_episodes):
        try:
            # Reset environment
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            # Reset LSTM states
            if hasattr(model.policy, 'reset_lstm_states'):
                model.policy.reset_lstm_states()
            
            while not done:
                try:
                    # Get action from model
                    action, _states = model.predict(obs, deterministic=True)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Update metrics
                    episode_reward += reward
                    episode_length += 1
                except Exception as e:
                    print(f"Error during evaluation step in episode {i+1}: {e}")
                    break
            
            # Try to determine if the AI won
            try:
                if "scores" in info and len(info["scores"]) >= 2:
                    if info["scores"][0] < info["scores"][1]:
                        win_count += 1
            except Exception as e:
                print(f"Error determining win in episode {i+1}: {e}")
            
            # Log episode results
            rewards.append(episode_reward)
            lengths.append(episode_length)
            print(f"Episode {i+1}/{num_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        except Exception as e:
            print(f"Error in episode {i+1}: {e}")
            continue
    
    # Print summary
    if len(rewards) > 0:
        mean_reward = np.mean(rewards)
        mean_length = np.mean(lengths)
        win_rate = win_count / len(rewards)
        
        print(f"\nEvaluation Summary:")
        print(f"Mean Reward: {mean_reward:.2f} +/- {np.std(rewards) if len(rewards) > 1 else 0:.2f}")
        print(f"Mean Episode Length: {mean_length:.2f} +/- {np.std(lengths) if len(lengths) > 1 else 0:.2f}")
        print(f"Win Rate: {win_rate:.2f}")
    else:
        print("\nNo successful evaluation episodes completed")
        mean_reward = 0
        win_rate = 0
    
    # Close environment
    env.close()
    
    return mean_reward, win_rate


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Print arguments
        print("Arguments:")
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")
        
        # Train model
        print("\nStarting training...")
        model = train(args)
        
        # Evaluate model
        print("\nEvaluating final model...")
        try:
            mean_reward, win_rate = evaluate(model, args)
            
            # Final wandb log
            try:
                if wandb.run is not None:
                    wandb.log({
                        "final/mean_reward": mean_reward,
                        "final/win_rate": win_rate
                    })
            except Exception as e:
                print(f"Error logging final metrics to wandb: {e}")
        except Exception as e:
            print(f"Error during final evaluation: {e}")
        
        print("\nTraining and evaluation complete!")
        return 0
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 