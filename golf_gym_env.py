import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from golf_game_v2 import GolfGame, GameConfig, Action, GameInfo

class GolfGymEnv(gym.Env):
    """
    Gymnasium wrapper for the Golf card game environment.
    
    This adapter makes the GolfGame compatible with standard Gymnasium-based
    reinforcement learning algorithms.
    """
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, config: Optional[GameConfig] = None, render_mode: Optional[str] = None):
        """
        Initialize the Gym environment.
        
        Args:
            config: Configuration for the Golf game. If None, default config is used.
            render_mode: The render mode to use ('human' or 'ansi')
        """
        # Initialize the underlying game
        self.game = GolfGame(config)
        self.render_mode = render_mode
        
        # Get cards per player for observation space calculation
        self.cards_per_player = self.game.config.grid_rows * self.game.config.grid_cols
        
        # Calculate observation space size based on the game's observation
        obs_size = (2 * self.cards_per_player  # Player hands
                   + 2  # Discard and drawn card
                   + 2 * self.cards_per_player  # Revealed flags
                   + 3)  # Drawn from discard, final round flags, and turn progress
        
        # Define observation space
        # The observation contains card indices (0-13) and binary features (0-1)
        self.observation_space = spaces.Box(
            low=0,
            high=14,  # Max value is Card enum length + 1 (unknown card)
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Define action space using the Action enum
        self.action_space = spaces.Discrete(len(Action))
        
        # Keep track of the current valid actions
        self.current_valid_actions = []
        
        # Initialize episode information
        self.episode_rewards = 0
        self.episode_length = 0
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed
            options: Additional options for environment reset
            
        Returns:
            Tuple of (observation, info)
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the underlying game
        obs = self.game.reset()
        
        # Reset episode information
        self.episode_rewards = 0
        self.episode_length = 0
        
        # Update valid actions
        self.current_valid_actions = self.game._get_valid_actions()
        
        # Info dictionary
        info = {
            "valid_actions": [a.value for a in self.current_valid_actions]
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take (int corresponding to Action enum)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        valid_actions = self.game._get_valid_actions()
        if action not in [a.value for a in valid_actions]:
            # If invalid action, return current state with negative reward
            obs = self.game._get_observation()
            return obs, -1.0, False, False, {"error": f"Invalid action: {action}"}
        
        # Take action in the underlying game
        obs, reward, done, info = self.game.step(action)
        
        # Update episode information
        self.episode_rewards += reward
        self.episode_length += 1
        
        # Update valid actions for next step
        if not done:
            self.current_valid_actions = self.game._get_valid_actions()
        
        # Add episode stats to info
        info_dict = {
            "episode_reward": self.episode_rewards if done else None,
            "episode_length": self.episode_length if done else None,
            "valid_actions": [a.value for a in self.current_valid_actions],
            "turn_count": self.game.turn_count,
            "final_round": self.game.final_round,
        }
        
        # In Gymnasium, done is split into terminated and truncated
        # terminated: episode ended due to reaching a terminal state
        # truncated: episode ended due to reaching max steps or external termination
        terminated = done
        truncated = False
        
        # If max turns reached, it's truncated
        if isinstance(info, GameInfo) and info.max_turns_reached:
            terminated = False
            truncated = True
        
        # Convert GameInfo to dict and merge with info_dict
        if isinstance(info, GameInfo):
            for key, value in info.__dict__.items():
                info_dict[key] = value
        
        return obs, reward, terminated, truncated, info_dict
    
    def render(self) -> Optional[str]:
        """
        Render the environment.
        
        Returns:
            String representation of the environment if mode is 'ansi', else None
        """
        render_str = self.game.render()
        
        if self.render_mode == 'human':
            print(render_str)
            return None
        elif self.render_mode == 'ansi':
            return render_str
        else:
            return render_str
    
    def close(self) -> None:
        """
        Clean up resources.
        """
        pass
    
    def get_valid_actions(self) -> List[int]:
        """
        Get the list of valid actions for the current state.
        
        Returns:
            List of valid action indices
        """
        return [a.value for a in self.current_valid_actions]
    
    def flip_state_perspective(self, state: np.ndarray) -> np.ndarray:
        """
        Flip the state to represent the opponent's perspective.
        
        This is useful for self-play or multi-agent environments.
        
        Args:
            state: Current state observation
            
        Returns:
            State from the opponent's perspective
        """
        return self.game.flip_state_perspective(state) 