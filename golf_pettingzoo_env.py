import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import agent_selector

from golf_game_v2 import GolfGame, GameConfig, Action, GameInfo, Card

class GolfPettingZooEnv(AECEnv):
    """
    PettingZoo wrapper for the Golf card game environment.
    
    This adapter makes the Golf game compatible with PettingZoo's multi-agent
    reinforcement learning interface.
    """
    metadata = {"render_modes": ["human", "ansi"], "name": "golf_v0", "is_parallelizable": False}
    
    def __init__(self, config: Optional[GameConfig] = None, render_mode: Optional[str] = None):
        """
        Initialize the PettingZoo environment.
        
        Args:
            config: Configuration for the Golf game. If None, default config is used.
            render_mode: The render mode to use ('human' or 'ansi')
        """
        super().__init__()
        
        # Initialize the underlying game
        if config is None:
            config = GameConfig()
        self.game = GolfGame(config)
        self.render_mode = render_mode

        raise "Foo"

        
        # PettingZoo requires a list of agent names
        self.possible_agents = [f"player_{i}" for i in range(self.game.config.num_players)]
        self.agents = self.possible_agents.copy()
        
        # Get cards per player for observation space calculation
        self.cards_per_player = self.game.config.grid_rows * self.game.config.grid_cols
        
        # Calculate observation space size based on the game's observation
        obs_size = (2 * self.cards_per_player  # Player hands
                   + 2  # Discard and drawn card
                   + 2 * self.cards_per_player  # Revealed flags
                   + 3)  # Drawn from discard, final round flags, and turn progress
        
        # Define observation and action spaces
        self.observation_spaces = {
            agent: spaces.Box(
                low=0,
                high=14,  # Max value is Card enum length + 1 (unknown card)
                shape=(obs_size,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: spaces.Discrete(len(Action))
            for agent in self.possible_agents
        }
        
        # Initialize additional info
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        
        # Initialize the agent selection
        self._agent_selector = None
        self.agent_selection = None
        
        # Additional tracking variables
        self.num_moves = 0
        self.current_valid_actions = []
        
    def observe(self, agent):
        """
        Get the observation for a specific agent.
        
        Args:
            agent: The agent ID
        
        Returns:
            The observation for the agent
        """
        if agent not in self.agents:
            return None
        
        agent_idx = int(agent.split("_")[1])
        
        # Get the base observation
        obs = self.game._get_observation()
        
        # If this is not the current player, flip the perspective
        if agent_idx != self.game.current_player:
            # Flip the perspective so each agent sees themselves as player 0
            obs = self.game.flip_state_perspective(obs)
        
        return obs
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options for environment reset
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the underlying game
        self.game.reset()
        
        # Reset PettingZoo specific attributes
        self.agents = self.possible_agents.copy()
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Set up the agent selector to cycle through agents in order
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        # Make sure the agent selection matches the current player in the game
        while int(self.agent_selection.split("_")[1]) != self.game.current_player:
            self.agent_selection = self._agent_selector.next()
        
        # Update valid actions
        self.current_valid_actions = self.game._get_valid_actions()
        
        # Update infos with valid actions
        for agent in self.agents:
            self.infos[agent]["valid_actions"] = [a.value for a in self.current_valid_actions]
        
        self.num_moves = 0
    
    def step(self, action):
        if (
            self.terminations[self.agent_selection] 
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        agent_idx = int(agent.split("_")[1])
        
        assert agent_idx == self.game.current_player, "Wrong agent is taking an action"
        
        valid_actions = self.game._get_valid_actions()
        valid_action_values = [a.value for a in valid_actions]
        
        if action not in valid_action_values:
            self.rewards[agent] = -1.0
            self.infos[agent]["error"] = f"Invalid action: {action}"
            return
        
        obs, reward, done, info = self.game.step(action)
        
        self.rewards[agent] = reward
        
        if done:
            scores = info.scores
            min_score = min(scores)
            num_tied_for_first = scores.count(min_score)
            
            for a in self.agents:
                a_idx = int(a.split("_")[1])
                player_score = scores[a_idx]
                if player_score == min_score:
                    if num_tied_for_first > 1:
                        margin = sum(scores) / len(scores) - player_score
                        terminal_reward = 2.5 + 0.3 * margin
                    else:
                        margin = sum(scores) / len(scores) - player_score
                        terminal_reward = 5.0 + 0.3 * margin
                else:
                    terminal_reward = -0.3 * (player_score - min_score)
                self.rewards[a] = terminal_reward
            
            if isinstance(info, GameInfo) and info.max_turns_reached:
                for a in self.agents:
                    self.truncations[a] = True
            else:
                for a in self.agents:
                    self.terminations[a] = True
            
            for a in self.agents:
                a_idx = int(a.split("_")[1])
                self.infos[a].update({
                    "episode_length": self.num_moves,
                    "score": info.scores[a_idx],
                    "final_round": self.game.final_round,
                    "game_end_reason": info.game_end_reason if hasattr(info, "game_end_reason") else None
                })
        else:
            self.agent_selection = self._agent_selector.next()
            while int(self.agent_selection.split("_")[1]) != self.game.current_player:
                self.agent_selection = self._agent_selector.next()
            
            self.current_valid_actions = self.game._get_valid_actions()
            for a in self.agents:
                self.infos[a]["valid_actions"] = [a.value for a in self.current_valid_actions]
        
        self.num_moves += 1
    
    def render(self):
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
    
    def close(self):
        """
        Clean up resources.
        """
        pass
    
    def state(self):
        """
        Return the global state of the environment.
        
        Returns:
            A dictionary containing the full game state
        """
        return {
            "player_hands": [list(hand) for hand in self.game.player_hands],
            "player_revealed": [list(revealed) for revealed in self.game.player_revealed],
            "discard_pile": list(self.game.discard_pile),
            "deck": list(self.game.deck),
            "drawn_card": self.game.drawn_card,
            "drawn_from_discard": self.game.drawn_from_discard,
            "current_player": self.game.current_player,
            "turn_count": self.game.turn_count,
            "final_round": self.game.final_round,
            "final_round_starter": self.game.final_round_starter,
            "game_over": any(self.terminations.values()) or any(self.truncations.values())
        }
    
    def observation_space(self, agent):
        """
        Get the observation space for a specific agent.
        
        Args:
            agent: The agent ID
        
        Returns:
            The observation space for the agent
        """
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        """
        Get the action space for a specific agent.
        
        Args:
            agent: The agent ID
        
        Returns:
            The action space for the agent
        """
        return self.action_spaces[agent]
    
    def get_valid_actions(self, agent=None):
        """
        Get the list of valid actions for an agent.
        
        Args:
            agent: The agent ID. If None, use the current agent.
            
        Returns:
            List of valid action indices
        """
        if agent is None:
            agent = self.agent_selection
        
        # Only the current agent can take actions
        if agent != self.agent_selection:
            return []
        
        return [a.value for a in self.current_valid_actions]
    
    def seed(self, seed=None):
        """
        Set the seed for the environment. This is kept for backward compatibility.
        
        Args:
            seed: The seed to use
        """
        if seed is not None:
            np.random.seed(seed)
    
    def last(self):
        """
        Returns the observation, reward, terminations, truncations, and info for the current agent.
        """
        agent = self.agent_selection
        observation = self.observe(agent)
        return (
            observation,
            self.rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def _was_dead_step(self, action):
        """
        Handle steps for agents that are already done.
        """
        # Select the next agent
        self.agent_selection = self._agent_selector.next()
        
        # Make sure the agent selection matches the current player in the game
        # This is only needed if the game is still ongoing
        if not (all(self.terminations.values()) or all(self.truncations.values())):
            while int(self.agent_selection.split("_")[1]) != self.game.current_player:
                self.agent_selection = self._agent_selector.next()


# Create a wrapped version that follows PettingZoo conventions
def env(config=None, render_mode=None):
    """
    Create a PettingZoo-compatible Golf environment.
    
    Args:
        config: Configuration for the Golf game
        render_mode: The render mode to use
    
    Returns:
        A wrapped PettingZoo environment
    """
    env = GolfPettingZooEnv(config=config, render_mode=render_mode)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env 