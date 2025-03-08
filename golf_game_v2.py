from enum import IntEnum
import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Union, Set
from dataclasses import dataclass

class Action(IntEnum):
    """Enumeration of possible actions in the game."""
    DRAW_FROM_DECK = 0
    DRAW_FROM_DISCARD = 1
    REPLACE_0 = 2
    REPLACE_1 = 3
    REPLACE_2 = 4
    REPLACE_3 = 5
    REPLACE_4 = 6
    REPLACE_5 = 7
    DISCARD = 8

class Card(IntEnum):
    """Enumeration of card ranks."""
    ACE = 0
    TWO = 1
    THREE = 2
    FOUR = 3
    FIVE = 4
    SIX = 5
    SEVEN = 6
    EIGHT = 7
    NINE = 8
    TEN = 9
    JACK = 10
    QUEEN = 11
    KING = 12

    @property
    def point_value(self) -> int:
        """Get the point value of the card."""
        if self == Card.ACE:
            return 1
        if self == Card.TWO:
            return -2
        if self >= Card.JACK:
            return 10
        return int(self) + 1  # THREE=3, FOUR=4, etc.

    @property
    def symbol(self) -> str:
        """Get the symbol representation of the card."""
        symbols = "A23456789TJQK"
        return symbols[int(self)]

@dataclass
class GameConfig:
    """Configuration for the Golf game."""
    num_players: int = 2
    grid_rows: int = 2
    grid_cols: int = 3
    initial_revealed: int = 2
    max_turns: int = 200
    normalize_rewards: bool = False
    copies_per_rank: int = 4  # Number of copies of each rank (simulating suits)

class GolfGame:
    """
    Implementation of the Golf card game environment with improved structure.
    
    Key improvements:
    - Enums for actions and cards
    - Configuration class for game parameters
    - Better type hints and documentation
    - More modular code structure
    - Constants moved to configuration
    - Better error handling
    """
    
    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the game to initial state and return the initial observation."""
        self._init_deck()
        self._init_players()
        self._init_game_state()
        return self._get_observation()
    
    def _init_deck(self) -> None:
        """Initialize the deck of cards."""
        self.deck = []
        for _ in range(self.config.copies_per_rank):
            for rank in Card:
                self.deck.append(rank)
        random.shuffle(self.deck)
    
    def _init_players(self) -> None:
        """Initialize player hands and revealed cards."""
        cards_per_player = self.config.grid_rows * self.config.grid_cols
        self.player_hands = []
        self.revealed_cards = []
        
        for _ in range(self.config.num_players):
            # Deal cards
            hand = []
            for _ in range(cards_per_player):
                hand.append(self.deck.pop())
            self.player_hands.append(hand)
            
            # Reveal initial cards
            revealed = set(random.sample(range(cards_per_player), 
                                      self.config.initial_revealed))
            self.revealed_cards.append(revealed)
    
    def _init_game_state(self) -> None:
        """Initialize game state variables."""
        self.current_player = 0
        self.discard_pile = [self.deck.pop()]
        self.game_over = False
        self.drawn_card = None
        self.drawn_from_discard = False
        self.final_round = False
        self.last_player = None
        self.turn_count = 0
    
    def _calculate_column_score(self, hand: List[Card], col: int) -> int:
        """Calculate score for a single column."""
        top_idx = col
        bottom_idx = col + self.config.grid_cols
        
        top_card = hand[top_idx]
        bottom_card = hand[bottom_idx]
        
        if top_card == bottom_card:
            return -4 if top_card == Card.TWO else 0
        return top_card.point_value + bottom_card.point_value
    
    def _calculate_score(self, player: int) -> int:
        """Calculate the total score for a player."""
        hand = self.player_hands[player]
        return sum(self._calculate_column_score(hand, col) 
                  for col in range(self.config.grid_cols))
    
    def _get_valid_actions(self) -> List[Action]:
        """Return list of valid actions for current player."""
        if self.game_over:
            return []
            
        if self.drawn_card is None:
            valid = [Action.DRAW_FROM_DECK]
            if self.discard_pile:
                valid.append(Action.DRAW_FROM_DISCARD)
            return valid
        
        valid = [Action(i) for i in range(Action.REPLACE_0, Action.REPLACE_5 + 1)]
        if not self.drawn_from_discard:
            valid.append(Action.DISCARD)
        return valid
    
    def _normalize_reward(self, reward: float, is_terminal: bool = False) -> float:
        """Normalize rewards to a consistent scale."""
        if not self.config.normalize_rewards:
            return reward
            
        if is_terminal:
            if reward > 0:
                return 0.5 + min(0.5, reward / 10.0)
            return -0.5 + max(-0.5, reward / 10.0)
        return np.clip(reward / 4.0, -0.5, 0.5)
    
    def _get_observation(self) -> np.ndarray:
        """Convert the game state to a neural network input representation.
        
        The observation is a 1D array with the following structure:
        - First 2*cards_per_player elements: The visible cards in the current player's and opponent's hands
        - Next 2 elements: The top card of the discard pile and the drawn card (if any)
        - Next 2*cards_per_player elements: Binary flags indicating which cards are revealed
        - Last 3 elements: 
            - Flag indicating if the drawn card came from the discard pile
            - Flag indicating if the game is in the final round
            - Turn progress (normalized value between 0 and 1, where 1 means max_turns reached)
        """
        cards_per_player = self.config.grid_rows * self.config.grid_cols
        obs_size = (2 * cards_per_player  # Player hands
                   + 2  # Discard and drawn card
                   + 2 * cards_per_player  # Revealed flags
                   + 3)  # Drawn from discard, final round flags, and turn progress
        
        obs = np.zeros(obs_size, dtype=np.float32)
        unknown_card = len(Card)  # 13 for unknown cards
        
        # Initialize all cards as unknown
        for i in range(2 * cards_per_player):
            obs[i] = unknown_card
        
        # Current player's hand
        for i, rank in enumerate(self.player_hands[self.current_player]):
            if i in self.revealed_cards[self.current_player]:
                obs[i] = rank
                obs[2 * cards_per_player + i] = 1.0  # Revealed flag
        
        # Opponent's visible cards
        opponent = (self.current_player + 1) % self.config.num_players
        for i, rank in enumerate(self.player_hands[opponent]):
            if i in self.revealed_cards[opponent]:
                obs[cards_per_player + i] = rank
                obs[2 * cards_per_player + cards_per_player + i] = 1.0
        
        # Discard and drawn cards
        if self.discard_pile:
            obs[2 * cards_per_player] = self.discard_pile[-1]
        if self.drawn_card is not None:
            obs[2 * cards_per_player + 1] = self.drawn_card
            if self.drawn_from_discard:
                obs[-3] = 1.0
        
        # Final round flag
        if self.final_round:
            obs[-2] = 1.0
            
        # Turn progress (how close we are to max_turns)
        # Normalize to [0, 1] where 0 is start and 1 is max_turns
        obs[-1] = self.turn_count / self.config.max_turns
            
        return obs
    
    def step(self, action: Union[Action, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step in the environment."""
        action = Action(action)
        if action not in self._get_valid_actions():
            invalid_info = {
                "error": "Invalid action",
                "turn_count": self.turn_count,
                "max_turns": self.config.max_turns,
                "turn_progress": self.turn_count / self.config.max_turns
            }
            return (self._get_observation(), 
                    self._normalize_reward(-1), 
                    False, 
                    invalid_info)
        
        reward = 0
        info = {}
        
        # Handle drawing phase
        if self.drawn_card is None:
            # Increment turn counter at the start of each complete turn
            self.turn_count += 1
            
            # Add turn progress information to info dictionary
            info["turn_count"] = self.turn_count
            info["max_turns"] = self.config.max_turns
            info["turn_progress"] = self.turn_count / self.config.max_turns
            
            # Check max turns
            if self.turn_count >= self.config.max_turns:
                self.game_over = True
                info["max_turns_reached"] = True
                info["game_end_reason"] = f"Game ended after reaching maximum turns ({self.config.max_turns})"
                # Calculate final scores and add to info
                scores = [self._calculate_score(p) for p in range(self.config.num_players)]
                info["scores"] = scores
                # Calculate proper terminal reward
                reward = self._calculate_terminal_reward(info)
                return self._get_observation(), self._normalize_reward(reward, True), True, info
            
            if action == Action.DRAW_FROM_DECK:
                if not self.deck:
                    self._reshuffle_deck()
                self.drawn_card = self.deck.pop()
                self.drawn_from_discard = False
            else:  # Draw from discard
                self.drawn_card = self.discard_pile.pop()
                self.drawn_from_discard = True
        else:
            # Handle card replacement phase
            if Action.REPLACE_0 <= action <= Action.REPLACE_5:
                position = action - Action.REPLACE_0
                old_card = self.player_hands[self.current_player][position]
                
                # Store whether this position was previously revealed (for reward calculation)
                was_previously_revealed = position in self.revealed_cards[self.current_player]
                
                # Replace the card
                self.player_hands[self.current_player][position] = self.drawn_card
                self.discard_pile.append(old_card)
                
                # Add to revealed cards
                self.revealed_cards[self.current_player].add(position)
                
                # Store the information about whether this was a new reveal (for reward calculation)
                info["was_previously_revealed"] = was_previously_revealed
                
                # Check for final round trigger
                cards_per_player = self.config.grid_rows * self.config.grid_cols
                if (len(self.revealed_cards[self.current_player]) == cards_per_player 
                    and not self.final_round):
                    self.final_round = True
                    self.last_player = ((self.current_player + self.config.num_players - 1) 
                                      % self.config.num_players)
                    info["final_round"] = True
                    info["trigger_player"] = self.current_player
            else:  # Discard
                self.discard_pile.append(self.drawn_card)
            
            self.drawn_card = None
            
            # Check if game should end before advancing turn
            if self.final_round and self.current_player == self.last_player:
                self.game_over = True
                info["game_end_reason"] = "Game ended after final round completed"
                # Calculate final scores
                scores = [self._calculate_score(p) for p in range(self.config.num_players)]
                info["scores"] = scores
            
            self._advance_turn()
        
        # Calculate rewards
        reward = self._calculate_step_reward(action, info)
        
        # Return observation, reward, done flag, and info
        return (self._get_observation(), 
                self._normalize_reward(reward, self.game_over), 
                self.game_over, 
                info)
    
    def _reshuffle_deck(self) -> None:
        """Reshuffle the discard pile into the deck."""
        self.deck = self.discard_pile[:-1]
        self.discard_pile = [self.discard_pile[-1]]
        random.shuffle(self.deck)
    
    def _advance_turn(self) -> None:
        """Advance to the next player's turn."""
        self.current_player = (self.current_player + 1) % self.config.num_players
    
    def _calculate_step_reward(self, action: Action, info: Dict) -> float:
        """Calculate the reward for the current step."""
        if self.game_over:
            return self._calculate_terminal_reward(info)
        
        # For all actions, calculate the intermediate reward
        return self._calculate_intermediate_reward(action, info)
    
    def _calculate_terminal_reward(self, info: Dict) -> float:
        """Calculate terminal reward based on final scores."""
        # Use scores from info if available, otherwise calculate them
        if "scores" in info:
            scores = info["scores"]
        else:
            scores = [self._calculate_score(p) for p in range(self.config.num_players)]
            info["scores"] = scores
            
        min_score = min(scores)
        current_player_score = scores[self.current_player]
        
        # Count how many players have the minimum score (tied for first)
        num_tied_for_first = scores.count(min_score)
        
        if current_player_score == min_score:
            # If there's a tie for first place
            if num_tied_for_first > 1:
                # Reduced reward for ties - still positive but less than a clear win
                margin = sum(scores) / len(scores) - current_player_score
                return 2.5 + 0.3 * margin  # Half the normal winning reward
            else:
                # Clear winner gets full reward
                margin = sum(scores) / len(scores) - current_player_score
                return 5.0 + 0.3 * margin
        else:
            # Losing player's reward is based on how far they are from the minimum score
            return -0.3 * (current_player_score - min_score)
    
    def _calculate_intermediate_reward(self, action: Action, info: Dict) -> float:
        """Calculate intermediate reward based on action taken."""
        reward = 0.0
        
        if Action.REPLACE_0 <= action <= Action.REPLACE_5:
            position = action - Action.REPLACE_0
            old_card = self.discard_pile[-1]
            new_card = self.player_hands[self.current_player][position]
            
            # Reward for value improvement (increased weight)
            value_improvement = old_card.point_value - new_card.point_value
            if value_improvement > 0:
                reward += value_improvement  # Increased from 0.5 to 1.0
            
            # Reward for matches
            col = position % self.config.grid_cols
            other_position = (position + self.config.grid_cols) % (2 * self.config.grid_cols)
            other_card = self.player_hands[self.current_player][other_position]
            
            if (new_card == other_card and 
                position in self.revealed_cards[self.current_player] and 
                other_position in self.revealed_cards[self.current_player]):
                reward += 2.0  # Increased from 1.0 to 2.0
            
            # Reward for revealing cards - use the stored information about whether the position was previously revealed
            if not info.get("was_previously_revealed", True):
                reward += 0.2  # Increased from 0.1 to 0.2
        
        elif action == Action.DISCARD:
            # Penalty for discarding potential matches
            discarded_rank = self.discard_pile[-1]
            for col in range(self.config.grid_cols):
                top_pos = col
                bottom_pos = col + self.config.grid_cols
                if (top_pos in self.revealed_cards[self.current_player] and 
                    bottom_pos in self.revealed_cards[self.current_player]):
                    if (self.player_hands[self.current_player][top_pos] == discarded_rank or 
                        self.player_hands[self.current_player][bottom_pos] == discarded_rank):
                        reward -= 1.0  # Increased from 0.5 to 1.0
        
        # Reduced penalty based on visible score
        visible_score = self._calculate_visible_score()
        reward -= 0.005 * visible_score  # Reduced from 0.01 to 0.005
        
        return reward
    
    def _calculate_visible_score(self) -> int:
        """Calculate score only considering revealed cards."""
        score = 0
        hand = self.player_hands[self.current_player]
        
        for col in range(self.config.grid_cols):
            top_pos = col
            bottom_pos = col + self.config.grid_cols
            top_revealed = top_pos in self.revealed_cards[self.current_player]
            bottom_revealed = bottom_pos in self.revealed_cards[self.current_player]
            
            if top_revealed and bottom_revealed:
                if hand[top_pos] == hand[bottom_pos]:
                    if hand[top_pos] == Card.TWO:
                        score -= 4
                    continue
                score += hand[top_pos].point_value + hand[bottom_pos].point_value
            elif top_revealed:
                score += hand[top_pos].point_value
            elif bottom_revealed:
                score += hand[bottom_pos].point_value
        
        return score
    
    def render(self) -> str:
        """Return a string representation of the game state."""
        output = []
        output.append(f"Player {self.current_player}'s turn")
        
        for player in range(self.config.num_players):
            output.append(f"\nPlayer {player}'s hand:")
            hand = self.player_hands[player]
            revealed = self.revealed_cards[player]
            
            # Print top row
            row = []
            for i in range(self.config.grid_cols):
                if i in revealed:
                    row.append(Card(hand[i]).symbol)
                else:
                    row.append("#")
            output.append(" ".join(row))
            
            # Print bottom row
            row = []
            for i in range(self.config.grid_cols, 2 * self.config.grid_cols):
                if i in revealed:
                    row.append(Card(hand[i]).symbol)
                else:
                    row.append("#")
            output.append(" ".join(row))
            
            output.append(f"Cards revealed: {len(revealed)}/{self.config.grid_rows * self.config.grid_cols}")
            output.append(f"Score: {self._calculate_score(player)}")
        
        if self.discard_pile:
            output.append(f"\nDiscard pile top: {Card(self.discard_pile[-1]).symbol}")
        
        if self.drawn_card is not None:
            output.append(f"Drawn card: {Card(self.drawn_card).symbol}")
        
        if self.final_round:
            output.append("\nFinal round in progress!")
            
        output.append(f"\nDeck size: {len(self.deck)}")
        
        return "\n".join(output) 